import AVFoundation
import Foundation

// Wire offsets count speech samples only. The player's clock also counts empty
// time while the next TTS phrase is being generated. Schedule audio and its mouth
// envelopes on the same player timeline, retaining wire offsets for expressions.
struct SpeechPlaybackTimeline {
  private struct Envelope {
    let frames: Range<Int64>
    let offset: UInt64
    let level: Float
  }
  private var end: Int64 = 0
  private var envelopes: [Envelope] = []

  mutating func schedule(
    samples: UnsafeBufferPointer<Float>, offset: UInt64, playerSample: Int64?
  ) -> Int64 {
    // Leave one meter interval (33 ms) for a late buffer to reach the render
    // thread. Already-buffered speech stays contiguous, with no added delay.
    let start = playerSample.map { max(end, $0 + 800) } ?? end
    end = start + Int64(samples.count)
    for i in stride(from: 0, to: samples.count, by: 800) {
      let limit = min(i + 800, samples.count)
      var square: Float = 0
      for j in i..<limit { square += samples[j] * samples[j] }
      envelopes.append(Envelope(
        frames: (start + Int64(i))..<(start + Int64(limit)),
        offset: offset + UInt64(i),
        level: min(1, sqrt(square / Float(limit - i)) * 5)))
    }
    return start
  }

  mutating func position(at sample: Int64) -> (offset: UInt64, level: Float)? {
    while let first = envelopes.first, first.frames.upperBound <= sample {
      envelopes.removeFirst()
    }
    guard let first = envelopes.first, first.frames.contains(sample) else { return nil }
    return (first.offset + UInt64(sample - first.frames.lowerBound), first.level)
  }
}

@MainActor final class AudioController {
  private let engine = AVAudioEngine()
  private let captureEngine = AVAudioEngine()
  private var capturePipe: CapturePipe?
  private var cachedConverter: AVAudioConverter?
  private var captureTimer: Timer?
  var onPlaybackStarted: (() -> Void)?
  var onPlaybackGap: ((Double) -> Void)?
  private var hasStartedPlayback = false
  private var gapStarted: TimeInterval?
  private let player = AVAudioPlayerNode()
  private let outputFormat = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
  private var tapping = false
  private var deviceObservers: [NSObjectProtocol] = []
  private(set) var capturedFrames = 0
  private(set) var capturedPeak: Float = 0
  private(set) var captureFormat = ""
  private var generation = UUID()
  private var currentTurn: String?
  private var expectedSequence: UInt32 = 0
  private var expectedOffset: UInt64 = 0
  private var playback = SpeechPlaybackTimeline()
  private var expressions: [(Int64, String)] = []
  private var pendingGesture: String?
  private var timer: Timer?
  var onPCM: ((Data) -> Void)?
  var onAck: ((String, UInt64) -> Void)?
  var onLevel: ((Float) -> Void)?
  var onInputLevel: ((Float) -> Void)?
  var onExpression: ((String) -> Void)?
  var onGesture: ((String) -> Void)?
  var onFailure: ((String) -> Void)?
  init() {
    engine.attach(player)
    engine.connect(player, to: engine.mainMixerNode, format: outputFormat)
    timer = Timer.scheduledTimer(withTimeInterval: 1 / 30, repeats: true) { [weak self] _ in
      Task { @MainActor in self?.meter() }
    }
    for observedEngine in [engine, captureEngine] {
      let isCapture = observedEngine === captureEngine
      deviceObservers.append(NotificationCenter.default.addObserver(
        forName: .AVAudioEngineConfigurationChange, object: observedEngine, queue: .main
      ) { [weak self] _ in
        Task { @MainActor in
          // Voice-processing setup also posts this event. A successful restart
          // requires no recovery; an actual device loss cancels this capture.
          guard let self, self.tapping || self.currentTurn != nil else { return }
          guard !(isCapture ? self.captureEngine : self.engine).isRunning else { return }
          self.cachedConverter = nil
          self.stopCapture()
          self.stopPlayback()
          self.onFailure?("The audio device changed. Select Talk to reconnect the microphone.")
        }
      })
    }
  }
  static func inputConverter(from source: AVAudioFormat) throws -> AVAudioConverter {
    guard source.sampleRate > 0, source.channelCount > 0,
      let target = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true),
      let converter = AVAudioConverter(from: source, to: target)
    else {
      throw NSError(
        domain: "Audio", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "No usable microphone is connected."])
    }
    // VoiceProcessingIO adds discrete reference channels. Its default N-to-mono
    // downmix can produce silence; channel 0 carries the processed microphone.
    converter.channelMap = [0]
    return converter
  }
  func startCapture(voiceProcessing: Bool) throws {
    guard !tapping else { return }
    captureEngine.stop()
    if captureEngine.inputNode.isVoiceProcessingEnabled != voiceProcessing {
      try captureEngine.inputNode.setVoiceProcessingEnabled(voiceProcessing)
    }
    if voiceProcessing {
      captureEngine.inputNode.isVoiceProcessingInputMuted = false
      captureEngine.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
    }
    let source = captureEngine.inputNode.outputFormat(forBus: 0)
    if cachedConverter?.inputFormat != source { cachedConverter = try Self.inputConverter(from: source) }
    let pipe = CapturePipe(cachedConverter!)
    capturePipe = pipe
    capturedFrames = 0
    capturedPeak = 0
    captureFormat = "\(Int(source.sampleRate)) Hz · \(source.channelCount) channels · \(voiceProcessing ? "echo cancellation" : "direct input")"
    captureEngine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: source) { buffer, _ in
      pipe.accept(buffer)
    }
    tapping = true
    captureTimer = Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { [weak self] _ in
      Task { @MainActor in self?.drainCapture() }
    }
    do { try captureEngine.start() } catch { stopCapture(); throw error }
  }
  private func drainCapture(finish: Bool = false) {
    guard let pipe = capturePipe else { return }
    let (bytes, failed) = pipe.drain(finish: finish)
    if failed { stopCapture(); onFailure?("Audio capture fell behind or conversion failed. Please record again."); return }
    guard !bytes.isEmpty else { return }
    var square: Float = 0
    bytes.withUnsafeBytes { raw in
      for i in stride(from: 0, to: bytes.count, by: 2) {
        let sample = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i, as: Int16.self))) / 32768
        square += sample * sample
      }
    }
    let level = min(1, sqrt(square / Float(bytes.count / 2)) * 8)
    capturedFrames += bytes.count / 2
    capturedPeak = max(capturedPeak, level)
    onInputLevel?(level)
    for start in stride(from: 0, to: bytes.count, by: 2048) {
      onPCM?(bytes.subdata(in: start..<min(start + 2048, bytes.count)))
    }
  }
  func finishCapture() {
    guard tapping else { return }
    captureEngine.inputNode.removeTap(onBus: 0)
    captureEngine.stop()
    captureTimer?.invalidate(); captureTimer = nil
    drainCapture(finish: true)
    capturePipe = nil
    tapping = false
    onInputLevel?(0)
  }
  func stopCapture() {
    capturePipe?.cancel(); capturePipe = nil
    captureTimer?.invalidate(); captureTimer = nil
    guard tapping else { return }
    tapping = false
    onInputLevel?(0)
    captureEngine.inputNode.removeTap(onBus: 0)
    captureEngine.stop()
  }
  func beginTurn(_ turn: String) {
    stopPlayback()
    currentTurn = turn
    hasStartedPlayback = false
    gapStarted = nil
    expectedSequence = 0
    expectedOffset = 0
    // Do not keep a previous smile or frown while a new reply's mood is pending.
    onExpression?("neutral")
  }
  func accept(_ packet: Data) throws {
    guard packet.count >= 32, String(data: packet.prefix(4), encoding: .utf8) == "LVA1",
      (packet.count - 32) % 2 == 0
    else { return }
    let hex = packet[4..<20].map { String(format: "%02x", $0) }.joined()
    let turn =
      "\(hex.prefix(8))-\(hex.dropFirst(8).prefix(4))-\(hex.dropFirst(12).prefix(4))-\(hex.dropFirst(16).prefix(4))-\(hex.suffix(12))"
    guard turn == currentTurn else { return }
    let sequence = packet.withUnsafeBytes {
      UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 20, as: UInt32.self))
    }
    let offset = packet.withUnsafeBytes {
      UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 24, as: UInt64.self))
    }
    guard sequence == expectedSequence, offset == expectedOffset else {
      stopPlayback()
      throw NSError(
        domain: "Audio", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "Audio arrived out of order. Playback stopped."])
    }
    let count = (packet.count - 32) / 2
    guard
      let buffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(count)),
      let output = buffer.floatChannelData?[0]
    else { return }
    buffer.frameLength = AVAudioFrameCount(count)
    packet.withUnsafeBytes { bytes in
      for i in 0..<count {
        output[i] =
          Float(
            Int16(littleEndian: bytes.loadUnaligned(fromByteOffset: 32 + i * 2, as: Int16.self)))
          / 32768
      }
    }
    expectedSequence += 1
    expectedOffset += UInt64(count)
    if !engine.isRunning { try engine.start() }
    let start = playback.schedule(
      samples: UnsafeBufferPointer(start: output, count: count), offset: offset,
      playerSample: playbackSample)
    let epoch = generation
    let end = expectedOffset
    player.scheduleBuffer(
      buffer, at: AVAudioTime(sampleTime: start, atRate: 24000), options: [],
      completionCallbackType: .dataPlayedBack
    ) { [weak self] _ in
      Task { @MainActor in
        guard let self, self.generation == epoch, self.currentTurn == turn else { return }
        self.onAck?(turn, end)
      }
    }
    if !player.isPlaying { player.play() }
  }
  func stopPlayback() {
    generation = UUID()
    player.stop()
    currentTurn = nil
    playback = SpeechPlaybackTimeline()
    expressions = []
    pendingGesture = nil
    onGesture?("cancel")
    onLevel?(0)
    if !tapping { engine.stop() }
  }
  private var playbackSample: Int64? {
    guard player.isPlaying, let render = player.lastRenderTime,
      let time = player.playerTime(forNodeTime: render)
    else { return nil }
    return Int64(Double(time.sampleTime) * 24000 / time.sampleRate)
  }
  private func meter() {
    guard let sample = playbackSample else { return }
    guard let position = playback.position(at: sample) else {
      onLevel?(0)
      if hasStartedPlayback && gapStarted == nil { gapStarted = ProcessInfo.processInfo.systemUptime }
      return
    }
    if !hasStartedPlayback { hasStartedPlayback = true; onPlaybackStarted?() }
    if let gap = gapStarted { onPlaybackGap?(ProcessInfo.processInfo.systemUptime - gap); gapStarted = nil }
    if let turn = currentTurn { playbackStarted(turn: turn) }
    while let expression = expressions.first, expression.0 <= Int64(position.offset) {
      expressions.removeFirst()
      onExpression?(expression.1)
    }
    onLevel?(position.level)
  }
  func queueExpression(turn: String, offset: UInt64, expression: String) {
    guard turn == currentTurn else { return }
    if offset == 0 {
      onExpression?(expression)
    } else {
      expressions.append((Int64(offset), expression))
    }
  }
  func applyGesture(turn: String, gesture: String, waitForAudio: Bool = false) {
    guard turn == currentTurn,
      ["wave", "explain", "nod", "shrug", "shift", "step", "twirl", "raise-arms", "point"].contains(gesture)
    else { return }
    if waitForAudio { pendingGesture = gesture } else { onGesture?(gesture) }
  }
  func playbackStarted(turn: String) {
    guard turn == currentTurn, let gesture = pendingGesture else { return }
    pendingGesture = nil
    onGesture?(gesture)
  }
  func close() {
    stopCapture()
    stopPlayback()
    timer?.invalidate()
    for observer in deviceObservers { NotificationCenter.default.removeObserver(observer) }
    deviceObservers.removeAll()
  }
}
