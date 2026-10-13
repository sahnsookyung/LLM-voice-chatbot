import AVFoundation
import Foundation

@MainActor final class AudioController {
  private let engine = AVAudioEngine()
  private let player = AVAudioPlayerNode()
  private let outputFormat = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
  private var tapping = false
  private var captureGeneration = UUID()
  private var lastInputMeter = Date.distantPast
  private(set) var capturedFrames = 0
  private(set) var capturedPeak: Float = 0
  private(set) var captureFormat = ""
  private var generation = UUID()
  private var currentTurn: String?
  private var expectedSequence: UInt32 = 0
  private var expectedOffset: UInt64 = 0
  private var envelopes: [(Int64, Float)] = []
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
    NotificationCenter.default.addObserver(
      forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
    ) { [weak self] _ in
      Task { @MainActor in
        // Enabling voice processing also posts configuration notifications. If our
        // restart succeeded, no device recovery is needed.
        guard let self, !self.engine.isRunning, self.tapping || self.currentTurn != nil else {
          return
        }
        self.stopCapture()
        self.stopPlayback()
        self.onFailure?("The audio device changed. Select Talk to reconnect the microphone.")
      }
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
    engine.stop()
    try engine.inputNode.setVoiceProcessingEnabled(voiceProcessing)
    if voiceProcessing {
      engine.inputNode.isVoiceProcessingInputMuted = false
      engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
          enableAdvancedDucking: false, duckingLevel: .min)
    }
    let source = engine.inputNode.outputFormat(forBus: 0)
    let converter = try Self.inputConverter(from: source)
    let target = converter.outputFormat
    let capture = UUID()
    captureGeneration = capture
    capturedFrames = 0
    capturedPeak = 0
    captureFormat =
      "\(Int(source.sampleRate)) Hz · \(source.channelCount) channels · \(voiceProcessing ? "echo cancellation" : "direct input")"
    engine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: source) {
      [weak self] buffer, _ in
      let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000 / source.sampleRate + 16)
      guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
        return
      }
      var supplied = false
      var error: NSError?
      let result = converter.convert(to: converted, error: &error) { _, status in
        if supplied {
          status.pointee = .noDataNow
          return nil
        }
        supplied = true
        status.pointee = .haveData
        return buffer
      }
      if result == .error {
        Task { @MainActor in
          guard let self, self.tapping, self.captureGeneration == capture else { return }
          self.stopCapture()
          self.onFailure?(
            "Microphone audio could not be converted. Select Talk or Hands-free to reconnect.")
        }
        return
      }
      guard let samples = converted.int16ChannelData?[0],
        converted.frameLength > 0
      else { return }
      let bytes = Data(bytes: samples, count: Int(converted.frameLength) * 2)
      var square: Float = 0
      for i in 0..<Int(converted.frameLength) {
        let sample = Float(samples[i]) / 32768
        square += sample * sample
      }
      let level = min(1, sqrt(square / Float(converted.frameLength)) * 8)
      Task { @MainActor in
        guard let self, self.tapping, self.captureGeneration == capture else { return }
        self.capturedFrames += bytes.count / 2
        self.capturedPeak = max(self.capturedPeak, level)
        if Date().timeIntervalSince(self.lastInputMeter) >= 0.05 {
          self.lastInputMeter = Date()
          self.onInputLevel?(level)
        }
        self.onPCM?(bytes)
      }
    }
    tapping = true
    do { try engine.start() } catch {
      stopCapture()
      throw error
    }
  }
  func stopCapture() {
    guard tapping else { return }
    tapping = false
    captureGeneration = UUID()
    onInputLevel?(0)
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    try? engine.inputNode.setVoiceProcessingEnabled(false)
  }
  func beginTurn(_ turn: String) {
    stopPlayback()
    currentTurn = turn
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
    for i in stride(from: 0, to: count, by: 800) {
      var square: Float = 0
      let end = min(i + 800, count)
      for j in i..<end { square += output[j] * output[j] }
      envelopes.append((Int64(offset) + Int64(i), min(1, sqrt(square / Float(end - i)) * 5)))
    }
    expectedSequence += 1
    expectedOffset += UInt64(count)
    if !engine.isRunning { try engine.start() }
    let epoch = generation
    let end = expectedOffset
    player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
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
    envelopes = []
    expressions = []
    pendingGesture = nil
    onGesture?("cancel")
    onLevel?(0)
    if !tapping { engine.stop() }
  }
  private func meter() {
    guard player.isPlaying, let render = player.lastRenderTime,
      let time = player.playerTime(forNodeTime: render)
    else { return }
    let sample = Int64(Double(time.sampleTime) * 24000 / time.sampleRate)
    if sample > 0, let turn = currentTurn { playbackStarted(turn: turn) }
    while let expression = expressions.first, expression.0 <= sample {
      expressions.removeFirst()
      onExpression?(expression.1)
    }
    while envelopes.count > 1 && envelopes[1].0 <= sample { envelopes.removeFirst() }
    onLevel?(envelopes.first.map { sample - $0.0 < 1600 ? $0.1 : 0 } ?? 0)
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
  }
}
