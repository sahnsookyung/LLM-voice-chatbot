import Foundation
import AVFoundation

struct SpeechSettings: Codable, Equatable {
  var version = 1
  var recognition_mode = "accurate"
  var input_language = "auto"
  var reply_language = "auto"
  var vocabulary = ["Ari", "Codex"]
  var context_ms = 320
  var sensitivity = 0.7
  var pause_tolerance = 0.3
  var warm_up = true
  var fish_precision = 8
  var fish_streaming = true
  var dictionary: [String: Any] {
    (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(self))) as? [String: Any] ?? [:]
  }
}

struct MicrophoneFraming {
  var capture = UUID()
  var sequence: UInt32 = 0
  var offset: UInt64 = 0
  mutating func packet(_ pcm: Data) -> Data {
    var value = Data("LVM2".utf8)
    var id = capture.uuid
    withUnsafeBytes(of: &id) { value.append(contentsOf: $0) }
    var seq = sequence.littleEndian
    var position = offset.littleEndian
    withUnsafeBytes(of: &seq) { value.append(contentsOf: $0) }
    withUnsafeBytes(of: &position) { value.append(contentsOf: $0) }
    value.append(pcm)
    sequence += 1
    offset += UInt64(pcm.count / 2)
    return value
  }
}

// One consumer owns websocket sends. Memory is bounded; cancellation removes
// unsent microphone packets without waiting for a chain of per-packet Tasks.
@MainActor final class SpeechSender {
  private var queue: [(URLSessionWebSocketTask.Message, Int)] = []
  private var bytes = 0
  private var task: Task<Void, Never>?
  private var epoch = UUID()
  var onFailure: (() -> Void)?
  nonisolated static func discardOnCancellation(_ message: URLSessionWebSocketTask.Message, stopTask: Bool) -> Bool {
    if case .data = message { return true }
    guard case .string(let text) = message, let data = text.data(using: .utf8),
      let command = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let kind = command["type"] as? String else { return false }
    if ["start", "commit", "transcription_action"].contains(kind) { return true }
    if stopTask && kind == "text" { return true }
    if stopTask && kind == "codex", let action = (command["codex"] as? [String: Any])?["action"] as? String {
      return ["new", "select", "settings", "approval", "answer", "voice_select"].contains(action)
    }
    return false
  }
  func send(_ message: URLSessionWebSocketTask.Message, socket: URLSessionWebSocketTask, urgent: Bool = false, stopTask: Bool = false) {
    if urgent {
      queue.removeAll { Self.discardOnCancellation($0.0, stopTask: stopTask) }
      bytes = queue.reduce(0) { $0 + $1.1 }
    }
    let size: Int
    if case .data(let data) = message { size = data.count } else { size = 0 }
    guard bytes + size <= 340000, queue.count < 1024 else { reset(); onFailure?(); return }
    if urgent { queue.insert((message, size), at: 0) } else { queue.append((message, size)) }
    bytes += size
    guard task == nil else { return }
    let generation = epoch
    task = Task { [weak self] in
      guard let self else { return }
      defer { if self.epoch == generation { self.task = nil } }
      while !self.queue.isEmpty, !Task.isCancelled, self.epoch == generation {
        let (item, length) = self.queue.removeFirst()
        self.bytes -= length
        do { try await socket.send(item) }
        catch { self.reset(); self.onFailure?(); return }
      }
    }
  }
  func reset() {
    epoch = UUID()
    task?.cancel()
    task = nil
    queue.removeAll()
    bytes = 0
  }
}

// Tap callbacks convert and enqueue under a small lock. MainActor drains this
// mailbox, including the converter tail, before sending commit. No late Task
// can deliver bytes into another capture.
final class CapturePipe: @unchecked Sendable {
  private let lock = NSLock()
  private let converter: AVAudioConverter
  private var pending = Data()
  private var closed = false
  private var failed = false
  init(_ converter: AVAudioConverter) { self.converter = converter; converter.reset() }
  func accept(_ buffer: AVAudioPCMBuffer) {
    lock.lock(); defer { lock.unlock() }
    guard !closed else { return }
    convert(buffer)
  }
  private func convert(_ buffer: AVAudioPCMBuffer?) {
    var supplied = false
    repeat {
      let capacity = AVAudioFrameCount(buffer.map { Double($0.frameLength) * 16000 / converter.inputFormat.sampleRate + 256 } ?? 4096)
      guard let output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity) else { failed = true; return }
      var error: NSError?
      let status = converter.convert(to: output, error: &error) { _, state in
        if buffer == nil { state.pointee = .endOfStream; return nil }
        if supplied { state.pointee = .noDataNow; return nil }
        supplied = true; state.pointee = .haveData; return buffer
      }
      if status == .error { failed = true; return }
      if let samples = output.int16ChannelData?[0], output.frameLength > 0 {
        pending.append(Data(bytes: samples, count: Int(output.frameLength) * 2))
      }
      if pending.count > 320000 { failed = true; pending.removeAll(); closed = true; return }
      if buffer != nil || status == .endOfStream || output.frameLength == 0 { break }
    } while true
  }
  func drain(finish: Bool = false) -> (Data, Bool) {
    lock.lock(); defer { lock.unlock() }
    if finish && !closed { closed = true; convert(nil) }
    let data = pending
    pending.removeAll(keepingCapacity: !finish)
    return (data, failed)
  }
  func cancel() {
    lock.lock(); defer { lock.unlock() }
    closed = true; pending.removeAll()
  }
}
