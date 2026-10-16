// JSON-lines probe for real local playback. No microphone is opened.
import AppKit
import Foundation

@MainActor final class PlaybackProbe {
  let audio = AudioController()
  var released: TimeInterval?
  var heardSignal = false
  func emit(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([10]))
  }
  func run() {
    audio.onAck = { [weak self] turn, samples in self?.emit(["type": "ack", "turn": turn, "samples": samples]) }
    audio.onPlaybackStarted = { [weak self] in
      guard let self else { return }
      if let release = released { emit(["type": "audible", "seconds": ProcessInfo.processInfo.systemUptime - release]) }
    }
    audio.onPlaybackGap = { [weak self] seconds in self?.emit(["type": "gap", "seconds": seconds]) }
    audio.onLevel = { [weak self] level in
      guard let self, let release = released, level > 0.01, !heardSignal else { return }
      heardSignal = true
      emit(["type": "non_silent_playback", "seconds": ProcessInfo.processInfo.systemUptime - release,
            "level": level])
    }
    DispatchQueue.global().async {
      while let line = readLine() {
        guard let data = line.data(using: .utf8), let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
        DispatchQueue.main.async { self.command(value) }
      }
      DispatchQueue.main.async { self.audio.close(); exit(0) }
    }
  }
  func command(_ value: [String: Any]) {
    do {
      switch value["type"] as? String {
      case "release": released = ProcessInfo.processInfo.systemUptime; heardSignal = false
      case "begin": audio.beginTurn(value["turn"] as! String)
      case "audio": try audio.accept(Data(base64Encoded: value["data"] as! String)!)
      case "cancel":
        let start = ProcessInfo.processInfo.systemUptime
        audio.stopPlayback()
        emit(["type": "cutoff_requested", "seconds": ProcessInfo.processInfo.systemUptime - start])
      case "quit": audio.close(); exit(0)
      default: break
      }
    } catch { emit(["type": "error", "message": "Native playback failed"]); audio.close(); exit(1) }
  }
}

@main struct Main {
  @MainActor static func main() {
    let probe = PlaybackProbe()
    probe.run()
    RunLoop.main.run()
  }
}
