// Compile with native/Sources/Companion/{Audio,SpeechTransport}.swift. Plays quiet synthetic pulses;
// no microphone or model is opened. Checks real AVAudioPlayerNode underruns.
import AppKit
import AVFoundation
import WebKit

@MainActor final class LipSyncCheck: NSObject, WKNavigationDelegate {
  let audio = AudioController()
  var turn = UUID()
  var started = Date()
  var levels: [(Double, Float)] = []
  var acknowledgments: [UInt64] = []
  var sequence: UInt32 = 0
  var offset: UInt64 = 0
  var web: WKWebView?
  var window: NSWindow?
  var jaws: [(Double, Double)] = []
  var expressionTimes: [(Double, String)] = []
  var measuring = false

  func packet(samples: Int) -> Data {
    var data = Data("LVA1".utf8)
    var id = turn.uuid
    withUnsafeBytes(of: &id) { data.append(contentsOf: $0) }
    var seq = sequence.littleEndian, position = offset.littleEndian
    withUnsafeBytes(of: &seq) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: &position) { data.append(contentsOf: $0) }
    for i in 0..<samples {
      let frame = Int(offset) + i
      let amplitude = (frame / 2400) % 2 == 0 ? 0.12 : 0.0
      var value = Int16(sin(Double(frame) * 2 * .pi * 330 / 24000) * amplitude * 32767).littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    sequence += 1; offset += UInt64(samples)
    return data
  }

  func burst() {
    do {
      for count in [7680, 7680, 3840] { try audio.accept(packet(samples: count)) }
    } catch { fail(error.localizedDescription) }
  }

  func fail(_ message: String) -> Never {
    audio.close()
    print("FAIL: \(message)")
    exit(1)
  }

  func run() {
    // An optional renderer/index.html argument checks the real WKWebView/VRM
    // morph as well as AudioController's meter. Uses the app's same JS bridge.
    if CommandLine.arguments.count > 1 {
      let config = WKWebViewConfiguration()
      config.websiteDataStore = .nonPersistent()
      let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 300, height: 540), configuration: config)
      self.web = web; web.navigationDelegate = self
      window = NSWindow(contentRect: web.frame, styleMask: [.titled], backing: .buffered, defer: false)
      window?.title = "Speech and mouth synchronization test"
      window?.contentView = web; window?.orderFrontRegardless()
      let url = URL(fileURLWithPath: CommandLine.arguments[1])
      web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
      DispatchQueue.main.asyncAfter(deadline: .now() + 45) { self.fail("renderer timed out") }
    } else { beginPlayback() }
  }

  func command(_ message: String) {
    web?.evaluateJavaScript("window.companion(\(message))") { _, error in
      if let error { self.fail(error.localizedDescription) }
    }
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    command("{visible:true,visualMode:'model',framing:'portrait',expression:'happy'}")
    pollRenderer()
  }

  func pollRenderer() {
    web?.evaluateJavaScript("JSON.stringify(window.characterStatus())") { value, error in
      guard error == nil, let json = value as? String, let data = json.data(using: .utf8),
        let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      else { self.fail("renderer status unavailable: \(String(describing: error))") }
      if let error = status["modelError"] as? String, !error.isEmpty { self.fail(error) }
      if self.measuring {
        let weights = status["expressionWeights"] as? [String: Double]
        if let jaw = weights?["aa"] { self.jaws.append((Date().timeIntervalSince(self.started), jaw)) }
      } else if status["modelLoading"] as? Bool == false, (status["frames"] as? Int ?? 0) > 10 {
        self.beginPlayback()
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + 1 / 30) { self.pollRenderer() }
    }
  }

  func beginPlayback() {
    measuring = true
    audio.onLevel = { [weak self] level in
      guard let self else { return }
      self.levels.append((Date().timeIntervalSince(self.started), level))
      self.command("{level:\(level)}")
    }
    audio.onExpression = { [weak self] expression in
      guard let self else { return }
      self.expressionTimes.append((Date().timeIntervalSince(self.started), expression))
      self.command("{expression:'\(expression)'}")
    }
    audio.onAck = { [weak self] _, offset in self?.acknowledgments.append(offset) }
    audio.beginTurn(turn.uuidString.lowercased())
    audio.queueExpression(turn: turn.uuidString.lowercased(), offset: 0, expression: "happy")
    audio.queueExpression(turn: turn.uuidString.lowercased(), offset: 19200, expression: "concerned")
    started = Date(); burst()
    DispatchQueue.main.asyncAfter(deadline: .now() + 2.4) { self.burst() }
    DispatchQueue.main.asyncAfter(deadline: .now() + 4.2) { self.check() }
  }

  func check() {
    let first = levels.filter { (0.12...0.75).contains($0.0) }
    let gap = levels.filter { (1.2...2.2).contains($0.0) }
    let second = levels.filter { (2.52...3.15).contains($0.0) }
    let ending = levels.filter { (3.8...4.2).contains($0.0) }
    let firstJaw = jaws.filter { (0.12...0.75).contains($0.0) }.map { $0.1 }
    let secondJaw = jaws.filter { (2.52...3.15).contains($0.0) }.map { $0.1 }
    let gapJaw = jaws.filter { (1.3...2.2).contains($0.0) }.map { $0.1 }
    // The renderer deliberately eases the jaw closed; allow that settling time.
    let endJaw = jaws.filter { (3.9...4.2).contains($0.0) }.map { $0.1 }
    let expressionAt = expressionTimes.first { $0.1 == "concerned" }?.0 ?? -1
    var stats: [String: Any] = [
      "first_peak": first.map { $0.1 }.max() ?? -1,
      "gap_peak": gap.map { $0.1 }.max() ?? -1,
      "second_peak": second.map { $0.1 }.max() ?? -1,
      "ending_peak": ending.map { $0.1 }.max() ?? -1,
      "first_samples": first.count, "second_samples": second.count,
      "acknowledged_samples": acknowledgments.last ?? 0,
      "second_expression_seconds": expressionAt,
    ]
    if web != nil {
      stats["first_jaw_peak"] = firstJaw.max() ?? -1
      stats["second_jaw_peak"] = secondJaw.max() ?? -1
      stats["gap_jaw_peak"] = gapJaw.max() ?? -1
      stats["ending_jaw_peak"] = endJaw.max() ?? -1
    }
    let data = try! JSONSerialization.data(withJSONObject: stats, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
    guard first.contains(where: { $0.1 > 0.1 }) else { fail("first burst did not move the mouth") }
    guard second.contains(where: { $0.1 > 0.1 }) else { fail("mouth stayed closed after a synthesis gap") }
    guard gap.allSatisfy({ $0.1 == 0 }), ending.allSatisfy({ $0.1 == 0 }) else { fail("mouth moved during silence") }
    guard acknowledgments == [7680, 15360, 19200, 26880, 34560, 38400]
    else { fail("playback acknowledgments changed") }
    guard (2.4...2.65).contains(expressionAt) else { fail("second expression did not wait for its phrase") }
    for burst in [first, second] {
      let openings = zip(burst, burst.dropFirst()).filter { $0.1 == 0 && $1.1 > 0.1 }.count
      guard openings >= 2 else { fail("mouth did not keep opening and closing within a phrase") }
    }
    if web != nil {
      for burst in [firstJaw, secondJaw] {
        guard burst.count >= 5, (burst.max() ?? 0) > 0.15,
          (burst.max() ?? 0) - (burst.min() ?? 0) > 0.08
        else { fail("rendered mouth did not follow the changing audio") }
      }
      guard !gapJaw.isEmpty, !endJaw.isEmpty, gapJaw.allSatisfy({ $0 < 0.015 }),
        endJaw.allSatisfy({ $0 < 0.015 }) else { fail("rendered mouth stayed open during silence") }
    }
    print("PASS: mouth follows both audio bursts, closes during the gap, and preserves acknowledgments")
    checkInterruption()
  }

  func restart() {
    turn = UUID(); sequence = 0; offset = 0
    levels = []; jaws = []; acknowledgments = []
    started = Date()
    audio.beginTurn(turn.uuidString.lowercased())
    burst()
  }

  func checkInterruption() {
    restart()
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) {
      self.audio.stopPlayback()
      do { try self.audio.accept(self.packet(samples: 7680)) }
      catch { self.fail(error.localizedDescription) }
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
      guard self.acknowledgments.isEmpty, self.levels.last?.1 == 0,
        self.web == nil || (self.jaws.last?.1 ?? 1) < 0.015
      else { self.fail("interrupted speech left mouth movement or stale acknowledgments") }
      self.restart()
      DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) {
        guard self.levels.contains(where: { $0.1 > 0.1 }), self.levels.last?.1 == 0,
          self.acknowledgments == [7680, 15360, 19200],
          self.web == nil || self.jaws.contains(where: { $0.1 > 0.15 })
        else { self.fail("new turn did not recover after interruption") }
        self.audio.close()
        print("PASS: interruption closes the mouth, rejects stale packets, and the next turn animates")
        exit(0)
      }
    }
  }
}

@main @MainActor struct CheckLipSync {
  static func main() {
    let app = NSApplication.shared
    app.setActivationPolicy(CommandLine.arguments.count > 1 ? .accessory : .prohibited)
    let check = LipSyncCheck()
    check.run()
    app.run()
  }
}
