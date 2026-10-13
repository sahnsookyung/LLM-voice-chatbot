import AppKit
import WebKit

// Test the packaged WKWebView file origin and bridge, including GPU lifecycle.
final class RendererCheck: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
  var web: WKWebView!
  var window: NSWindow!
  var phase = 0
  var pausedFrames = 0
  var start = Date()
  var studio = false
  var hairPeak = 0.0
  var hairSpeedPeak = 0.0
  let gestures = ["twirl", "raise-arms", "shrug", "point"]
  var gestureIndex = 0
  var posePeak = 0.0
  var captured = false
  var snapshots = 0
  var gestureFrames = 0
  var radiusPeak = 0.0
  var slowRadius = 0.0
  var fastTurn = false
  let portraitOnly = CommandLine.arguments.last == "--portrait-only"
  func run(_ path: String) {
    studio = path.hasSuffix("studio.html")
    let controller = WKUserContentController()
    controller.add(self, name: "character")
    let configuration = WKWebViewConfiguration()
    configuration.userContentController = controller
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: studio ? 300 : 700, height: studio ? 540 : 1100), configuration: configuration)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = web
    window.orderFrontRegardless()
    let url = URL(fileURLWithPath: path)
    web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now() + 90) {
      self.web.evaluateJavaScript("JSON.stringify(window.characterStatus())") { result, _ in
        self.fail("renderer check timed out in phase \(self.phase): \(result ?? "no status")")
      }
    }
  }
  func fail(_ message: String) -> Never { print("FAIL:", message); exit(1) }
  func command(_ js: String) { web.evaluateJavaScript(js) { _, error in if let error { self.fail(error.localizedDescription) } } }
  func capture(_ gesture: String) {
    guard !studio, CommandLine.arguments.count >= 3 else { return }
    snapshots += 1
    web.takeSnapshot(with: nil) { image, error in
      defer { self.snapshots -= 1 }
      guard error == nil, let tiff = image?.tiffRepresentation,
            let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
      else { self.fail("gesture snapshot failed") }
      do {
        let directory = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try png.write(to: directory.appendingPathComponent("ari-v9-\(gesture).png"))
      } catch { self.fail(error.localizedDescription) }
    }
  }
  func poll() {
    web.evaluateJavaScript("JSON.stringify(window.characterStatus())") { value, error in
      guard error == nil, let text = value as? String,
        let data = text.data(using: .utf8),
        let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      else { self.fail("bridge status unavailable") }
      let ready = status["modelReady"] as? Bool ?? false
      switch self.phase {
      case 0:
        if status["loaded"] as? Bool == true {
          print("PASS: bundled full-body illustration loaded")
          self.phase = 1
          self.command("window.companion({visualMode:'model',framing:'full',visible:true,expression:'happy'})")
        }
      case 1:
        if ready, let calls = status["drawCalls"] as? Int, calls > 0 {
          guard status["bones"] as? Int == 54 else { self.fail("skeleton missing") }
          guard status["modelVersion"] as? String == "v9" else { self.fail("packaged model is stale") }
          print("PASS: bundled VRM rendered: \(calls) draws, \(status["triangles"] ?? "?") triangles")
          if self.portraitOnly {
            self.phase = 8; self.start = Date()
            self.command("window.companion({framing:'portrait',angle:45,reducedMotion:true})")
            break
          }
          // Capture the baseline in the same JS task that hides the model;
          // the status read above may precede an in-flight animation frame.
          self.web.evaluateJavaScript("window.companion({visible:false}); window.characterStatus().frames") { value, error in
            guard error == nil, let frames = value as? Int else { self.fail("hide acknowledgement missing") }
            self.pausedFrames = frames
            self.start = Date()
            self.phase = 2
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.poll() }
          }
          return
        }
      case 2:
        if Date().timeIntervalSince(self.start) > 0.4 {
          guard status["frames"] as? Int == self.pausedFrames else { self.fail("hidden model kept rendering") }
          print("PASS: hidden character stops rendering")
          self.phase = 3
          self.command("window.companion({visible:true,framing:'portrait',expression:'embarrassed',gesture:'step',level:0.4})")
        }
      case 3:
        if (status["frames"] as? Int ?? 0) > self.pausedFrames + 4 {
          guard (status["footError"] as? Double ?? 1) < 0.001 else { self.fail("feet drifted from their targets") }
          guard (status["blush"] as? Double ?? 0) > 0 else { self.fail("expression blush missing") }
          let weights = status["expressionWeights"] as? [String: Double] ?? [:]
          guard ["ariHappy", "ariCurious", "ariConcerned", "ariAnnoyed", "ariEmbarrassed"].allSatisfy({ weights[$0] != nil }),
                (weights["ariEmbarrassed"] ?? 0) > 0, (weights["aa"] ?? 0) > 0
          else { self.fail("custom facial controls or concurrent speech missing: \(weights)") }
          print("PASS: portrait expression, blush and planted feet (error \(status["footError"] ?? "?"))")
          self.phase = 31
          self.start = Date()
          self.command("window.companion({expression:'concerned',gesture:'cancel',level:0})")
        }
      case 31, 32:
        if Date().timeIntervalSince(self.start) > 1.8 {
          let weights = status["expressionWeights"] as? [String: Double] ?? [:]
          let target = self.phase == 31 ? "ariConcerned" : "ariAnnoyed"
          let previous = self.phase == 31 ? "ariEmbarrassed" : "ariConcerned"
          guard (weights[target] ?? 0) > 0.75, (weights["ariHappy"] ?? 1) < 0.001,
                (weights[previous] ?? 1) < 0.001
          else { self.fail("serious expression retained the previous face: \(weights)") }
          print("PASS: \(target) replaces the previous face without retaining a smile")
          self.start = Date()
          if self.phase == 31 {
            self.phase = 32; self.command("window.companion({expression:'annoyed'})")
          } else {
            self.phase = 4; self.command("window.companion({expression:'neutral',gesture:'hair-sway',level:0})")
          }
        }
      case 4:
        guard (status["hairJoints"] as? Int ?? 0) > 20,
              let hair = status["hairDeflection"] as? Double, hair.isFinite, hair < 1
        else { self.fail("missing or unstable hair dynamics") }
        self.hairPeak = max(self.hairPeak, hair)
        self.hairSpeedPeak = max(self.hairSpeedPeak,status["hairSpeed"] as? Double ?? 0)
        if Date().timeIntervalSince(self.start) > 8 {
          guard self.hairPeak > 0.01, self.hairSpeedPeak > 0.05,
                (status["hairSpeed"] as? Double ?? 1) < self.hairSpeedPeak * 0.35 + 0.01,
                status["hairContactVolumes"] as? Int == 9
          else { self.fail("hair did not move and settle: peak \(self.hairPeak), final \(hair)") }
          print("PASS: nine hair contact volumes; angular speed settles from \(self.hairSpeedPeak) to \(status["hairSpeed"] ?? "?") rad/s")
          self.capture("portrait")
          self.phase = 5
          self.gestureFrames = status["frames"] as? Int ?? 0
          self.start = Date()
          self.command("window.companion({framing:'full',gesture:'twirl'})")
        }
      case 5:
        let gesture = self.gestures[self.gestureIndex]
        let elapsed = Date().timeIntervalSince(self.start)
        if gesture == "raise-arms", elapsed > 1.8, elapsed < 2.5 {
          guard let correction = status["garmentCorrectives"] as? [String: Any],
            (correction["meshes"] as? Int ?? 0) > 0,
            let weights = correction["weights"] as? [Double], weights.count == 2,
            weights.allSatisfy({ $0 > 0.8 && $0 <= 1 })
          else { self.fail("raised arms did not engage garment fold corrections") }
        }
        guard (status["footError"] as? Double ?? 1) < 0.001,
              (status["limbExtent"] as? Double ?? 2) < 0.97,
              let hair = status["hairDeflection"] as? Double, hair.isFinite, hair < 1.6
        else { self.fail("unstable hair/feet or cropped gesture during \(gesture): \(status["limbExtent"] ?? "?")") }
        let metric = gesture == "twirl" ? (status["bodyYaw"] as? Double ?? 0)
          : gesture == "raise-arms" ? (status["rightHandHeight"] as? Double ?? 0)
          : gesture == "shrug" ? (status["shoulderAngle"] as? Double ?? 0)
          : (status["rightMiddleCurl"] as? Double ?? 0) - (status["rightIndexCurl"] as? Double ?? 0)
        self.posePeak = max(self.posePeak,metric)
        if elapsed > 2.3 && !self.captured { self.captured = true; self.capture(gesture) }
        if elapsed > 6.2 {
          let minimum = gesture == "twirl" ? 6.2 : gesture == "raise-arms" ? 1.5 : gesture == "shrug" ? 0.14 : 0.6
          guard self.posePeak > minimum else { self.fail("\(gesture) pose missing: \(self.posePeak)") }
          let fps = Double((status["frames"] as? Int ?? 0) - self.gestureFrames) / elapsed
          print("PASS: \(gesture) pose (peak \(self.posePeak)), planted feet, finite hair, \(fps) rendered FPS")
          self.gestureFrames = status["frames"] as? Int ?? 0
          self.gestureIndex += 1; self.posePeak = 0; self.captured = false; self.start = Date()
          if self.gestureIndex < self.gestures.count {
            self.command("window.companion({gesture:'\(self.gestures[self.gestureIndex])'})")
          } else {
            self.phase = 7
            self.radiusPeak = 0
            self.command("window.companion({expression:'neutral',gestureSpeed:0.65,gesture:'twirl'})")
          }
        }
      case 7:
        guard status["physicsHz"] as? Int == 60, (status["hairTips"] as? Int ?? 0) >= 8,
              let radius = status["hairTipRadius"] as? Double, radius.isFinite, radius < 0.65,
              (status["footError"] as? Double ?? 1) < 0.001
        else { self.fail("fixed-step hair response or contact unstable") }
        self.radiusPeak = max(self.radiusPeak,radius)
        if Date().timeIntervalSince(self.start) > (self.fastTurn ? 5.5 : 9.5) {
          if !self.fastTurn {
            self.slowRadius = self.radiusPeak; self.radiusPeak = 0; self.fastTurn = true; self.start = Date()
            self.command("window.companion({gestureSpeed:1.5,gesture:'twirl'})")
          } else {
            guard self.radiusPeak > self.slowRadius + 0.005 else { self.fail("fast twirl did not swing farther: slow \(self.slowRadius), fast \(self.radiusPeak)") }
            print("PASS: speed-dependent hair radius: slow \(self.slowRadius), fast \(self.radiusPeak) metres")
            print("TIMING: p95 simulation \(status["simulationMsP95"] ?? "?") ms per 60 Hz step; draw submission \(status["drawSubmissionMsP95"] ?? "?") ms")
            self.phase = 8; self.start = Date()
            self.command("window.companion({framing:'portrait',angle:45,reducedMotion:true,gestureSpeed:1,gesture:'wave',level:0.4})")
          }
        }
      case 8:
        if Date().timeIntervalSince(self.start) > 2.5 {
          guard (status["hairSpeed"] as? Double ?? 1) < 0.0001 else { self.fail("reduced-motion hair still moving: \(status["hairSpeed"] ?? "?") rad/s") }
          print("PASS: reduced motion stops hair dynamics")
          self.capture("three-quarter")
          self.phase = 6
          self.command("window.companion({visualMode:'illustration'})")
        }
      case 6:
        if self.snapshots > 0 { break }
        if !ready {
          self.web.evaluateJavaScript("document.querySelectorAll('canvas').length === 0 && document.querySelector('.ari-art').naturalWidth > 0") { value, error in
            guard error == nil, value as? Bool == true else { self.fail("3D canvas was not disposed") }
            print("PASS: illustration fallback disposes 3D canvas")
            exit(0)
          }
          return
        }
      default: break
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.poll() }
    }
  }
  func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
    guard let event = message.body as? [String: String] else { return }
    if event["type"] == "error" { fail(event["message"] ?? "renderer error") }
    if event["type"] == "ready" {
      if studio {
        web.evaluateJavaScript("document.querySelectorAll('.emotion-grid button').length === 8 && !!document.querySelector('#reference-target') && !!document.querySelector('#model-controls') && document.querySelectorAll('[data-gesture]').length === 10 && !!document.querySelector('#sequence-play') && document.querySelectorAll('#sequence-mode option').length === 6") { value, error in
          guard error == nil, value as? Bool == true else { self.fail("studio controls missing") }
          print("PASS: packaged studio has eight expressions, ten gestures and six rehearsal modes")
          self.poll()
        }
      } else { poll() }
    }
  }
  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { fail(error.localizedDescription) }
  func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { fail(error.localizedDescription) }
}
guard (2...3).contains(CommandLine.arguments.count) || (CommandLine.arguments.count == 4 && CommandLine.arguments[3] == "--portrait-only") else {
  print("Usage: swift scripts/check_renderer.swift <built-renderer/index.html> [snapshot-directory] [--portrait-only]"); exit(2)
}
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let check = RendererCheck()
check.run(CommandLine.arguments[1])
app.run()
