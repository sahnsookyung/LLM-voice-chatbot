import AppKit
import WebKit

// Reproducible CPU-side profile at the actual 300 x 540 overlay size.
// Run against the same built renderer before/after an optimization. No inference
// workload is generated, and draw submission time does not measure GPU execution.
final class RendererBenchmark: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
  var web: WKWebView!
  var window: NSWindow!
  var started = Date()
  var baseline: [String: Double] = [:]
  var phase = 0
  func run(_ path: String) {
    let controller = WKUserContentController()
    controller.add(self, name: "character")
    controller.addUserScript(WKUserScript(source: "let seed=12345;Math.random=()=>((seed=Math.imul(1664525,seed)+1013904223>>>0)/4294967296);", injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let config = WKWebViewConfiguration(); config.userContentController = controller
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: 300, height: 540), configuration: config)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = web; window.orderFrontRegardless()
    let file = URL(fileURLWithPath: path)
    var url = URLComponents(url: file, resolvingAgainstBaseURL: false)!
    url.query = "profile=1"
    web.loadFileURL(url.url!, allowingReadAccessTo: file.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now() + 60) { self.fail("timeout") }
  }
  func fail(_ message: String) -> Never { print("FAIL: \(message)"); exit(1) }
  func command(_ script: String) {
    web.evaluateJavaScript(script) { _, error in if let error { self.fail(error.localizedDescription) } }
  }
  func poll() {
    web.evaluateJavaScript("window.characterStatus()") { value, error in
      guard error == nil, let status = value as? [String: Any] else { self.fail("status missing") }
      if status["modelReady"] as? Bool == true, let totals = status["cpuTotals"] as? [String: Double] {
        let elapsed = Date().timeIntervalSince(self.started)
        if self.phase == 0 {
          if elapsed > 3 { self.baseline = totals; self.started = Date(); self.phase = 1 }
        } else if elapsed > 10 {
          let delta = totals.merging(self.baseline) { current, old in current - old }
          let steps = max(1, delta["steps"] ?? 0), draws = max(1, delta["draws"] ?? 0)
          let result: [String: Any] = [
            "scenario": self.phase == 1 ? "idle" : "twirl", "seconds": elapsed,
            "simulationHz": steps / elapsed, "drawFPS": draws / elapsed,
            "poseMsMean": (delta["pose"] ?? 0) / steps,
            "vrmMsMean": (delta["vrm"] ?? 0) / steps,
            "simulationMsMean": (delta["simulation"] ?? 0) / steps,
            "contactMsMean": (delta["contact"] ?? 0) / draws,
            "drawSubmissionMsMean": (delta["draw"] ?? 0) / draws,
            "simulationMsP95": status["simulationMsP95"] ?? 0,
            "drawSubmissionMsP95": status["drawSubmissionMsP95"] ?? 0,
            "triangles": status["triangles"] ?? 0, "drawCalls": status["drawCalls"] ?? 0,
            "footError": status["footError"] ?? 1,
          ]
          let json = try! JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
          print(String(data: json, encoding: .utf8)!)
          if self.phase == 2 {
            self.web.evaluateJavaScript("(()=>{const t=performance.now();for(let i=0;i<500;i++)window.characterStatus(false);return (performance.now()-t)/500})()") { value, error in
              guard error == nil else { self.fail("status benchmark failed") }
              print("Lightweight status mean ms: \(value ?? "?")"); exit(0)
            }
            return
          }
          self.phase = 2; self.started = Date(); self.baseline = totals
          self.command("window.companion({gesture:'twirl'});setTimeout(()=>window.companion({gesture:'twirl'}),5000)")
        }
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.poll() }
    }
  }
  func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
    guard let event = message.body as? [String: String] else { return }
    if event["type"] == "error" { fail(event["message"] ?? "renderer error") }
    if event["type"] == "ready" {
      command("window.companion({visualMode:'model',framing:'full',visible:true,expression:'neutral'})")
      started = Date(); poll()
    }
  }
  func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { fail(error.localizedDescription) }
}
guard CommandLine.arguments.count == 2 else { print("Usage: swift scripts/benchmark_renderer.swift <built-renderer/index.html>"); exit(2) }
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let benchmark = RendererBenchmark()
benchmark.run(CommandLine.arguments[1])
app.run()
