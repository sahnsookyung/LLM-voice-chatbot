import AppKit
import WebKit

// Exercise the native overlay's complete appearance message, including changes
// while hidden. Studio localStorage and its control callbacks are not involved.
final class OverlayCheck: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
  var web: WKWebView!, window: NSWindow!
  var phase = 0, frames = 0, start = Date()
  func fail(_ message: String) -> Never { print("FAIL: \(message)"); exit(1) }
  func command(_ source: String) {
    web.evaluateJavaScript(source) { _, error in if let error { self.fail(error.localizedDescription) } }
  }
  func run() {
    let config = WKWebViewConfiguration()
    config.websiteDataStore = .nonPersistent()
    config.userContentController.add(self, name: "character")
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: 300, height: 540), configuration: config)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.titled], backing: .buffered, defer: false)
    window.title = "Overlay controls test"; window.contentView = web; window.orderFrontRegardless()
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
    web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now() + 60) { self.fail("timed out at phase \(self.phase)") }
  }
  func userContentController(_ c: WKUserContentController, didReceive message: WKScriptMessage) {
    if let value = message.body as? [String: Any], value["type"] as? String == "error" {
      fail(value["message"] as? String ?? "renderer error")
    }
  }
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    command("window.companion({visible:false,visualMode:'model',framing:'full',wardrobe:{outfit:'cloud-sweater',hair:'bob'},accessories:{glasses:'round',headwear:'none'},expression:'happy',level:.25})")
    poll()
  }
  func poll() {
    web.evaluateJavaScript("JSON.stringify({s:window.characterStatus(),canvases:document.querySelectorAll('canvas').length})") { value, error in
      guard error == nil, let string = value as? String, let data = string.data(using: .utf8),
        let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let s = result["s"] as? [String: Any]
      else { self.fail("overlay status unavailable: \(String(describing: error))") }
      let count = s["frames"] as? Int ?? 0, key = s["appliedWardrobe"] as? String ?? ""
      if let message = s["modelError"] as? String, !message.isEmpty { self.fail(message) }
      switch self.phase {
      case 0:
        if key == "cloud-sweater/bob", s["modelLoading"] as? Bool == false {
          self.phase = 1; self.frames = count; self.start = Date()
        }
      case 1:
        if Date().timeIntervalSince(self.start) > 0.6 {
          guard count == self.frames, s["visible"] as? Bool == false else { self.fail("hidden startup drew frames") }
          print("PASS: saved custom overlay loads while hidden without rendering")
          self.phase = 2; self.command("window.companion({visible:true})")
        }
      case 2:
        if count > self.frames + 3 {
          guard result["canvases"] as? Int == 1, key == "cloud-sweater/bob" else { self.fail("show lost custom overlay") }
          print("PASS: show resumes the selected outfit, with one canvas")
          self.phase = 3; self.frames = count; self.start = Date()
          self.command("window.companion({visible:false,wardrobe:{outfit:'navy-blouse',hair:'bob'}})")
        }
      case 3:
        if key == "navy-blouse/bob", s["modelLoading"] as? Bool == false, Date().timeIntervalSince(self.start) > 0.6 {
          guard count == self.frames, s["visible"] as? Bool == false else { self.fail("wardrobe change made hidden overlay render") }
          print("PASS: hidden outfit swap retains the independent hairstyle and stays paused")
          self.phase = 4; self.command("window.companion({visible:true,wardrobe:{outfit:'butter-coat',hair:'original'},accessories:{glasses:'round',headwear:'beret'}})")
        }
      case 4:
        if key == "butter-coat/original", s["modelLoading"] as? Bool == false, count > self.frames + 3 {
          guard let accessories = s["accessories"] as? [String: String], accessories["headwear"] == "beret", accessories["glasses"] == "round",
            let weights = s["expressionWeights"] as? [String: Double], (weights["aa"] ?? 0) > 0,
            (weights["ariHappy"] ?? 0) > 0, (s["footError"] as? Double ?? 1) < 0.001
          else { self.fail("custom overlay lost accessories, expression, speech or foot contact") }
          print("PASS: visible wardrobe change retains accessories, speech, expression and planted feet")
          self.phase = 5; self.command("window.companion({visualMode:'illustration'})")
        }
      case 5:
        if s["mode"] as? String == "illustration" {
          guard result["canvases"] as? Int == 0 else { self.fail("illustration retained a canvas") }
          print("PASS: overlay appearance checks complete"); exit(0)
        }
      default: break
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { self.poll() }
    }
  }
}
let app = NSApplication.shared; app.setActivationPolicy(.accessory)
let check = OverlayCheck(); check.run(); app.run()
