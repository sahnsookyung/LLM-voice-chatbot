import AppKit
import WebKit

// Deterministic visual inspection of the real renderer; no application settings
// are changed. The test controls RAF so each snapshot has an exact pose time.
final class ArmInspection: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
  let path = URL(fileURLWithPath: CommandLine.arguments[1])
  let output = URL(fileURLWithPath: CommandLine.arguments[2])
  let outfits = ["original", "cloud-sweater", "ruffle-dress"]
  let frames = [0, 45, 60, 78, 150, 210]
  var web: WKWebView!, window: NSWindow!
  var look = 0, angle = 0, frameIndex = 0, previousFrame = 0
  var rows: [[String: Any]] = []
  func fail(_ message: String) -> Never { print("FAIL: \(message)"); exit(1) }
  func js(_ source: String, then: @escaping (Any?) -> Void = { _ in }) {
    web.evaluateJavaScript(source) { value, error in
      if let error { self.fail(error.localizedDescription) }; then(value)
    }
  }
  func run() {
    let config = WKWebViewConfiguration(); config.websiteDataStore = .nonPersistent()
    config.userContentController.add(self, name: "character")
    config.userContentController.addUserScript(WKUserScript(source: """
      (()=>{let time=1000,id=0;const frames=new Map();
      window.requestAnimationFrame=f=>{frames.set(++id,f);return id};
      window.cancelAnimationFrame=i=>frames.delete(i);
      window.inspectFrames=count=>{for(let i=0;i<count;i++){
        time+=1000/60;const pending=[...frames.values()];frames.clear();
        for(const f of pending)f(time);
      }};})();
      """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: 600, height: 820), configuration: config)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.titled], backing: .buffered, defer: false)
    window.title = "Ari arm-raising assessment"; window.contentView = web; window.orderFrontRegardless()
    try! FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    web.loadFileURL(path, allowingReadAccessTo: path.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now() + 120) { self.fail("inspection timed out") }
  }
  func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
    if let v = m.body as? [String: Any], v["type"] as? String == "error" {
      fail(v["message"] as? String ?? "renderer error")
    }
  }
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { select() }
  func select() {
    frameIndex = 0; previousFrame = 0
    js("window.companion({visualMode:'model',visible:true,framing:'full',wardrobe:{outfit:'\(outfits[look])',hair:'bob'},accessories:{glasses:'none',headwear:'none'},gesture:'cancel',angle:\(angle),expression:'neutral',gestureSpeed:1});") { _ in self.ready() }
  }
  func ready() {
    js("JSON.stringify(window.characterStatus())") { value in
      let state = self.decode(value)
      if state["appliedWardrobe"] as? String == "\(self.outfits[self.look])/bob", state["modelLoading"] as? Bool == false {
        self.js("window.inspectFrames(240);window.companion({gesture:'raise-arms'});") { _ in self.capture() }
      } else { DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.ready() } }
    }
  }
  func decode(_ value: Any?) -> [String: Any] {
    guard let string = value as? String, let data = string.data(using: .utf8),
      let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { fail("renderer status unavailable") }; return state
  }
  func capture() {
    let frame = frames[frameIndex], step = frame - previousFrame
    let name = "\(outfits[look])-\(angle)-\(frame)"
    js("window.inspectFrames(\(step));JSON.stringify(window.characterStatus())") { value in
      let state = self.decode(value)
      guard (state["footError"] as? Double ?? 1) < 0.001 else { self.fail("foot contact lost") }
      if frame == 78 || frame == 150 {
        guard (state["rightHandHeight"] as? Double ?? 0) > 1.5 else { self.fail("raise did not advance; another window may have hidden the renderer") }
      }
      var row: [String: Any] = ["outfit": self.outfits[self.look], "angle": self.angle, "seconds": Double(frame)/60, "image": name + ".png"]
      for key in ["rightHandHeight", "shoulderAngle", "footError", "modelError"] { row[key] = state[key] }
      self.rows.append(row)
      let config = WKSnapshotConfiguration(); config.snapshotWidth = 600
      self.web.takeSnapshot(with: config) { image, error in
        guard error == nil, let tiff = image?.tiffRepresentation,
          let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
        else { self.fail("snapshot unavailable") }
        try! png.write(to: self.output.appendingPathComponent(name + ".png"))
        self.previousFrame = frame; self.frameIndex += 1
        if self.frameIndex < self.frames.count { self.capture() }
        else {
          print("Captured \(self.outfits[self.look]) at \(self.angle) degrees")
          if self.angle == 0 { self.angle = 45; self.select() }
          else { self.angle = 0; self.look += 1; self.look < self.outfits.count ? self.select() : self.finish() }
        }
      }
    }
  }
  func finish() {
    try! JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("poses.json"))
    let cards = rows.map { row in
      "<figure><img src='\(row["image"]!)'><figcaption>\(row["outfit"]!) · \(row["angle"]!)° · \(row["seconds"]!) s</figcaption></figure>"
    }.joined()
    let html = """
      <!doctype html><meta charset="utf-8"><title>Ari arm-raising assessment</title>
      <style>body{font:15px system-ui;background:#eee9e2;margin:24px;color:#252b2a}main{display:grid;grid-template-columns:repeat(6,minmax(180px,1fr));gap:8px}figure{margin:0;background:white;border-radius:8px;overflow:hidden}img{width:100%;display:block}figcaption{padding:8px}h1{font-size:24px}</style>
      <h1>Ari — arm-raising inspection</h1><p>Six fixed times per view. Bob hair exposes the shoulder. Each outfit has front and 45° rows. Captured from the supplied renderer without changing saved app settings.</p><main>\(cards)</main>
      """
    try! html.write(to: output.appendingPathComponent("index.html"), atomically: true, encoding: .utf8)
    print("PASS: 36 deterministic pose captures, raised hands and planted feet verified"); exit(0)
  }
}
let app = NSApplication.shared; app.setActivationPolicy(.accessory)
let inspection = ArmInspection(); inspection.run(); app.run()
