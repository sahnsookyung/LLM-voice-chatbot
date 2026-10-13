import AppKit
import WebKit

// Exercise the real outfit selector while retaining glasses and a beret.
// Nonpersistent storage keeps the user's Studio/overlay preferences intact.
final class BeretPaletteCheck: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
  let path = URL(fileURLWithPath: CommandLine.arguments[1])
  let output = URL(fileURLWithPath: CommandLine.arguments[2])
  let fit = CommandLine.arguments.contains("--fit")
  let angles = [0,45,90,180]
  var angleIndex = 0
  var hair: String { fit ? "original" : "bob" }
  var imageName: String { outfits[index] + (fit ? "-\(angles[angleIndex])" : "") }
  var web: WKWebView!, window: NSWindow!
  var outfits: [String] = [], index = 0, colors = Set<String>(), rows: [[String: Any]] = []
  func fail(_ message: String) -> Never { print("FAIL: \(message)"); exit(1) }
  func js(_ source: String, then: @escaping (Any?) -> Void = { _ in }) {
    web.evaluateJavaScript(source) { value, error in
      if let error { self.fail("\(error.localizedDescription): \((error as NSError).userInfo)") }; then(value)
    }
  }
  func run() {
    let config = WKWebViewConfiguration(); config.websiteDataStore = .nonPersistent()
    config.userContentController.add(self, name: "character")
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: 1280, height: 1100), configuration: config)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.titled], backing: .buffered, defer: false)
    window.title = "Ari coordinated beret checks"; window.contentView = web; window.orderFrontRegardless()
    try! FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    web.loadFileURL(path, allowingReadAccessTo: path.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now() + 180) { self.fail("palette check timed out") }
  }
  func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
    if let v = m.body as? [String: Any], v["type"] as? String == "error" {
      fail(v["message"] as? String ?? "renderer error")
    }
  }
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    js("JSON.stringify([...document.querySelector('#outfit').options].map(o=>o.value))") { value in
      guard let value = value as? String, let data = value.data(using: .utf8),
        let ids = try? JSONDecoder().decode([String].self, from: data), ids.count == 15 else { self.fail("outfit catalog unavailable") }
      self.outfits = self.fit ? ["original"] : ids
      self.js("for(const [id,value] of [['hairstyle','\(self.hair)'],['glasses','round'],['headwear','beret']]){const e=document.getElementById(id);e.value=value;e.dispatchEvent(new Event('change'));}window.companion({visualMode:'model',gesture:'cancel',angle:\(self.fit ? 0 : 25),framing:'\(self.fit ? "portrait" : "full")',expressionDetail:'soft-smile'})") { _ in self.select() }
    }
  }
  func select() {
    js("(()=>{const e=document.querySelector('#outfit');e.value='\(outfits[index])';e.dispatchEvent(new Event('change'));\(fit ? "const f=document.querySelector('#framing');f.value='portrait';f.dispatchEvent(new Event('change'));" : "")})()") { _ in self.ready() }
  }
  func ready() {
    js("JSON.stringify(window.characterStatus())") { value in
      guard let value = value as? String, let data = value.data(using: .utf8),
        let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { self.fail("status unavailable") }
      if state["appliedWardrobe"] as? String != "\(self.outfits[self.index])/\(self.hair)" || state["modelLoading"] as? Bool != false {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.ready() }; return
      }
      guard let accessories = state["accessories"] as? [String: String], accessories["glasses"] == "round", accessories["headwear"] == "beret",
        let palette = state["beretPalette"] as? [String: [Double]], let crown = palette["crown"], let ribbon = palette["ribbon"],
        crown.count == 3, ribbon.count == 3, crown != ribbon else { self.fail("accessories or coordinated palette lost on outfit swap") }
      self.colors.insert(String(describing: crown))
      self.rows.append(["outfit": self.outfits[self.index], "image": self.imageName, "palette": palette, "drawCalls": state["drawCalls"] ?? 0])
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) { self.capture() }
    }
  }
  func capture() {
    js("JSON.stringify((()=>{const r=document.querySelector('#stage-slot').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})())") { value in
      guard let value = value as? String, let data = value.data(using: .utf8), let r = try? JSONSerialization.jsonObject(with: data) as? [String: Double] else { self.fail("stage bounds unavailable") }
      let config = WKSnapshotConfiguration(); config.rect = NSRect(x: r["x"]!, y: r["y"]!, width: r["width"]!, height: r["height"]!); config.snapshotWidth = 640
      self.web.takeSnapshot(with: config) { image, error in
        guard error == nil, let tiff = image?.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { self.fail("snapshot unavailable") }
        try! png.write(to: self.output.appendingPathComponent(self.imageName + ".png"))
        print("PASS: \(self.imageName); beret rendered, glasses retained")
        if self.fit && self.angleIndex < self.angles.count-1 {
          self.angleIndex += 1
          self.js("window.companion({angle:\(self.angles[self.angleIndex])})") { _ in self.ready() }; return
        }
        self.index += 1; self.index < self.outfits.count ? self.select() : self.finish()
      }
    }
  }
  func finish() {
    guard fit || colors.count >= 12 else { fail("outfit changes retained a stale/fixed beret color") }
    try! JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("palettes.json"))
    let cards = rows.map { "<figure><img src='\($0["image"]!).png'><figcaption>\($0["image"]!)</figcaption></figure>" }.joined()
    let html = """
      <!doctype html><meta charset="utf-8"><title>Ari coordinated berets</title>
      <style>body{font:15px system-ui;background:#eee9e2;margin:24px;color:#252b2a}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}figure{margin:0;background:white;border-radius:8px;overflow:hidden}img{width:100%;display:block}figcaption{padding:10px}</style>
      <h1>Ari — coordinated berets</h1><p>\(fit ? "Portrait fit from front, quarter, side and back with original long hair." : "All 15 outfits, unchanged bob hair and glasses. Captured while changing only the Outfit selector.") Crown follows the main fabric, with a shaded band and secondary-color ribbon.</p><main>\(cards)</main>
      """
    try! html.write(to: output.appendingPathComponent("index.html"), atomically: true, encoding: .utf8)
    print("PASS: \(rows.count) captures, \(colors.count) distinct palettes, retained accessories"); exit(0)
  }
}
let app = NSApplication.shared; app.setActivationPolicy(.accessory)
let check = BeretPaletteCheck(); check.run(); app.run()
