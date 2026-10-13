import AppKit
import WebKit

/// Compile together with DesignAssets.swift. Uses an isolated copy of the art.
@main struct LookLibraryCheck {
  static func main() throws {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let check = try Check(workspace: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    check.run()
    withExtendedLifetime(check) { app.run() }
  }
}

final class Check: NSObject, WKNavigationDelegate {
  let root: URL
  let fixture: URL
  let originalCatalog: Data
  var web: WKWebView!
  var window: NSWindow!
  var phase = 0
  var index = 0
  var since = Date()
  var previousURL = ""
  var savedImage: Data?
  let names = ["tailored", "streetwear", "linen", "bob", "ponytail", "braid"]

  init(workspace: URL) throws {
    root = workspace
    fixture = root.appendingPathComponent(".runtime/look-check-\(UUID().uuidString)")
    originalCatalog = try Data(contentsOf: root.appendingPathComponent("characters/ari/design/looks/catalog.json"))
    super.init()
    try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: root.appendingPathComponent("characters/ari/design/looks"), to: fixture.appendingPathComponent("looks"))
    let handler = DesignAssets(root: fixture)
    for path in ["../character.json", "looks/../../character.json", "looks/a.svg"] {
      do { _ = try handler.assetURL(path); fail("unsafe asset path accepted") } catch { }
    }
    try FileManager.default.createSymbolicLink(at: fixture.appendingPathComponent("looks/escape.png"), withDestinationURL: root.appendingPathComponent("README.md"))
    do { _ = try handler.assetURL("looks/escape.png"); fail("symlink escape accepted") } catch { }
    print("PASS: native artwork handler rejects traversal and symlinks outside the asset root")
    let config = WKWebViewConfiguration()
    config.setURLSchemeHandler(handler, forURLScheme: "ari-design")
    config.userContentController.addUserScript(WKUserScript(source: "window.ariLookBase='ari-design://assets/';try{localStorage.removeItem('ari-reference-look')}catch{}", injectionTime: .atDocumentStart, forMainFrameOnly: true))
    web = WKWebView(frame: NSRect(x: 0, y: 0, width: 1200, height: 1000), configuration: config)
    web.navigationDelegate = self
    window = NSWindow(contentRect: web.frame, styleMask: [.titled], backing: .buffered, defer: false)
    window.title = "Character Studio — reference checks"
    window.contentView = web
  }
  func run() {
    window.orderFrontRegardless()
    let url = root.appendingPathComponent("renderer/dist/studio.html")
    web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline: .now()+75) { self.fail("timed out in phase \(self.phase)") }
  }
  func fail(_ reason: String) -> Never {
    try? FileManager.default.removeItem(at: fixture)
    print("FAIL:",reason); exit(1)
  }
  func command(_ source: String) {
    web.evaluateJavaScript(source) { _, error in if let error { self.fail(error.localizedDescription) } }
  }
  func select(_ name: String) { command("document.querySelector('[data-look=\"\(name)\"]').click()") }
  func capture(_ name: String, completion: @escaping () -> Void = {}) {
    web.takeSnapshot(with: nil) { image, error in
      guard error == nil, let tiff=image?.tiffRepresentation, let png=NSBitmapImageRep(data:tiff)?.representation(using:.png, properties:[:]) else {self.fail("snapshot failed")}
      do {try png.write(to:self.root.appendingPathComponent("reports/ari-looks-\(name).png"))} catch {self.fail(error.localizedDescription)}
      completion()
    }
  }
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { poll() }
  func poll() {
    web.evaluateJavaScript("JSON.stringify({s:window.characterStatus(false),cards:document.querySelectorAll('[data-look]').length,note:document.querySelector('#look-status').textContent,referenceHidden:document.querySelector('#reference-target').hidden,modelHidden:document.querySelector('#model-controls').hidden,art:document.querySelector('.ari-art').src})") { value,error in
      guard error == nil, let string=value as? String, let data=string.data(using:.utf8), let result=try? JSONSerialization.jsonObject(with:data) as? [String:Any], let status=result["s"] as? [String:Any] else {self.fail("studio bridge unavailable: \(error?.localizedDescription ?? "no data")")}
      let image=status["imageURL"] as? String ?? ""
      switch self.phase {
      case 0:
        if result["cards"] as? Int == 7, status["loaded"] as? Bool == true {
          print("PASS: original plus six reference cards load under the packaged file-origin CSP")
          self.phase=1
          self.command("document.querySelector('#reference-look').value='tailored';document.querySelector('#reference-look').dispatchEvent(new Event('change'))")
        }
      case 1:
        if image.contains("/\(self.names[self.index])-v1.png") {
          print("PASS: swapped \(self.names[self.index])")
          self.index += 1
          if self.index < self.names.count {self.select(self.names[self.index])}
          else {
            self.phase=2;self.since=Date()
            self.command("document.querySelector('[data-look=bob]').click();document.querySelector('[data-look=original]').click()")
          }
        }
      case 2:
        if Date().timeIntervalSince(self.since)>0.5 {
          guard image.contains("full-body-neutral-v1.png"), status["referenceLook"] as? String == "original" else {self.fail("a stale in-flight look replaced Original Ari")}
          print("PASS: rapidly selecting then reverting cancels the stale image")
          self.phase=3;self.select("bob")
        }
      case 3:
        if image.contains("bob-v1.png") {
          self.previousURL=image;self.phase=4
          self.command("document.querySelector('#reload-looks').click()")
        }
      case 4:
        if image != self.previousURL, image.contains("bob-v1.png") {
          print("PASS: Reload artwork bypasses the cached image without a page reload")
          self.previousURL=image;self.phase=5
          self.command("window.companion({expression:'curious'})")
          do {try FileManager.default.setAttributes([.modificationDate:Date().addingTimeInterval(2)],ofItemAtPath:self.fixture.appendingPathComponent("looks/bob-v1.png").path)} catch {self.fail(error.localizedDescription)}
        }
      case 5:
        if image != self.previousURL, image.contains("bob-v1.png") {
          guard status["expression"] as? String == "curious" else {self.fail("hot reload reset the expression")}
          print("PASS: automatic workspace revision reload preserves character state")
          self.previousURL=image;self.phase=6;self.since=Date()
          do {try Data("bad JSON".utf8).write(to:self.fixture.appendingPathComponent("looks/catalog.json"),options:.atomic)} catch {self.fail(error.localizedDescription)}
          self.command("document.querySelector('#watch-looks').checked=false;document.querySelector('#reload-looks').click()")
        }
      case 6:
        if Date().timeIntervalSince(self.since)>0.7 {
          guard (result["note"] as? String ?? "").contains("unavailable"), image==self.previousURL else {self.fail("bad catalog did not retain last good artwork")}
          print("PASS: invalid catalog preserves the current library and artwork")
          do {try self.originalCatalog.write(to:self.fixture.appendingPathComponent("looks/catalog.json"),options:.atomic)} catch {self.fail(error.localizedDescription)}
          self.phase=7;self.command("document.querySelector('#reload-looks').click()")
        }
      case 7:
        if image != self.previousURL {
          self.previousURL=image;self.phase=8;self.since=Date()
          do {
            let file=self.fixture.appendingPathComponent("looks/bob-v1.png")
            self.savedImage=try Data(contentsOf:file)
            try Data("incomplete PNG".utf8).write(to:file,options:.atomic)
          } catch {self.fail(error.localizedDescription)}
          self.command("document.querySelector('#reload-looks').click()")
        }
      case 8:
        if Date().timeIntervalSince(self.since)>0.8 {
          guard image==self.previousURL else {self.fail("broken image replaced the previous image")}
          print("PASS: incomplete image save retains the last decoded image")
          do {try self.savedImage!.write(to:self.fixture.appendingPathComponent("looks/bob-v1.png"),options:.atomic)} catch {self.fail(error.localizedDescription)}
          self.phase=9;self.command("document.querySelector('#reload-looks').click()")
        }
      case 9:
        if image != self.previousURL {
          print("PASS: repaired image reloads successfully")
          self.phase=10;self.since=Date()
          self.command("document.querySelector('#framing').value='portrait';document.querySelector('#framing').dispatchEvent(new Event('change'))")
        }
      case 10:
        if Date().timeIntervalSince(self.since)>0.3 {
          guard status["framing"] as? String == "portrait", image.contains("bob-v1.png") else {self.fail("portrait lost selected hairstyle")}
          self.capture("portrait")
          self.phase=11
          self.command("document.querySelector('#mode').value='model';document.querySelector('#mode').dispatchEvent(new Event('change'))")
        }
      case 11:
        if status["modelReady"] as? Bool == true {
          guard result["referenceHidden"] as? Bool == false else {self.fail("VRoid comparison reference missing")}
          print("PASS: selected hairstyle remains alongside the animated VRoid comparison")
          self.phase=12;self.since=Date()
          self.command("document.querySelector('#mode').value='illustration';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#framing').value='full';document.querySelector('#framing').dispatchEvent(new Event('change'));document.querySelector('#watch-looks').checked=true;document.querySelector('#look-library').scrollIntoView({block:'start'})")
        }
      case 12:
        if Date().timeIntervalSince(self.since)>0.5 {
          self.phase=13
          self.capture("library") {
            try? FileManager.default.removeItem(at:self.fixture)
            print("PASS: all reference library integration checks completed")
            exit(0)
          }
        }
      default: return
      }
      DispatchQueue.main.asyncAfter(deadline:.now()+0.15) {self.poll()}
    }
  }
}
