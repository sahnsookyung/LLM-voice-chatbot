import Foundation
import WebKit

/// Read-only access to the reference library, never arbitrary workspace files.
/// A custom scheme keeps WKWebView's file-origin sandbox and CSP intact.
final class DesignAssets: NSObject, WKURLSchemeHandler {
  let root: URL

  init(root: URL) { self.root = root.resolvingSymlinksInPath(); super.init() }

  static func studioConfiguration() -> WKWebViewConfiguration {
    let resources = Bundle.main.resourceURL!
    let workspace = try? String(contentsOf: resources.appendingPathComponent("workspace.txt"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    let source = workspace.map { URL(fileURLWithPath: $0).appendingPathComponent("characters/ari/design") }
    let root = source.flatMap { FileManager.default.fileExists(atPath: $0.appendingPathComponent("looks/catalog.json").path) ? $0 : nil }
      ?? resources.appendingPathComponent("renderer/ari")
    let config = WKWebViewConfiguration()
    config.setURLSchemeHandler(DesignAssets(root: root), forURLScheme: "ari-design")
    config.userContentController.addUserScript(WKUserScript(source: "window.ariLookBase='ari-design://assets/';", injectionTime: .atDocumentStart, forMainFrameOnly: true))
    return config
  }

  func assetURL(_ path: String) throws -> URL {
    guard path.range(of: "^looks/[a-z0-9-]+\\.png$", options: .regularExpression) != nil || path == "looks/catalog.json" else { throw URLError(.noPermissionsToReadFile) }
    let file = root.appendingPathComponent(path).resolvingSymlinksInPath()
    guard file.path.hasPrefix(root.path + "/") else { throw URLError(.noPermissionsToReadFile) }
    return file
  }

  func catalogScript() throws -> Data {
    let data = try Data(contentsOf: assetURL("looks/catalog.json"))
    guard var catalog = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      var looks = catalog["looks"] as? [[String: Any]], looks.count <= 60
    else { throw URLError(.cannotParseResponse) }
    for index in looks.indices {
      guard let path = looks[index]["image"] as? String else { throw URLError(.cannotParseResponse) }
      let file = try assetURL(path)
      // Missing or partially saved art must not remove the rest of the library.
      let values = try? file.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
      looks[index]["revision"] = "\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)-\(values?.fileSize ?? 0)"
    }
    catalog["looks"] = looks
    let json = try JSONSerialization.data(withJSONObject: catalog, options: [.sortedKeys])
    var script = Data("window.dispatchEvent(new CustomEvent('ari:catalog',{detail:".utf8)
    script.append(json); script.append(Data("}));".utf8))
    return script
  }

  func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
    do {
      guard let url = urlSchemeTask.request.url, url.host == "assets" else { throw URLError(.badURL) }
      let path = String(url.path.dropFirst())
      let script = path == "looks/catalog.js"
      let data = try script ? catalogScript() : Data(contentsOf: assetURL(path))
      let response = URLResponse(url: url, mimeType: script ? "text/javascript" : "image/png", expectedContentLength: data.count, textEncodingName: script ? "utf-8" : nil)
      urlSchemeTask.didReceive(response); urlSchemeTask.didReceive(data); urlSchemeTask.didFinish()
    } catch { urlSchemeTask.didFailWithError(error) }
  }

  func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
    // Requests finish synchronously and retain no task or file handles.
  }
}
