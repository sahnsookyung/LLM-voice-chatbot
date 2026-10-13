import Foundation

struct WardrobeOption: Decodable, Identifiable {
  let id: String
  let label: String
}

struct WardrobeCatalog: Decodable {
  let outfits: [WardrobeOption]
  let hairstyles: [WardrobeOption]

  static let fallback = WardrobeCatalog(
    outfits: [WardrobeOption(id: "original", label: "Original cardigan")],
    hairstyles: [WardrobeOption(id: "original", label: "Long layered hair")])

  static func load(from url: URL?) -> WardrobeCatalog {
    guard let url, let data = try? Data(contentsOf: url),
      let catalog = try? JSONDecoder().decode(WardrobeCatalog.self, from: data),
      catalog.outfits.contains(where: { $0.id == "original" }),
      catalog.hairstyles.contains(where: { $0.id == "original" })
    else { return fallback }
    return catalog
  }
}

struct OverlayLook: Codable, Equatable {
  static let savedLookKey = "ariOverlayLookV1"
  static let randomizeOnStartupKey = "ariRandomizeOutfitOnStartup"
  var outfit = "original"
  var hair = "original"
  var glasses = "none"
  var headwear = "none"

  func normalized(in catalog: WardrobeCatalog) -> OverlayLook {
    var look = self
    if !catalog.outfits.contains(where: { $0.id == look.outfit }) { look.outfit = "original" }
    if !catalog.hairstyles.contains(where: { $0.id == look.hair }) { look.hair = "original" }
    if !["none", "round"].contains(look.glasses) { look.glasses = "none" }
    if !["none", "beret"].contains(look.headwear) || look.hair == "double-buns" { look.headwear = "none" }
    return look
  }

  var bridge: [String: Any] {
    ["wardrobe": ["outfit": outfit, "hair": hair],
     "accessories": ["glasses": glasses, "headwear": headwear]]
  }

  static func restore(from data: Data?, catalog: WardrobeCatalog) -> OverlayLook {
    guard let data, let look = try? JSONDecoder().decode(OverlayLook.self, from: data)
    else { return OverlayLook() }
    return look.normalized(in: catalog)
  }

  static func randomizesOnStartup(in defaults: UserDefaults) -> Bool {
    // bool(forKey:) alone treats a missing preference as false.
    defaults.object(forKey: randomizeOnStartupKey) as? Bool ?? true
  }

  static func loadForStartup<R: RandomNumberGenerator>(
    defaults: UserDefaults, catalog: WardrobeCatalog, using random: inout R
  ) -> OverlayLook {
    var look = restore(from: defaults.data(forKey: savedLookKey), catalog: catalog)
    if randomizesOnStartup(in: defaults) {
      // Avoid repeating the last outfit when the catalog offers alternatives.
      let alternatives = catalog.outfits.filter { $0.id != look.outfit }
      if let outfit = alternatives.randomElement(using: &random) { look.outfit = outfit.id }
      // Disabling randomization later keeps the outfit currently being worn.
      if let data = try? JSONEncoder().encode(look) { defaults.set(data, forKey: savedLookKey) }
    }
    return look
  }
}
