import Foundation
import XCTest
@testable import Companion

final class OverlayAppearanceTests: XCTestCase {
  private struct CountingRandom: RandomNumberGenerator {
    var draws = 0
    mutating func next() -> UInt64 { draws += 1; return UInt64.max / 2 + UInt64(draws) }
  }
  private func withPreferences(_ test: (UserDefaults, String) throws -> Void) rethrows {
    let suite = "CompanionTests.Overlay.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    try test(defaults, suite)
  }
  private var catalog: WardrobeCatalog {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    return WardrobeCatalog.load(from: root.appendingPathComponent("renderer/dist/wardrobe-catalog.json"))
  }
  func testBundledCatalogIncludesEveryRendererChoice() {
    XCTAssertEqual(catalog.outfits.count, 15)
    XCTAssertEqual(catalog.hairstyles.count, 9)
    XCTAssertEqual(Set(catalog.outfits.map(\.id)).count, catalog.outfits.count)
    XCTAssertEqual(Set(catalog.hairstyles.map(\.id)).count, catalog.hairstyles.count)
  }
  func testEveryLookSurvivesPersistenceAndBridgeSerialization() throws {
    for outfit in catalog.outfits {
      for hair in catalog.hairstyles {
        let look = OverlayLook(outfit: outfit.id, hair: hair.id, glasses: "round", headwear: "beret").normalized(in: catalog)
        XCTAssertEqual(OverlayLook.restore(from: try JSONEncoder().encode(look), catalog: catalog), look)
        let decoded = try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: look.bridge)) as! [String: [String: String]]
        XCTAssertEqual(decoded["wardrobe"], ["outfit": outfit.id, "hair": hair.id])
        XCTAssertEqual(decoded["accessories"], ["glasses": "round", "headwear": hair.id == "double-buns" ? "none" : "beret"])
      }
    }
  }
  func testUnavailableSavedPartsRecoverIndependently() throws {
    let stale = OverlayLook(outfit: "removed", hair: "braid", glasses: "round", headwear: "removed")
    let recovered = OverlayLook.restore(from: try JSONEncoder().encode(stale), catalog: catalog)
    XCTAssertEqual(recovered, OverlayLook(outfit: "original", hair: "braid", glasses: "round", headwear: "none"))
    XCTAssertEqual(OverlayLook.restore(from: Data("broken".utf8), catalog: catalog), OverlayLook())
    XCTAssertEqual(WardrobeCatalog.load(from: nil).outfits.first?.id, "original")
  }
  func testBunsRemoveOnlyHatAndOutfitRetainsIndependentHair() {
    var look = OverlayLook(outfit: "butter-coat", hair: "original", glasses: "round", headwear: "beret")
    look.hair = "double-buns"
    look = look.normalized(in: catalog)
    XCTAssertEqual(look, OverlayLook(outfit: "butter-coat", hair: "double-buns", glasses: "round", headwear: "none"))
    look.outfit = "navy-blouse"
    XCTAssertEqual(look.normalized(in: catalog).hair, "double-buns")
  }
  func testStartupRandomizationDefaultsOnAndChangesOnlyOutfit() throws {
    try withPreferences { defaults, _ in
      let saved = OverlayLook(outfit: "butter-coat", hair: "braid", glasses: "round", headwear: "beret")
      defaults.set(try JSONEncoder().encode(saved), forKey: OverlayLook.savedLookKey)
      var random = CountingRandom()
      XCTAssertTrue(OverlayLook.randomizesOnStartup(in: defaults))
      let next = OverlayLook.loadForStartup(defaults: defaults, catalog: catalog, using: &random)
      XCTAssertNotEqual(next.outfit, saved.outfit)
      XCTAssertTrue(catalog.outfits.contains { $0.id == next.outfit })
      XCTAssertEqual(next.hair, saved.hair)
      XCTAssertEqual(next.glasses, saved.glasses)
      XCTAssertEqual(next.headwear, saved.headwear)
      XCTAssertGreaterThan(random.draws, 0)
      XCTAssertEqual(OverlayLook.restore(from: defaults.data(forKey: OverlayLook.savedLookKey), catalog: catalog), next)
    }
  }
  func testDisabledPreferenceSurvivesReloadAndKeepsCurrentLook() throws {
    try withPreferences { defaults, suite in
      let saved = OverlayLook(outfit: "navy-blouse", hair: "soft-waves", glasses: "round", headwear: "beret")
      defaults.set(try JSONEncoder().encode(saved), forKey: OverlayLook.savedLookKey)
      defaults.set(false, forKey: OverlayLook.randomizeOnStartupKey)
      let reloaded = UserDefaults(suiteName: suite)!
      var random = CountingRandom()
      XCTAssertFalse(OverlayLook.randomizesOnStartup(in: reloaded))
      XCTAssertEqual(OverlayLook.loadForStartup(defaults: reloaded, catalog: catalog, using: &random), saved)
      XCTAssertEqual(random.draws, 0)
      reloaded.set(true, forKey: OverlayLook.randomizeOnStartupKey)
      XCTAssertTrue(OverlayLook.randomizesOnStartup(in: defaults))
      XCTAssertNotEqual(OverlayLook.loadForStartup(defaults: reloaded, catalog: catalog, using: &random).outfit, saved.outfit)
    }
  }
  func testTurningOffAfterRandomStartupKeepsTheRandomizedOutfit() {
    withPreferences { defaults, _ in
      var random = CountingRandom()
      let current = OverlayLook.loadForStartup(defaults: defaults, catalog: catalog, using: &random)
      defaults.set(false, forKey: OverlayLook.randomizeOnStartupKey)
      XCTAssertEqual(OverlayLook.loadForStartup(defaults: defaults, catalog: catalog, using: &random), current)
    }
  }
  func testSingleOutfitCatalogKeepsValidDefaultWithoutDrawingRandomness() {
    withPreferences { defaults, _ in
      var random = CountingRandom()
      XCTAssertEqual(OverlayLook.loadForStartup(defaults: defaults, catalog: .fallback, using: &random), OverlayLook())
      XCTAssertEqual(random.draws, 0)
    }
  }
}
