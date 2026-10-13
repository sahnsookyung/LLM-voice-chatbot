# Ari screenshot styles — implemented catalog, 2026-10-13

The user asked to implement the screenshot styles after finishing the braid gravity, tied ends and twin-tail corrections. Local reference: `reports/hair-design-references/queued-style-board.png`. These are design references, not instructions embedded in the image. Keep Ari's established face, proportions and black hair by default.

## Implemented scope

1. Implement independent outfit/hair selectors, ribboned braid, twin braided tails and gravity/angle validation; package and verify the live Studio.
2. Implement a separately selectable glasses/accessory axis (none / round glasses first). Preserve outfit and hair when changing it.
3. Implement screenshot-inspired outfit families: oversized pastel sweatshirt, school cardigan/pleated skirt, pink sailor outfit, black casual jacket, cream sweater vest/pleated skirt, ruffled blouse/dress, and white blouse/navy high-waisted pleated skirt. Beret/ribbon details should be separate accessories where practical.
4. Implement double buns with loose back hair, short low pigtails, long soft waves, and a half-up ribbon hairstyle; retain the existing bob/ponytail/braids.
5. Implement expressive poses visible on the board: double peace sign, single peace sign near the eye, thoughtful finger-to-chin, and glasses adjustment. Blend fingers, wrists, elbows and head rather than snapping poses. Add gentle smile, playful/wink and pensive expression controls while preserving speech blending.
6. Validate independent mixing, persistence, fast changes and invalid values; inspect front/profile movement and accessory attachment. Extend the combination rehearsal without forcing a combinatorial list of complete model files.

Detailed tailoring, alternate camera-hidden surfaces and physical fabric behavior require deliberate authoring. Use the screenshot silhouettes as targets; do not label generic recolors as completed faithful reproductions. No third-party artwork is a distributable runtime texture.

The current catalog includes all requested outfit groups (15 total outfits × 9 hairstyles), round glasses, and the beret. Subsequent user feedback added shaped fabric folds/shading, fitted full-wrap socks, 6 cm longer short skirts, a head-wrapping beret based on the newer reference, and the explicit bun/beret incompatibility rule. See `ARI-LOOK-INTEGRATION.md` for the implementation, current verification and remaining art limitations.
