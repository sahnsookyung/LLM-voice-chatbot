# Ari clothing and hairstyle references

Open **Character → Open character studio** and use **Reference look** beside the preview, or choose a thumbnail in **Wardrobe & Hair**. **Original Ari** restores the original outfit and eight expression portraits. New looks retain their neutral expression when an emotion is selected. Portrait framing magnifies the upper body from the same full-body asset.

| Type | Look | Design |
| --- | --- | --- |
| Outfit | City tailoring | Taupe blazer, ivory blouse, charcoal pleated trousers, black loafers |
| Outfit | Utility weekend | Sage bomber, cream tee, charcoal cargo trousers, ivory sneakers |
| Outfit | Woodland linen | Sage midi dress, oatmeal knit cardigan, layered ivory hem, brown boots |
| Hair | Soft tapered bob | Chin-length black hair, swept fringe, one side tucked |
| Hair | High ponytail | Lifted crown, long gathered tail, soft fringe and loose wisps |
| Hair | Loose side braid | Single substantial three-strand braid, swept fringe, delicate wisps |

Outfit studies preserve Ari's original long hair. Hair studies preserve her original cardigan outfit. All six are 1024×1536 PNGs with alpha transparency, stored under `characters/ari/design/looks/`. They were generated with the built-in image_gen tool using the existing Ari portrait and full-body image as identity references. Exact prompts, input paths, creation date and web inspiration are in `looks/provenance.json`; the studio library is `looks/catalog.json`.

## Reloading and adding references

The rebuilt app needs to be reopened once to install the new studio controls. Subsequent artwork changes do not require rebuilding or reopening the studio.

1. Replace a PNG in `characters/ari/design/looks/` while the studio is open. Preserve the filename and transparent background. An atomic file save is preferable.
2. **Auto reload** checks the small catalog and file modification times every three seconds while the studio is visible. Only a changed selection reloads its image. The current expression, framing and 3D animation state survive.
3. Use **Reload artwork** to bypass the selected image's cache immediately. Disable **Auto reload** to pause automatic checks.
4. To add another look, save a PNG named with lowercase letters, numbers and hyphens, then add a unique entry in `catalog.json`. Use category `outfit` or `hair`, an image path such as `looks/new-look-v1.png`, and source IDs present in `sources`. The library supports up to 60 looks. A new valid entry appears on the next refresh.

The desktop studio reads this workspace's design directory via a read-only `ari-design:` handler. Access is limited to look PNGs and the catalog; traversal and symlinks outside the directory are rejected. When the source catalog is absent at app launch, it uses the packaged asset copy. Vite development (`cd renderer && npm exec vite -- --host 127.0.0.1`, then `/?studio`) serves source artwork directly too. A plain static build opened outside the native app uses its bundled copies; source-file watching requires the native app or Vite.

A malformed catalog keeps the previous library. An incomplete or missing selected PNG keeps the last successfully decoded preview. After repairing the file, use Reload artwork or leave Auto reload enabled. Selection is remembered in local browser storage when available; it does not change the floating desktop companion.

## 3D scope

In **VRoid** mode, selecting a look changes the comparison image. The v9 VRM still has its existing cardigan and long-hair meshes, skin weights, materials and spring chains. These PNGs cannot change that geometry, and they are not separate compositing layers for combining arbitrary outfits and hairstyles. Animated variants require authoring/exporting matching garment and hair geometry and verifying skinning, expressions, collisions and resource disposal before they can become 3D presets. No new VRM variants are supplied by this reference library.

## Internet design research

These sources supplied design direction; their images are linked for inspection and are not redistributed in the project.

- [UDN: spring blazer styling](https://woman.udn.com/woman/story/123163/7853493): neutral, relaxed tailoring and varied blazer proportions.
- [Vogue: women's street style at men's fashion week](https://www.vogue.com/article/best-womens-street-style-mens-fashion-week): utility bomber and cargo combinations.
- [Mycloset: layered linen dress](https://item.rakuten.co.jp/mycloset/106009/): natural textures and layered dress silhouettes.
- [NovelAI: character creation hairstyle grid](https://docs.novelai.net/en/image/tutorial-charactercreation/): bob, braid and ponytail variations with a consistent character.
- [Coloso / 2SHAM: anime illustration hair studies](https://coloso.global/zh-TW/products/50chapter-illustrator-2sham-us): coherent black-hair shapes and readable ponytail/braid structure.

## Verification

`npm --prefix renderer test` checks catalog integrity, source references, safe image paths and revision-based URLs, alongside existing motion/model regressions. `npm --prefix renderer run build` typechecks and packages every listed image.

Native integration test:

```sh
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swiftc -module-cache-path .cache/clang scripts/check_look_library.swift native/Sources/Companion/DesignAssets.swift -o .runtime/check-look-library
.runtime/check-look-library
```

The native check uses temporary copies of the assets. It tests six swaps, the sidebar selector, rapid revert cancellation, manual cache bypass, automatic saved-file reload, expression preservation, broken catalog/image recovery, portrait framing and VRoid comparison. It verifies traversal/symlink rejection and writes UI snapshots to `reports/ari-looks-library.png` and `reports/ari-looks-portrait.png`.
