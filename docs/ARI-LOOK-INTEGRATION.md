# Ari wardrobe and hairstyle integration

Implemented in Character Studio, 2026-10-13. Outfit and Hairstyle are independent controls for **animated 3D Ari**. Fifteen outfits and nine hairstyles provide 135 combinations, with independent round glasses and a fitted beret. The screenshot board has been adapted to Ari’s existing adult body and face. These remain stylized authored approximations, not exact reproductions of the illustrations.

## Using the Studio

Open Character → Open character studio. Choose Outfit and Hairstyle under **3D wardrobe**. Changing either automatically opens the 3D view and retains the other selection. The pair is saved locally for the next Studio load. No application restart is needed between selections. **Cycle all 135 looks** waits for each model to load before displaying it for 2.5 seconds; Stop or hiding the character stops the cycle.

Glasses and Headwear are separate saved selections and update without rebuilding the model. Double buns disable the beret; choosing buns removes an existing beret and explains the fit restriction. Portrait framing includes the hat.

The Comparison reference selector and reference cards choose a complete illustration to compare against the model. They do not change either 3D selection. The six PNGs cannot be combined into arbitrary outfit/hair pairs by layering them.

The native **Character → Desktop wardrobe** controls apply the same catalog to the desktop overlay. Outfit, hairstyle, glasses and headwear persist independently of Studio's preview choices. Selecting a part switches the overlay to 3D without changing its show/hide state. The native catalog is generated from the renderer's IDs and labels during the Vite build; it is not maintained as a second hardcoded list. Unknown saved parts fall back independently to valid defaults.

The app header, Character tab, menu-bar sparkle and Command–Shift–H toggle the overlay. Visibility persists across restart, and hidden rendering stays paused through wardrobe changes. Showing it retains its current window position and selected look.

These changes do not create editable `.vroid` projects: the latest editable VRoid source remains v6, with v7–v9 finishing and the new wardrobe applied after export.

## Supported parts

| Part | Current implementation | Remaining art or motion work |
| --- | --- | --- |
| Original cardigan / long hair | Unchanged v9 geometry, textures and rig | Existing likeness limitations remain. |
| City tailoring | Taupe shortened outer layer, fitted lapels/welt details, wider charcoal trousers and closed shoes | Jacket tailoring, closures, shoe detail and cloth folds need custom authoring. It retains some cardigan construction. |
| Utility weekend | Shorter, fuller sage outer layer, cream inner layer, cargo pockets and light shoes with laces | Jacket seams, cuff shape and cargo volume are simplified. |
| Woodland linen | Sage continuous skirt, oatmeal under-hem, cardigan and brown boots; 12 skirt guide chains | Drape and garment layering are approximate. No general cloth or self-collision solver. |
| Soft tapered bob | Preserved fringe, 100 fine short sections and ten short guide chains | Asymmetry, tucked ear detail and strand grouping can be refined. |
| High ponytail | Swept crown, fixed tie and one eight-node guide chain with finer tail strands | Crown silhouette and tie detail remain simpler than the drawing. |
| Loose side braid | Three interwoven bundles gathered into a ribbon band and bow, with a separate tapered loose tuft; nine-node guide chain | Plait shape and large-motion deformation still need refinement. |
| Twin braided tails | Two independent nine-node guides, swept crown sections, two ribbon bows and loose ends; based on the user’s attached hairstyle references | Crown flow, shoulder clearance and ribbon folds remain stylized. |

## Screenshot wardrobe and fabric pass

Additional outfits: cloud blue and butter yellow sweatshirts, gray and cocoa school cardigans with pleats, cream school vest, white ribbon blouse with navy skirt, pink sailor dress, ruffled ribbon dress, black jacket, butter coat, and forest pinafore. Additional hair: double buns with loose lengths, short low pigtails, long soft waves, and half-up ribbon waves.

The generated garments now use sampled curved surfaces with waist gathering and tension folds, rolled hems, separate waistbands, curved collars, folded ribbon loops, plackets, domed buttons and pocket flaps. Pleats use 192 circumferential samples to resolve their shape. New cloth uses rough physically based materials with distance-filtered weave detail. Original cardigan texture luminance is retained during recoloring instead of clipping the bright knit samples. The original/original v9 appearance is unchanged.

Knee socks copy and offset Ari’s actual calf surface, clip cleanly at the upper/lower boundaries, and retain the existing skin weights. They cover the front, sides and back. The short skirts were lengthened by 6 cm; the long linen skirt is unchanged. Low-waist skirt roots fit the hips beneath longer tops rather than intersecting them.

The beret is a shaped crown with a turned-under edge, a tilted elliptical opening that arches across the hairline, a narrow band and a side ribbon. Its form follows the user’s newer beret reference, saved only as a design reference in `reports/hair-design-references/beret-shape.png`. No pixels from that reference are used as runtime textures.

Screenshot poses include double peace, peace near the eye, thoughtful finger-to-chin and glasses adjustment. The arm/wrist targets were fitted against Ari’s normalized bones and fingertip locations. Gentle smile, playful wink and pensive detail controls blend with speech and the existing eight emotions. The expression-pair rehearsal still covers 56 directed transitions.

Fabric folds are modeled and skinned, not a per-vertex cloth simulation. Sparse skirt guides supply secondary motion. Dense weave detail fades with distance to avoid shimmer. Precise tailoring, shoe shaping, contact during extreme gestures and exact illustration matching remain art-refinement work.

## Identity and asset contract

`renderer/src/wardrobe-model.ts` adapts only the known finished v9 GLB layout. It builds the selected combination from cached base bytes at load time. It does not embed or keep 135 complete models. The original/original selection returns the original buffer directly.

The adapter copies the face mesh, all facial morphs, face/eye/skin materials, actual body skin vertices, humanoid rest transforms and expression definitions unchanged. Existing garment vertices retain their original skin influences. New panels inherit influences from the original garment; footwear follows foot/lower-leg bones. New hair and skirt joints get their own parent-local transforms and inverse bind matrices. Unit tests compare existing bind transforms with the authored baseline and require identity bind transforms for new joints.

This is a version-specific adapter, not an arbitrary VRM part importer. Layout guards and asset-contract tests must be updated deliberately for a new base export. Do not run earlier `finish_ari_v6.mjs`, `detail_ari_v7.mjs`, `refine_ari_v8.mjs` or `refine_ari_v9.mjs` against reordered VRoid exports by bypassing their hash guards.

The original file is unchanged: `characters/ari/model/ari-vroid-v9.vrm`, SHA-256 `fbae3f59fa8faa872167954656c05b8d843fe6f1fda8c8996567b1c12d462c8b`.

## Loading and resource ownership

Only the active model and one temporary candidate can be resident. While a candidate loads, the old character remains visible. Rapid changes replace the requested selection; obsolete candidates are disposed before the latest selection loads. Compatible expression and pose state is applied to the candidate, then its spring history is reset before it replaces the old model. An invalid request or failed current load retains the previous preview. A failed obsolete request does not prevent a newer choice from loading.

Retired models release their geometry, materials, textures, skeleton buffers, canvas and WebGL context. Returning to Illustration disposes the active 3D renderer. CPU mesh construction is synchronous; decoded textures still need loading for each candidate. Moving generation into a worker or caching selected assembled buffers is a future optimization only if measured swap latency warrants the extra memory and complexity.

## Hair and clothing motion

Main garment deformation uses skin weights. Secondary hair and skirt motion uses VRM spring bones with a fixed 60 Hz simulation. The bob, ponytail, braid and twin tails have separate guide layouts and response settings. Braids start below the ears, follow the collar/shoulder contour, and hang vertically over the upper torso. Only their roots are firm; lower joints use stronger downward gravity than restoring stiffness. A closed core fills the interwoven bundles, and the ribbon/band/loose tuft share the same tail rig. Outfit selection expands the nine existing hair contact ellipsoids slightly for the fuller tailoring and utility silhouettes; it does not grow an unbounded collider list. The skirt uses a continuous mesh blended between neighboring guides, with thigh capsule contacts.

These are conservative contact approximations. They do not guarantee collision-free arbitrary poses, hair-to-hair contact or an interacting cloth surface. The new skinning and guide rigs run locally; Blender and VRoid are not required during playback. See the [VRM spring-bone specification](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone-1.0/README.md) and [VRoid hair-bounce controls](https://vroid.pixiv.help/hc/en-us/articles/900006910023-I-want-to-edit-hair-bounce).

Future artist-authored garments should include explicit roles, skeleton/bind-space contracts, outfit-specific body visibility masks, contact envelopes and pose correctives. A new mask must fit the actual neckline, sleeve and hem coverage; it cannot simply inherit the mask of a different garment. See [VRoid skin masks](https://vroid.pixiv.help/hc/en-us/articles/4405516267033-How-to-use-Skin-Masks).

## Verification and performance limits

```sh
npm --prefix renderer test
npm --prefix renderer run build
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swift scripts/check_wardrobe.swift renderer/dist/studio.html reports/wardrobe
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swift scripts/check_style_board.swift renderer/dist/studio.html reports/style-final
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swift scripts/check_renderer.swift renderer/dist/index.html reports/wardrobe-baseline
```

Unit checks cover all 135 combinations, finite geometry, indices, normalized weights, valid spring references, new bind matrices, unchanged identity data, independent selection validation and saved state. Native checks exercise real selectors, 80 rapid changes, document-reload persistence, ribbon/tuft binding, expressions during swapping, planted feet, ponytail motion/settling, linen and twin-tail twirl/settling, invalid input retention, canvas disposal and the cycle/stop/hidden controls. The expanded native check cycles the whole current catalog. The style-board check captures front, three-quarter and back views, all four new poses, accessory persistence, bun/beret incompatibility, portrait framing and cleanup. Latest screenshots are retained in `reports/style-final/`; historical first-batch screenshots remain in `reports/wardrobe/`. The shared motion tests cover all 56 expression transitions; the full native expression rehearsal has not been repeated on every wardrobe pair.

Native rendered counts are recorded per combination in `reports/wardrobe-native.txt`. Original/original remains 25 draws and 163,466 rendered triangles. Fewer triangles alone do not establish a performance improvement. Existing 30 FPS idle / up-to-60 FPS active drawing, 1.5 DPR cap and hidden pause remain. Sustained memory, GPU/energy cost, swap latency and concurrent voice-inference responsiveness have not been benchmarked for the new variants.

Passing structural and motion checks is separate from art acceptance. Review neutral and moving poses against the references at front, profile, back and three-quarter angles before treating a look as finished.
