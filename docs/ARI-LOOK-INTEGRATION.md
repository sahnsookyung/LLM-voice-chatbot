# Ari wardrobe and hairstyle integration

Engineering assessment, 2026-10-13. The six new reference illustrations are in `characters/ari/design/looks/`. This document describes the next 3D authoring work; it does not imply that new 3D garments, hairstyles or cloth simulation already exist.

## What the references provide

The catalog contains City tailoring, Utility weekend and Woodland linen outfits, plus Soft tapered bob, High ponytail and Loose side braid hairstyles. Each PNG is a complete illustration. Clothing references retain the long hairstyle; hairstyle references retain the original cardigan outfit. They cannot be combined into arbitrary outfit/hair pairs by layering these PNGs.

The Studio library selects an illustration or uses it as the comparison target beside the existing v9 VRM. Reference hot reload and 3D model replacement are separate operations. An illustration has no skin weights, hidden surfaces, facial morphs or physics rig. Each new 3D look first needs consistent front, side and back design decisions, including closures, hair attachment, garment thickness and areas hidden by the arms.

## Preserve Ari's identity before changing parts

Keep the accepted face geometry, all facial morphs, eye and skin textures, body proportions, humanoid mappings and expression behavior as the common baseline. A wardrobe export from the editable v6 `.vroid` would not contain the post-export v7–v9 likeness improvements automatically.

The current finishing scripts are deliberately specific to their input files. `finish_ari_v6.mjs`, `detail_ari_v7.mjs`, `refine_ari_v8.mjs` and `refine_ari_v9.mjs` validate exact hashes and refer to particular mesh, material, texture or vertex layouts. A new VRoid export can reorder these or change topology. Do not bypass the guards and apply the old edits to a different export.

For the first variant, author and fit its parts against the existing finished body, then produce a separately validated complete VRM. This keeps the first runtime replacement small. Build a reusable part-export process once a second look proves which structure is actually shared. Independent outfit/hair selectors should become available only after those combinations have been built and validated.

If parts are assembled programmatically, identify them by explicit exported roles and validate their skeleton binding, bind matrices, units, coordinate system, material features and texture layout. Transfer new clothes onto the common skeleton; do not replace Ari's face with a fresh default export. Fixed raw array indices are not a durable part interface.

## Movement requirements by look

These are proposed authoring choices, not measurements of assets that have already been built.

| Reference | First implementation | Main difficulty and acceptance check |
| --- | --- | --- |
| City tailoring | Skin trousers and blazer to the body; keep shoulders structured; add restrained hem/lapel motion only where visible. | Raised arms expose shoulder and underarm weighting; lapels can intersect the blouse; wide trouser legs can meet during steps. Check shoe height against foot contact. |
| Utility weekend | Preserve jacket volume with skin weights and pose correctives; keep waistband and cuffs anchored. | Puffy sleeves need more clearance around torso, wrists and hair than the current fitted cardigan. Check shrugging, pointing and arms raised together. |
| Woodland linen | Anchor the dress at the waist; use a small skirt rig with leg clearance and authored turn/bend correctives. Treat the cardigan as another fitted layer. | Highest garment risk: legs crossing the hem, layer intersections, unnatural skirt collapse and excessive twirl flare. A few independent spring chains alone do not enforce cloth surface continuity. |
| Soft tapered bob | Short head-bound guide chains, firmer roots, restrained tip motion. | Cheek, ear and neck contact; preserve the tucked side and silhouette during head turns. It may need fewer contact regions than long hair. |
| High ponytail | A fixed tie/root and a continuous tail guide rig; separate, quieter fringe controls. | Tail must clear the skull, neck, shoulders and back at fast turns and interruptions. Retune inertia and settling for the new length and attachment; do not copy the loose-hair settings wholesale. |
| Loose side braid | Deform the interwoven shape as a cohesive bundle with controlled bending and twist; animate loose wisps separately. | Keep it braided while bending. Chest, shoulder, jacket and raised-arm contact matter. Define which side it rests on and how it returns after a turn. |

The existing gestures can share their humanoid motion, but a garment may need a corrective pose or a smaller range of motion. For example, a narrow skirt cannot support the same unrestricted step as trousers just by increasing collision strength. Shoes with different sole or heel heights also need updated contact offsets.

## Hair and clothing physics

Use a hybrid approach: skin weights for the main shape, pose correctives for predictable deformation, and a small number of simulated guide bones for secondary motion. Increase complexity only when a specific visual failure remains. Blender is useful for custom silhouettes, weights and corrective shapes; VRoid remains useful for hair authoring and supported garment bases. Neither needs to run during playback.

VRM spring bones provide inertia, stiffness, drag, gravity and sphere/capsule contacts for chains. They do not provide a full interacting cloth surface. [VRM spring-bone specification](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone-1.0/README.md)

VRoid exposes hair bone groups and bounce settings, so each hairstyle can have its own guide layout and response. More bones are not automatically a better likeness or better motion. [VRoid hair-bounce controls](https://vroid.pixiv.help/hc/en-us/articles/900006910023-I-want-to-edit-hair-bounce)

Current v9 has ten hair guide chains with 40 simulated joints. The app also pushes rendered hair away from nine bone-following ellipsoids in `renderer/src/hair-contact.ts`. Those ellipsoids fit the current silhouette; they are not automatic collisions against arbitrary new clothing. A bomber needs a different clearance envelope from a blouse. Skirts and loose hems would need their own motion and contact design. There is currently no hair self-collision or general cloth solver.

Store a collision envelope with each outfit and spring settings with each hairstyle. Choose relevant collider groups for each combination, then visually test it. Avoid solving every hair vertex against every garment triangle. If a moving hem affects hair, use a conservative envelope or a small number of bone-following proxies first; document the approximation instead of presenting it as physically exact.

Use body visibility masks per outfit to prevent concealed skin from poking through. They must follow the actual neckline, sleeve and hem coverage; a mask for one garment can leave visible holes under another. These masks do not solve hair contact or garment-to-garment intersections. [VRoid skin-mask documentation](https://vroid.pixiv.help/hc/en-us/articles/4405516267033-How-to-use-Skin-Masks)

## Proposed 3D look package

Start with one complete, validated VRM per supported combination and a small manifest containing:

- Stable look ID, display name, preview image, revision/hash and asset path.
- Common Ari identity revision and humanoid rig contract.
- Outfit and hairstyle IDs, declared compatibility, required runtime material features.
- Hair spring configuration, outfit collision profile, body visibility mask and foot-contact offsets.
- Any garment-specific corrective behavior, reduced-motion behavior and measured rendering cost.

Some of these values can live inside the VRM; the manifest should describe requirements rather than duplicate two conflicting sources of truth. Move to runtime assembly of separate meshes only when it meaningfully reduces duplication and the part contracts are proven. A complete model is easier to validate initially, but loading all versions simultaneously would waste memory.

For a future 3D hot swap: load and validate one candidate off-screen; retain the previous model on failure; copy only compatible expression and humanoid state; initialize the candidate's spring history at its posed transforms; publish the replacement atomically; dispose the retired model. Never transfer old hair-joint velocities by array index to a different rig. Prefer a neutral settled pose for the first implementation; swapping mid-twirl requires additional transition handling and testing.

Keep only the active model and one temporary replacement resident. Cancel stale loads during rapid selection. Avoid embedding every variant in the already large renderer script. A short crossfade temporarily renders two models and must be profiled before making it the default.

## Performance and verification

Keep the existing fixed 60 Hz simulation and adaptive draw cadence. Hidden windows should remain paused. Measure additional joints, collider tests, garment vertices, materials and texture memory per look; polygon count alone does not predict cost. The current hair contact shader performs two passes over nine volumes per hair vertex, so new profiles should select necessary volumes instead of growing a global list indefinitely.

A small guide rig is the initial performance choice. Full cloth simulation is deferred until a visual comparison proves it is necessary. This is a design recommendation, not a measured speedup for the new assets. Benchmark the overlay alongside voice inference and everyday apps before accepting additional persistent workload.

For each completed 3D combination:

1. Compare neutral face and silhouette against the selected illustration at matched front, side, back and three-quarter views. Confirm identity, ears and eyes have not regressed.
2. Run all eight expressions, all 56 directed transitions, speech blending, shuffled/custom sequences and rapid interruptions.
3. Run head turns, hair sway, point, shrug, raise arms, step and slow/fast twirls. Inspect hair/face, hair/outfit, garment/body and garment/garment contact from several angles. Check hands, feet and settling after interruption.
4. Test reduced motion, pause/resume, a long idle, hidden windows, cold load, rapid repeated selection and an invalid/missing asset. Verify the last valid model survives a failed swap.
5. Record active/idle frame pacing, simulation and draw CPU time, process memory after repeated swaps and a sustained multitasking run. CPU submission time is not GPU time; short isolated renderer tests do not establish battery impact.

There are four outfits and four hairstyles if the originals are included: full independent mixing implies up to 16 combinations, not just six tests. Keep unsupported combinations visibly unavailable until validated.

Recommended authoring order: the bob with the original outfit first; utility clothing with original hair second; then ponytail, tailoring and braid; linen layers last. This proves identity preservation, part fitting and profile selection before the most difficult drape problem.
