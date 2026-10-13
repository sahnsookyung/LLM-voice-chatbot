# Character authoring

Ari now has a full-body illustrated design, eight emotion portraits, and an optional articulated VRoid 3D model. Use Character → Open character studio to compare them and test motion. The illustration remains the default while the model’s likeness is refined. See [the full-body implementation and limitations](ARI-3D.md). Imported layered PNG and palette characters continue to use the original Pixi rig in `renderer/src/legacy.ts`.

The vector rig is deliberately editable code. It is not a Cubism model, a photorealistic talking head, or phoneme-accurate lip sync. The mouth follows an RMS envelope, which works across English, Korean, and Japanese without pretending to identify phonemes.

## Create a character in the app

Character → Create character makes a new local character using the built-in rig. Choose a name, personality, and Teal/Plum/Copper/Midnight palette. Reusing Ari’s original generated voice is explicit; uncheck it for text-only replies until you add a reference. This is deterministic rig customization, not prompt-to-image generation.

## Import your own art

Choose Character → Import character and select a ZIP with `character.json` at its root, plus PNG layers and an optional mono PCM16 voice reference. The archive cannot contain scripts, HTML, symlinks, absolute paths, or parent traversal. Limits: 64 MiB compressed, 128 MiB extracted, 150 entries, 100 layers, 64 MiB decoded image storage, maximum individual dimension 4,096 px. Prefer two 2,048 px atlases or less for new art; the initial vector rig needs no texture atlas.

Minimal example:

```json
{
  "version": 1,
  "id": "my-character",
  "name": "My character",
  "persona": "An adult, calm and curious conversation companion.",
  "attribution": "Original artwork by me.",
  "width": 800,
  "height": 1000,
  "layers": [
    {"id":"body", "image":"body.png", "role":"static", "z":0},
    {"id":"face", "image":"face.png", "role":"head", "z":1},
    {"id":"eyes", "image":"eyes.png", "role":"eyes", "x":400, "y":360, "pivot_x":100, "pivot_y":20, "z":2},
    {"id":"mouth-closed", "image":"mouth-closed.png", "role":"mouth", "openness":0, "x":380, "y":470, "z":3},
    {"id":"mouth-open", "image":"mouth-open.png", "role":"mouth", "openness":1, "x":380, "y":470, "z":3}
  ],
  "voice": {"audio":"voice.wav", "transcript":"The exact words spoken in this reference."}
}
```

Layer coordinates are in the common canvas coordinate system. All variants need the same scale and alignment. Eyes scale about their declared pivot when blinking. Mouth variants select the nearest `openness`; optional `expression` selects one of `neutral`, `happy`, `curious`, `concerned`, `surprised`, `annoyed`, `embarrassed`, `sleepy`. Keep brows/eyes separate from the face so the face does not jump between expressions. The imported sprite path currently supports expression visibility, blinking, and mouth variants; Ari's procedural head motion is specific to her vector rig.

Generate one approved master design, then edit that master to obtain consistent layers and expression variants. Independently prompting eight portraits often changes face shape, costume, camera angle, and lighting; the app cannot repair those mismatches automatically. A local image model such as FLUX.2 klein can be an optional creation tool, but diffusion generation and automatic image-to-rig conversion are not installed in the conversation runtime. A generated flat image still needs layer separation and visual review before it is a useful animated character. That is the boundary between implemented runtime/import support and a future visual authoring workbench.

## Create an original voice locally

The Character page now has a per-character voice picker and preview button. Original retains the character's own reference; Bright and Soft are generated original adult voices with higher pitch and a sweeter delivery. They reuse the existing 0.6B Base speech worker, so no extra voice model remains loaded. Their descriptions, seeds, and model revision live in `voices/*/voice.json`. Regenerate missing presets with `HF_HUB_OFFLINE=1 .runtime/tts/bin/python scripts/create_voice_presets.py`; existing presets are preserved.

The revised anime master reference is in `characters/ari/design/anime-reference-v2.png`, with its source references and full built-in image-generation prompt alongside it. It is a design reference, not yet separated animation layers.

Quit active conversations before running the creation model. It is a separate process and is never loaded by the app.

```sh
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_models.py design
HF_HUB_OFFLINE=1 .runtime/tts/bin/python scripts/create_voice.py
```

The script generates a 2–20 second reference with a fixed seed and a written description of an original adult voice, saves its exact transcript and model provenance, and updates Ari's manifest. Use `--character`, `--description`, and `--text` to author another bundle. A user-supplied recording can replace the generated reference when you have permission to use that voice. The Base TTS model consumes this reference at runtime; it does not supply preset voices.

Live2D would be an optional separate renderer with a `.moc3`/texture/physics bundle authored in Cubism. MCP is an automation interface, not a prerequisite for generating art or integrating a renderer. Automated raster art alone does not produce a well-rigged Live2D model.
