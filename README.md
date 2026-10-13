# Local Voice Companion

A native macOS conversation companion for Apple Silicon, with English, Korean, and Japanese speech, a floating animated character, and local history. Built for a 48 GB M4 Pro Mac while leaving room for everyday apps.

**Built with Fish Audio.** Speech runs locally with **Fish Audio S2 Pro, 8-bit MLX**. The model is free for personal/non-commercial use under its [Research License](third_party/fish-audio/LICENSE.md); commercial use requires a separate Fish Audio license.

Open **`build/Local Companion.app`** after setup. Click **Talk**, speak, then click **Send voice**, or hold **Control–Option–Space** to talk from any app. The microphone meter shows input level. Hands-free is explicit and stays on until muted. Escape stops a reply. The menu-bar sparkle restores the chat/character, toggles click-through, mutes, and quits.

The character starts near the lower-right corner of the chat's display. Use **Show avatar / Hide avatar** in the app header, **Show desktop avatar** in the Character tab, or **Command–Shift–H** while the app is active. The menu-bar sparkle also has the toggle. Visibility survives restart; hiding pauses rendering, and showing retains the current position and look. The source `renderer/index.html` requires Vite; opening it directly is not a way to launch the app.

The app starts with the microphone off. Audio capture and playback use AVAudioEngine. Talk uses direct microphone capture; hands-free enables echo cancellation. The Python service and inference workers bind only to loopback and require a per-launch token. There is no browser automation, tool execution, cloud inference, or automatic memory collection.

Use **Character → Open character studio** (or **Character → Open character studio** in the menu bar) to explore the full-body art, eight expressions, and the optional VRoid 3D model. Appearance and Framing also control the floating character. The editable VRoid model has facial expressions, finger articulation and hair springs; its likeness is still unfinished. See [asset details and limitations](docs/ARI-3D.md).

In the **Character** tab, **Desktop wardrobe** selects the overlay's outfit, hairstyle, glasses and headwear independently. All 15 outfits and nine hairstyles are available; choosing a look switches to 3D and saves it immediately, even while the avatar is hidden. **Coordinated beret** follows the outfit's main fabric color, with a darker band and a secondary-color ribbon, including after startup randomization. The beret is unavailable with double buns. Studio keeps its own preview choices so experimenting there does not change the desktop look.

**Randomize outfit on startup** is on by default. Each app launch picks a different outfit while keeping hair, accessories, appearance mode and visibility as selected. Turn it off to retain the current outfit across launches; this preference is saved. Opening a tab or reconnecting does not reroll the outfit.

The Character tab's 3D gesture buttons include **Cute twirl**, **Shrug**, **Raise Arms** and **Point**. Replies also carry conversational intent: greetings wave, explanations use Explain, acknowledgements sometimes nod, and uncertainty sometimes shrugs. Ordinary conversation occasionally shifts or steps. Optional movements have both turn and time cooldowns; concerned, annoyed and sleepy replies suppress automatic gestures. Spoken gestures wait for playback to start; text-only replies gesture immediately. Reduced Motion still applies, and interruption cancels the previous gesture smoothly.

You can also ask Ari to wave, nod, shrug, point, raise her arms or twirl. Reply action tags select one supported animation and use the same speech/cancellation timing. A friendly laugh request can produce a synthesized chuckle with a happy expression. A dance request currently offers a twirl; a full dance routine and dedicated laugh-body animation are not implemented. Hidden avatars, illustration mode and Reduced Motion disable requested body actions, and Ari is told that current view state. These capabilities are supplied separately from your saved personality.

Reply mood tags drive both the face and Fish's vocal delivery. New turns clear the previous mood; internal mood/intent/action tags are removed from displayed and spoken text. Only a happy standalone laughter interjection becomes a chuckle event; ordinary happy sentences and serious replies do not automatically laugh. Acting instructions are confined to synthesis and never stored in the chat transcript. These controls guide the model but do not guarantee perfect emotional delivery or pronunciation.

The studio's **Reference look** selector and **Wardrobe & Hair** gallery include three new outfits and three hairstyles. Select a look to swap the studio illustration or compare it alongside VRoid. **Auto reload** picks up saved workspace artwork within about three seconds; **Reload artwork** refreshes it immediately. These are complete reference illustrations, not interchangeable 3D clothing/hair meshes, and the selection is local to the studio. See [the look library and authoring workflow](docs/ARI-LOOKS.md).

Choose **Character → Voice** to switch between Original, Bright, and Soft, then use **Preview voice** to audition a Fish-generated sample. Bright and Soft are original higher-pitched adult voices generated locally; selection is saved per character. Their original reference clips are retained for cloning. Imported voices without a Fish preview play their reference clip.

The pinned MLX Fish implementation generates a complete phrase before returning audio. Playback then uses bounded 24 kHz packets, with Stop checked during generation and a process-termination fallback. This is phrase-based synthesis, not Fish's CUDA streaming server; expect more startup delay than the old Qwen voice. Both voices and dialogue remain offline after installation.

## Setup and build

Requirements: Apple Silicon, macOS 15+, Xcode with its Metal toolchain, Node.js, and [uv](https://docs.astral.sh/uv/). Setup uses Python 3.12 and isolated inference environments. Allow roughly 35 GB of disk for runtime models, creation weights, dependencies, and caches. Fish's weights and codec alone download about 6.7 GB.

```sh
xcodebuild -downloadComponent MetalToolchain
./scripts/setup.sh --models

# Creation-only download and original voice generation; run with conversations closed.
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_models.py design
HF_HUB_OFFLINE=1 .runtime/tts/bin/python scripts/create_voice.py

./scripts/build.sh
open "build/Local Companion.app"
```

Ari's generated reference is included in `characters/ari/voice.wav`; regenerating it is optional. The app bundle stores this workspace's absolute path and launches its Python environment. Keep the workspace in place or rebuild after moving it. This is an ad-hoc-signed local development app, not a standalone notarized installer. It does not copy gigabytes of models inside the `.app`.

For an existing installation, download the new speech model with `HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_models.py voice`, then quit and reopen the app. Voice selections and history are preserved; old Qwen runtime weights are no longer loaded or automatically deleted.

```sh
.venv/bin/python scripts/doctor.py
.venv/bin/pytest -q
.venv/bin/ruff check backend tests scripts
npm --prefix renderer run build
swiftc native/Sources/Companion/Audio.swift scripts/check_audio.swift -o .build/check-audio
.build/check-audio
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swift scripts/check_renderer.swift "$PWD/build/Local Companion.app/Contents/Resources/renderer/index.html"
```

If setup fails during a network interruption, rerun the model download command; completed files and partial transfers are reused. No model is downloaded silently during a conversation. Missing components produce a visible setup message. First use includes cold model loading and is much slower than subsequent turns.

## What is implemented

- Native SwiftUI conversation, history, approved-memory editing/deletion, export, personality editor, language/profile controls, and global push-to-talk.
- Floating nonactivating AppKit panel, full-body and portrait illustrations, and an optional Three.js/VRM character with expressions, gaze, blinking, speech-driven mouth motion and gestures. Reduced Motion and hidden-window throttling are supported; imported layered art retains Pixi.
- Original adult character Ari, an in-app palette/personality character creator, plus a validated layered-PNG/voice ZIP import format. Live2D and MCP are not required.
- CPU streaming Nemotron recognition, Pipecat Silero/Smart Turn, oMLX 4-bit dialogue, and an isolated persistent Fish S2 Pro 8-bit MLX voice worker.
- Bounded input/output queues, turn identifiers, actual-playback acknowledgments, interruption handling, idle unloading, and memory-pressure response.
- SQLite history using acknowledged speech for future assistant context. No embeddings or automatically inferred personal memory.

Data normally lives in `~/Library/Application Support/LocalVoiceCompanion`. `LVC_DATA` can point to a separate test directory. Deleting a saved memory removes that fact from the memory collection and invalidates derived caches; it does not erase old conversation transcripts or facts repeated in them. Delete the relevant conversations too if you want those historical references removed.

## Benchmarks and limits

```sh
# Real models; simulates timed playback, never opens a microphone.
PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --profile everyday --turns 6
PYTHONPATH=backend .venv/bin/python scripts/test_voice_pipeline.py
PYTHONPATH=backend .venv/bin/python scripts/check_fish_speech.py --transcribe
PYTHONPATH=backend .runtime/tts/bin/python -m unittest discover -s tests -p test_pcm_conversion.py
PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --profile everyday --turns 100 --with-recognition --interrupt-every 10

# Optional 30-minute pacing, while using your normal desktop workload.
PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --profile everyday --turns 100 --duration 1800 --with-recognition
```

Reports are local JSON in `reports/local-*`. They label cold starts separately from warm distributions and do not equate first generated PCM with physical speaker latency. Microphone permission, acoustic echo cancellation, headset/device changes, Korean/Japanese pronunciation quality, and a 30-minute run under your actual desktop workload require live-device acceptance testing. The implementation enables the relevant audio APIs but does not claim those human/device-dependent checks passed automatically.

For a model-free wiring check, launch the executable with `--demo`. It clearly labels itself **Development preview** and plays quiet test tones. Production never falls back to a canned response.

See [validation results](docs/VALIDATION.md), [character authoring](docs/CHARACTERS.md), [models and licenses](docs/MODELS.md), and [the local protocol](docs/PROTOCOL.md).

## Layout

- `native/`: SwiftUI/AppKit app, AVAudioEngine capture/playback, WebSocket client, process supervision.
- `backend/companion/`: authenticated API, turn state, model adapters, isolated speech worker, storage/import validation.
- `renderer/`: local TypeScript illustration, Three.js 3D preview, and Pixi import playback.
- `characters/ari/`: original character manifest, generated voice, and provenance.
- `scripts/`: setup, verified downloads, build, diagnostics, voice creation, and benchmark.
- `tests/`: protocol, history, import, authentication, parsing, and playback/cancellation checks.
