# Local protocol v1

The native client generates a random token for each launch, starts Python on an ephemeral `127.0.0.1` port, and supplies `Authorization: Bearer …` on HTTP and WebSocket requests. Browser origins are rejected. No token is placed in a URL. One active session owns the inference workers. `/health` includes `session_active`; a reconnect waits for cleanup. Local same-user processes are outside this authentication threat boundary.

`/session` accepts JSON commands and binary microphone packets. Microphone packets are mono signed PCM16 little-endian, 16,000 Hz, at most 8,192 bytes. The native client normally sends 1,024-frame capture conversions. The bounded ingress queue stops listening visibly if processing falls behind.

Commands: `text` (`text`, `speak`), `start` (`mode`: `ptt`/`handsfree`), `commit`, `mute`, `stop`, `ack` (`turn_id`, `samples`), `profile` (`quiet`/`everyday`), `language` (`auto`/`english`/`korean`/`japanese`), `character` (`character_id`), `pressure` (`normal`/`warning`/`critical`), `avatar` (`avatar_mode`: `illustration`/`model`, `avatar_visible`: boolean, `reduced_motion`: boolean). Unknown fields and invalid values are rejected. The native client sends avatar state on connection, appearance/visibility changes, and before text or microphone input. A session without this handshake defaults to illustration mode and does not emit body gestures.

Events: `ready`, `turn`, `state`, `listening`, `transcript`, `text`, `expression`, `gesture`, `cancel`, `done`, `profile`, `character`, `metrics`, `notice`, `error`. Generated events carry a turn UUID. Client state must discard audio from other turns.

The dialogue model is instructed to begin every reply with one `[mood=EXPRESSION]` tag, choosing its own expression from `neutral`, `happy`, `curious`, `concerned`, `surprised`, `annoyed`, `embarrassed`, and `sleepy`. The streaming parser removes this metadata before display, speech synthesis, and history storage; missing or unknown expressions fall back to neutral. Once the prefix is resolved, one `expression` event carries the expression and `sample_offset: 0`, which the native client applies immediately for that turn, including text-only replies. No separate sentiment model runs. Emotion portraits and the 3D model show these expressions; the full-body illustration remains a fixed neutral image.

The parsed expression is also passed separately to Fish S2 Pro's speech worker. The worker creates allowlisted delivery tags and converts a happy standalone laugh into a chuckle; these Fish controls never appear in the displayed/stored transcript. Synthesis is phrase-based in the pinned MLX runtime. Native playback continues to receive 24 kHz mono signed 16-bit PCM, resampled from Fish's 44.1 kHz output before packetization. Backpressure, playback acknowledgments and turn cancellation are unchanged. The first PCM wait is bounded to 120 seconds per provider segment; Stop checks generation at semantic-frame boundaries and retains a 150 ms cooperative-cancel deadline before killing the worker.

The reply header continues with `[intent=greeting|explaining|acknowledging|uncertain|conversation]` and `[action=none|ACTION]`. These are literal selected values, not pipe-separated output. Supported requested action IDs are `wave`, `nod`, `explain`, `shift`, `step`, `twirl`, `raise-arms`, `shrug`, and `point`. `dance` and `laugh` are not action IDs. Requested actions take precedence over probabilistic conversational gestures; concerned/annoyed/sleepy replies suppress all body actions, including model-requested ones. `none` retains normal conversational selection. Unsupported action values fail closed. Legacy headers without intent/action still parse, and repeated tags are removed without retriggering actions.

Explicit actions also require matching movement words in the latest user message. A conservative English/Korean/Japanese filter vetoes translation, story and negation cues, and limits the prompt's action choices to matching capabilities. The model still decides whether a request is actually being made; the filter prevents unrelated tags (for example, `twirl` for a laugh request) from moving the avatar. It is not a complete language parser: uncommon phrasing and mixed requests may need to be restated directly. Ordinary intent-based conversational gestures retain their existing probability and cooldown rules.

At most one `gesture` event is sent per reply, carrying `gesture` and `wait_for_audio`. Native playback validates the ID, queues spoken gestures until audio begins, applies text-only gestures immediately and discards pending/stale events on interruption. The backend advertises/emits body actions only for visible Ari in model mode with motion enabled. This is a bounded avatar animation interface, not arbitrary model-selected code execution. New animations must be added to the renderer, native allowlist and backend capability list before the model may claim them. Voice capability text is composed after voice-reference resolution, so text fallback does not promise audible speech. Model wording may still be wrong; the action gate remains authoritative.

Server audio packets contain a 32-byte header followed by mono 24,000 Hz PCM16:

| Offset | Field |
| --- | --- |
| 0 | Four ASCII bytes `LVA1` |
| 4 | 16 raw UUID bytes in RFC display order |
| 20 | uint32 little-endian sequence, starting at 0 |
| 24 | uint64 little-endian starting sample offset |
| 32 | PCM payload |

The native player acknowledges the cumulative sample position only from AVAudioPlayerNode's `dataPlayedBack` callback. The server caps unacknowledged audio at 24,000 samples (one second); this is a backpressure window, not a startup buffer. It persists each completed spoken clause after its ending sample is acknowledged. An interrupted partial clause is conservatively omitted from future context. Generated text remains visible in history with an interruption marker.

Interrupted replies contribute only delivered text to assistant context, including an empty assistant turn when nothing was heard. Keeping user turns separate avoids combining conflicting language requests. The next user message carries an interruption note. The model never receives an invented assistant utterance consisting solely of an interruption marker. An empty model response produces a visible error instead of a successful empty turn.

Cancel is sent before worker cleanup. TTS first attempts cooperative cancellation, including an MLX synchronization, with a 150 ms acknowledgment deadline. If that fails, its process group is terminated. A stale generation cannot enqueue playback for the next turn. This deadline is a control-path policy, not a claim that physical acoustic cutoff was measured.

HTTP resources: `/conversations`, `/conversations/{id}` (GET/DELETE), `/memories` (GET/POST), `/memories/{id}` (PUT/DELETE), `/characters` (GET/POST), `/characters/{id}`, `/characters/{id}/persona` (PUT), `/characters/{id}/assets/{path}`, `/characters/import` (multipart POST), `/export`. Memory writes require explicit `approved: true`. Deleting or editing a memory stops current generation and unloads dialogue caches.
