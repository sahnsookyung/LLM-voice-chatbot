import asyncio
import base64
import json
import os
import signal
import socket
from pathlib import Path

import httpx
import numpy as np
import websockets

from .config import PROFILES, model_ready
from .text import speech_language


class SetupRequired(RuntimeError):
    pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def stop_process(process):
    if process and process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        except ProcessLookupError:
            pass


class ServerWorker:
    def __init__(self, command, env, ready_path="/health"):
        self.command, self.env, self.ready_path = command, env, ready_path
        self.process = None
        self.port = free_port()
        self.lock = asyncio.Lock()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    async def start(self):
        async with self.lock:
            if self.process and self.process.returncode is None:
                return
            command = [part.replace("{port}", str(self.port)) for part in self.command]
            log_path = Path(self.env.get("LVC_WORKER_LOG", "/tmp/lvc-worker.log"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log:  # noqa: ASYNC230 — one descriptor at worker startup
                log_path.chmod(0o600)
                self.process = await asyncio.create_subprocess_exec(
                    *command, env=self.env, stdout=log, stderr=log, start_new_session=True
                )
            try:
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    for _ in range(120):
                        if self.process.returncode is not None:
                            raise SetupRequired(
                                "An inference worker exited. Run scripts/doctor.py for setup diagnostics."
                            )
                        try:
                            response = await client.get(
                                self.url + self.ready_path,
                                headers={
                                    "Authorization": "Bearer "
                                    + self.env.get("LVC_WORKER_TOKEN", "")
                                },
                            )
                            if response.status_code == 200:
                                return
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.5)
                raise SetupRequired("Inference worker startup timed out")
            except BaseException:
                await self.close()
                raise

    async def close(self):
        await stop_process(self.process)
        self.process = None


class LocalDialogue:
    def __init__(self, config):
        self.config = config
        self.worker = None
        self.profile = None
        self.history_skip = 0
        self.history_seen = 0
        self.cache_epoch = 0
        self.history_reset_pending = False
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=5), trust_env=False)

    async def prepare(self, profile):
        if self.profile == profile and self.worker:
            await self.worker.start()
            return
        await self.unload()
        executable = self.config.runtime / "omlx/bin/omlx"
        model = self.config.models / PROFILES[profile].model
        if not executable.is_file() or not model_ready(model):
            raise SetupRequired("Dialogue models are not installed. Run scripts/setup.sh --models.")
        env = {
            **os.environ,
            "OMLX_API_KEY": self.config.token,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMLX_CACHE_ENABLED": "1",
            "OMLX_HOT_CACHE_ONLY": "1",
            "OMLX_HOT_CACHE_WRITE_THROUGH": "0",
            "OMLX_GDN_SNAPSHOT_STORAGE": "embedded",
            "LVC_WORKER_TOKEN": self.config.token,
            "LVC_WORKER_LOG": str(self.config.data / "dialogue.log"),
        }
        # Discover only the selected model; never load the voice model or an HF cache model.
        discovery = self.config.data / "dialogue-models"
        discovery.mkdir(parents=True, exist_ok=True)
        for item in discovery.iterdir():
            if item.is_symlink():
                item.unlink()
        (discovery / model.name).symlink_to(model, target_is_directory=True)
        command = [
            str(executable),
            "serve",
            "--model-dir",
            str(discovery),
            "--host",
            "127.0.0.1",
            "--port",
            "{port}",
            "--headless",
            "--base-path",
            str(self.config.data / "omlx"),
            "--hot-cache-max-size",
            "1GB",
            "--paged-ssd-cache-max-size",
            "1GB",
            "--no-hf-cache",
            "--log-level",
            "warning",
            "--max-concurrent-requests",
            "1",
            "--memory-guard-gb",
            str(PROFILES[profile].llm_limit_gb),
        ]
        if os.environ.get("LVC_DISABLE_PREFIX_CACHE") == "1":
            command.append("--no-cache")
        self.worker = ServerWorker(command, env)
        self.profile = profile
        await self.worker.start()

    async def generate(self, messages, profile):
        # A cancelled hybrid-model prefix can occasionally return EOS immediately.
        # Retry only an entirely empty response, before any words/audio are delivered.
        for attempt in range(2):
            received_text = False
            async for text in self._generate_once(messages, profile):
                received_text = received_text or bool(text.strip())
                yield text
            if received_text or attempt == 1:
                return
            self.invalidate_cache()

    async def _generate_once(self, messages, profile):
        await self.prepare(profile)
        # Exact tokenizer budgeting avoids character-count estimates for Korean/Japanese.
        history = messages[1:]
        if len(history) < self.history_seen:
            self.history_skip = 0
        self.history_seen = len(history)
        if self.history_reset_pending:
            # Rebuilding a long hybrid-model prompt after interruption is costly.
            # Retain the latest four exchanges and current input; disk history is intact.
            self.history_skip = max(self.history_skip, max(0, len(history) - 9))
            self.history_reset_pending = False
        messages = [dict(messages[0])] + [dict(m) for m in history[self.history_skip :]]
        messages[0]["content"] = (
            f"Conversation revision {self.cache_epoch}.\n" + messages[0]["content"]
        )
        for message in messages[1:]:
            if message["role"] == "user":
                language = speech_language(message["content"], "auto")
                message["content"] += (
                    f"\n[Input language: {language}. Reply in this language unless the message "
                    "explicitly asks for another language or the system sets one.]"
                )
        target = 3840
        while True:
            response = await self.client.post(
                self.worker.url + "/tokenize",
                headers=self.headers,
                json={
                    "model": PROFILES[profile].model,
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            data = response.json()
            count = data.get("count", data.get("num_tokens", len(data.get("tokens", []))))
            if not count:
                raise SetupRequired("The dialogue server did not return a usable token count")
            if count <= target:
                break
            if len(messages) <= 2:
                if count <= 3840:
                    break
                raise SetupRequired(
                    "This message and character instructions exceed the 4,096-token context. Shorten them and retry."
                )
            # Trim in larger, persistent groups. Sliding by one pair every turn destroys
            # all reusable prefixes once a long conversation reaches its context limit.
            target = 2304
            pairs = max(1, (len(messages) - 2) // 2)
            remove = min(pairs * 2, max(2, (pairs // 3) * 2))
            del messages[1 : 1 + remove]
            self.history_skip += remove
        payload = {
            "model": PROFILES[profile].model,
            "messages": messages,
            "stream": True,
            "max_tokens": 256,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self.client.stream(
            "POST", self.worker.url + "/v1/chat/completions", headers=self.headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    break
                data = json.loads(line[6:])
                choices = data.get("choices", [])
                if choices and (text := choices[0].get("delta", {}).get("content")):
                    yield text

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.config.token}"}

    def invalidate_cache(self):
        # A reply may have been fully generated/cached but only partly heard.
        # Start a fresh prefix when the delivered history diverges after a cancel.
        self.cache_epoch += 1
        self.history_reset_pending = True

    async def unload(self):
        if self.worker:
            await self.worker.close()
        self.worker, self.profile = None, None

    async def close(self):
        await self.unload()
        await self.client.aclose()


class LocalSpeech:
    def __init__(self, config):
        self.config, self.process = config, None
        self.lock = asyncio.Lock()
        self.synthesizing = False

    async def prepare(self):
        if self.process and self.process.returncode is None:
            return
        python = self.config.runtime / "tts/bin/python"
        model = self.config.models / "Qwen3-TTS-12Hz-0.6B-Base-bf16"
        if not python.is_file() or not model_ready(model):
            raise SetupRequired(
                "The local speech runtime is missing. Run scripts/setup.sh --models."
            )
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(self.config.root / "backend"),
        }
        self.config.data.mkdir(parents=True, exist_ok=True)
        log_path = self.config.data / "voice.log"
        with log_path.open("ab") as log:
            log_path.chmod(0o600)
            self.process = await asyncio.create_subprocess_exec(
                str(python),
                "-m",
                "companion.tts_worker",
                str(model),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                env=env,
                start_new_session=True,
                limit=2**20,
            )
        try:
            ready = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 90))
            if ready.get("type") != "ready":
                raise SetupRequired(ready.get("message", "Speech worker could not load"))
        except BaseException:
            await self.close()
            raise

    async def generate(self, text, language, reference, transcript):
        async with self.lock:
            await self.prepare()
            process = self.process
            request = {
                "type": "speak",
                "text": text,
                "language": language,
                "reference": str(reference),
                "transcript": transcript,
            }
            process.stdin.write((json.dumps(request) + "\n").encode())
            await process.stdin.drain()
            self.synthesizing = True
            completed = False
            try:
                while True:
                    line = await asyncio.wait_for(process.stdout.readline(), 30)
                    if not line:
                        raise RuntimeError("Speech worker stopped unexpectedly")
                    result = json.loads(line)
                    if result["type"] == "done":
                        completed = True
                        self.synthesizing = False
                        return
                    if result["type"] == "error":
                        raise RuntimeError(result["message"])
                    if result["type"] == "audio":
                        yield base64.b64decode(result["pcm"], validate=True)
            finally:
                # Cooperative cancel is observed between generator yields; enforce a deadline.
                if not completed and process.returncode is None:
                    try:
                        process.stdin.write(b'{"type":"cancel"}\n')
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Preserve the original failure if the worker already exited.

    async def cancel(self):
        if not self.process or not self.synthesizing:
            return
        try:
            self.process.stdin.write(b'{"type":"cancel"}\n')
            await self.process.stdin.drain()
            async with asyncio.timeout(0.15):
                while line := await self.process.stdout.readline():
                    if json.loads(line).get("type") == "done":
                        self.synthesizing = False
                        return
        except (TimeoutError, OSError, ValueError):
            pass
        # Enforce a hard fallback if cooperative GPU cancellation misses its deadline.
        await self.close()

    async def close(self):
        await stop_process(self.process)
        self.process = None
        self.synthesizing = False


class NemoRecognition:
    def __init__(self, config, callback):
        self.config, self.callback = config, callback
        self.worker = self.socket = self.reader = None
        self.final = asyncio.Event()
        self.text = ""
        self.failure = None
        self.segments = []

    async def start(self, language="auto"):
        if self.socket:
            return
        self.failure = None
        binary = self.config.runtime / "nemo/bin/nemo-speech"
        model = self.config.models / "nemotron-asr.gguf"
        if not binary.is_file() or not model.is_file():
            raise SetupRequired(
                "Speech recognition is not installed. Run scripts/setup.sh --models."
            )
        if not self.worker:
            self.worker = ServerWorker(
                [
                    str(binary),
                    "serve",
                    "--asr-model",
                    str(model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "{port}",
                    "--threads",
                    "2",
                    "--api-key",
                    self.config.token,
                    "--device",
                    "cpu",
                    "--no-ui",
                    "--asr.batching.enabled=false",
                    "--asr.batching.state_arena_slots=1",
                ],
                {
                    **os.environ,
                    "LVC_WORKER_TOKEN": self.config.token,
                    "LVC_WORKER_LOG": str(self.config.data / "recognition.log"),
                },
            )
        await self.worker.start()
        self.socket = await websockets.connect(
            self.worker.url.replace("http:", "ws:") + "/v1/audio/transcriptions/realtime",
            additional_headers={"Authorization": f"Bearer {self.config.token}"},
            max_size=2**20,
        )
        await self.socket.recv()
        codes = {"english": "en-US", "korean": "ko-KR", "japanese": "ja-JP", "auto": "auto"}
        await self.socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "sample_rate": 16000,
                        "language": codes[language],
                        "endpointing_ms": 0,
                    },
                }
            )
        )
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for raw in self.socket:
                event = json.loads(raw)
                kind = event.get("type", "")
                if kind.endswith(".delta"):
                    self.text = event.get("transcript", self.text + event.get("delta", ""))
                    await self.callback(" ".join(self.segments + [self.text]), False)
                elif kind.endswith(".completed"):
                    text = event.get("transcript", event.get("text", self.text)).strip()
                    if text:
                        self.segments.append(text)
                    self.text = ""
                    await self.callback(" ".join(self.segments), True)
                elif kind == "input_audio_buffer.committed":
                    self.final.set()
                elif kind == "error":
                    raise RuntimeError("Recognition server rejected the audio stream")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
            self.failure = SetupRequired("Speech recognition connection was lost")
            self.final.set()
            await self.callback("", False, error="Speech recognition connection was lost")

    async def feed(self, pcm):
        if self.socket:
            await self.socket.send(pcm)

    async def commit(self):
        self.final.clear()
        await self.socket.send('{"type":"input_audio_buffer.commit"}')
        await asyncio.wait_for(self.final.wait(), 8)
        if self.failure:
            raise self.failure
        text = " ".join(self.segments)
        self.text = ""
        self.segments.clear()
        return text.strip()

    async def disconnect(self):
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.socket:
            await self.socket.close()
        self.reader = self.socket = None
        self.text = ""
        self.segments.clear()

    async def close(self):
        await self.disconnect()
        if self.worker:
            await self.worker.close()
        self.reader = self.socket = self.worker = None


class DemoDialogue:
    """Explicit development adapter, never selected as a production fallback."""

    async def prepare(self, profile):
        pass

    def invalidate_cache(self):
        pass

    async def generate(self, messages, profile):
        for word in [
            "[mood=happy]",
            "This",
            "is",
            "the",
            "development",
            "preview.",
            "Real",
            "models",
            "are",
            "configured",
            "separately.",
        ]:
            await asyncio.sleep(0.01)
            yield word + " "

    async def unload(self):
        pass

    async def close(self):
        pass


class DemoSpeech:
    async def generate(self, text, language, reference, transcript):
        # Quiet tone deliberately sounds unlike a real model and exists only for playback tests.
        for start in range(0, 24000, 7680):
            n = min(7680, 24000 - start)
            audio = np.sin(np.arange(start, start + n) * 2 * np.pi * 220 / 24000) * 0.015
            await asyncio.sleep(0.01)
            yield (audio * 32767).astype("<i2").tobytes()

    async def cancel(self):
        pass

    async def close(self):
        pass
