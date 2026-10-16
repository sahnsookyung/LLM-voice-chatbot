"""A dedicated, cancellable MLX process: complete utterances, never microphone streaming."""

import asyncio
import base64
import json
import os

from .adapters import SetupRequired, stop_process
from .config import model_ready

MODEL = "Qwen3-ASR-1.7B-8bit"
REVISION = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"


class FinalRecognition:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.lock = asyncio.Lock()

    async def prepare(self):
        async with self.lock:
            await self._prepare()

    async def _prepare(self):
        if self.process and self.process.returncode is None:
            return
        model = self.config.models / MODEL
        python = self.config.runtime / "tts/bin/python"
        if not model_ready(model) or not python.is_file():
            raise SetupRequired(
                "Accurate recognition needs Qwen3-ASR. Run scripts/download_models.py final-asr."
            )
        log_path = self.config.data / "final-recognition.log"
        with log_path.open("ab") as log:
            log_path.chmod(0o600)
            self.process = await asyncio.create_subprocess_exec(
                str(python),
                "-m",
                "companion.asr_worker",
                str(model),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                start_new_session=True,
                limit=2**20,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.config.root / "backend"),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
        try:
            response = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 90))
            if response.get("type") != "ready":
                raise SetupRequired("The final recognition worker could not load.")
        except BaseException:
            await self.close()
            raise

    async def transcribe(self, pcm, settings):
        async with self.lock:
            await self._prepare()
            request = {
                "pcm": base64.b64encode(pcm).decode(),
                "language": settings.input_language,
                "vocabulary": settings.vocabulary,
            }
            try:
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                await self.process.stdin.drain()
                result = json.loads(await self.process.stdout.readline())
                if result.get("type") != "final":
                    raise SetupRequired("Final recognition failed. Retry, edit, or use the draft.")
                return result["text"].strip()
            except BaseException:
                # MLX calls are synchronous. Terminating this isolated worker makes
                # cancellation definitive and prevents a result reaching a later capture.
                await self.close()
                raise

    async def close(self):
        process, self.process = self.process, None
        await stop_process(process)
