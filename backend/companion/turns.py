import time


class TurnDetector:
    """Pipecat's bundled local analyzers; no remote calls or raw-audio retention."""

    def __init__(self):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams

        self.vad = SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(confidence=0.7, start_secs=0.10, stop_secs=0.25, min_volume=0.3),
        )
        self.vad.set_sample_rate(16000)
        self.turn = LocalSmartTurnAnalyzerV3(sample_rate=16000, cpu_count=1)
        self.turn.set_sample_rate(16000)
        self.speaking = False
        self.had_speech = False
        self.silence = 0
        self.last_analysis = 0

    async def feed(self, pcm: bytes):
        state = await self.vad.analyze_audio(pcm)
        speaking = state.name in {"SPEAKING", "STOPPING"}
        started = speaking and not self.speaking
        self.speaking = speaking
        if speaking:
            self.had_speech = True
            self.silence = 0
        elif self.had_speech:
            self.silence += len(pcm) / 32000
        result = self.turn.append_audio(pcm, speaking)
        complete = self.had_speech and result.name == "COMPLETE"
        now = time.monotonic()
        if (
            self.had_speech
            and not speaking
            and self.silence >= 0.3
            and now - self.last_analysis >= 0.3
        ):
            self.last_analysis = now
            result, _ = await self.turn.analyze_end_of_turn()
            complete = complete or result.name == "COMPLETE" or self.silence >= 2.5
        if complete:
            self.had_speech = False
            self.silence = 0
            self.turn.clear()
        return started, complete

    async def close(self):
        await self.turn.cleanup()
        await self.vad.cleanup()
