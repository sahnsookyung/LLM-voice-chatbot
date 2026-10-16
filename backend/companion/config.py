import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("LVC_ROOT", Path(__file__).resolve().parents[2]))


def model_ready(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    try:
        files = (
            set(json.loads(index.read_text())["weight_map"].values())
            if index.exists()
            else {"model.safetensors"}
        )
        return bool(files) and all((path / name).is_file() for name in files)
    except (ValueError, KeyError, OSError):
        return False


@dataclass(frozen=True)
class Profile:
    model: str
    budget_gb: int
    llm_limit_gb: int


PROFILES = {
    "everyday": Profile("Qwen3.5-9B-4bit", 18, 10),
    "quiet": Profile("Qwen3.5-4B-4bit", 12, 6),
}


@dataclass
class Config:
    root: Path = ROOT
    data: Path = Path(
        os.environ.get("LVC_DATA", Path.home() / "Library/Application Support/LocalVoiceCompanion")
    )
    token: str = os.environ.get("LVC_TOKEN", "")
    demo: bool = False
    idle_seconds: float = 600
    audio_window_samples: int = 24000
    codex_validation_thread: str | None = None

    @property
    def models(self) -> Path:
        return self.root / ".models"

    @property
    def runtime(self) -> Path:
        return self.root / ".runtime"
