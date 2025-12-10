"""Project configuration and settings."""

from pathlib import Path
from dataclasses import dataclass
import os

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    env: str = os.getenv("ENV", "dev")
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"


settings = Settings()
