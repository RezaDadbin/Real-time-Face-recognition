from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class AppConfig:
    device: str
    database_path: Path
    recognition_threshold: float
    camera_index: int
    window_titles: dict

def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        device=raw.get("device", "cpu"),
        database_path=Path(raw.get("database_path", "face_data")),
        recognition_threshold=float(raw.get("recognition_threshold", 1.0)),
        camera_index=int(raw.get("camera_index", 0)),
        window_titles=raw.get("window_titles", {}),
    )
