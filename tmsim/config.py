import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("TMSIM_DATA", ROOT / "data"))
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "images").mkdir(exist_ok=True)
(DATA_DIR / "bulletins").mkdir(exist_ok=True)

_lock = threading.Lock()


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _save(name: str, data: dict) -> None:
    with _lock, open(CONFIG_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def settings() -> dict:
    return _load("settings.json")


def save_settings(data: dict) -> None:
    cur = settings()
    cur.update(data)
    _save("settings.json", cur)


def weights() -> dict:
    return _load("weights.json")


def save_weights(data: dict) -> None:
    _save("weights.json", data)
