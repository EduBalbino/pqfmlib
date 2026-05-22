"""Small I/O helpers used across the library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def save_circuit_draw(qc_t, base_folder: str, stem: str) -> None:
    """Save text and, when available, matplotlib circuit drawings."""
    folder = Path(base_folder)
    try:
        (folder / f"{stem}.txt").write_text(str(qc_t.draw(output="text", fold=-1)), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] Could not save text circuit drawing: {exc}")

    try:
        qc_t.draw(output="mpl", filename=str(folder / f"{stem}.png"), fold=-1, idle_wires=False)
    except Exception as exc:
        print(f"[warn] Could not save matplotlib circuit drawing: {exc}")
