from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """Dict with attribute access for nested sections."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value

    def path(self, *keys: str) -> Path:
        node = self
        for k in keys:
            node = node[k]
        p = Path(node)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    for key in ("interim", "features", "outputs"):
        cfg.path("paths", key).mkdir(parents=True, exist_ok=True)
    return cfg
