from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models/saved"))
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                if k in ("data_dir", "models_dir"):
                    setattr(cfg, k, Path(v))
                else:
                    setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        cfg.random_state = int(os.getenv("DATALAB_RANDOM_STATE", cfg.random_state))
        cfg.test_size = float(os.getenv("DATALAB_TEST_SIZE", cfg.test_size))
        cfg.cv_folds = int(os.getenv("DATALAB_CV_FOLDS", cfg.cv_folds))
        return cfg

    def ensure_dirs(self) -> None:
        for path in (self.data_dir / "raw", self.data_dir / "processed", self.models_dir):
            path.mkdir(parents=True, exist_ok=True)
