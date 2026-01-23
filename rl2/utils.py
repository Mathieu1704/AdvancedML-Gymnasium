from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_nonempty_dir(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def make_run_dir(base_dir: str, run_name: str) -> str:
    os.makedirs(base_dir, exist_ok=True)

    candidate = os.path.join(base_dir, run_name)
    if _is_nonempty_dir(candidate):
        k = 1
        while True:
            cand2 = os.path.join(base_dir, f"{run_name}_{k:03d}")
            if not os.path.exists(cand2):
                candidate = cand2
                break
            k += 1

    os.makedirs(candidate, exist_ok=True)
    os.makedirs(os.path.join(candidate, "checkpoints"), exist_ok=True)

    # Pointeur "dernier run"
    latest_path = os.path.join(base_dir, "LATEST.txt")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(candidate)
    except OSError:
        pass

    return candidate


class SafeSummaryWriter:
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self._writer: Optional[object] = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir)
            except Exception:
                self.enabled = False
                self._writer = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, scalar_value, global_step)
        except (PermissionError, OSError):
            self.enabled = False
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._writer = None
        self.enabled = False
