from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def _load_scalars(log_dir: str) -> Dict[str, ScalarSeries]:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Run directory not found: {log_dir}")

    ea = EventAccumulator(
        log_dir,
        size_guidance={
            "scalars": 0,  
            "images": 0,
            "histograms": 0,
            "tensors": 0,
        },
    )
    ea.Reload()

    out: Dict[str, ScalarSeries] = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        steps = np.array([e.step for e in evs], dtype=np.int64)
        vals = np.array([e.value for e in evs], dtype=np.float32)
        out[tag] = ScalarSeries(steps=steps, values=vals)

    return out


def _rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1:
        return x.copy(), np.zeros_like(x)

    mean = np.empty_like(x, dtype=np.float32)
    std = np.empty_like(x, dtype=np.float32)

    for i in range(len(x)):
        j0 = max(0, i - window + 1)
        w = x[j0 : i + 1]
        mean[i] = float(np.mean(w))
        std[i] = float(np.std(w))
    return mean, std


def _savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_rewards_episode_style(
    scalars: Dict[str, ScalarSeries],
    out_dir: str,
    window: int = 100,
) -> None:
    tag = "train/episode_return"
    if tag not in scalars:
        print(f"[WARN] Missing scalar: {tag} (rien à tracer)")
        return

    y = scalars[tag].values
    ep = np.arange(1, len(y) + 1, dtype=np.int64)  

    m, s = _rolling_mean_std(y, window=window)

    plt.figure()
    plt.title(f"Rewards per episode + moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(ep, y, alpha=0.25, linewidth=1, label="Return (raw)")
    plt.plot(ep, m, linewidth=2, label=f"Moving average ({window})")
    plt.fill_between(ep, m - s, m + s, alpha=0.2, label="± std (rolling)")
    plt.legend()

    _savefig(os.path.join(out_dir, f"rewards_ma{window}.png"))


def plot_loss(
    scalars: Dict[str, ScalarSeries],
    out_dir: str,
    smooth_steps: int = 2000,
) -> None:
    tag = "train/loss"
    if tag not in scalars:
        print(f"[WARN] Missing scalar: {tag} (rien à tracer)")
        return

    x = scalars[tag].steps
    y = scalars[tag].values

    w = max(1, int(len(y) * (smooth_steps / max(1, x[-1]))))
    m, _ = _rolling_mean_std(y, window=max(1, w))

    plt.figure()
    plt.title("Training loss (Huber) vs step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y, alpha=0.25, linewidth=1, label="Loss (raw)")
    plt.plot(x, m, linewidth=2, label="Loss (smoothed)")
    plt.legend()

    _savefig(os.path.join(out_dir, "loss.png"))


def plot_epsilon(
    scalars: Dict[str, ScalarSeries],
    out_dir: str,
) -> None:
    tag = "train/epsilon"
    if tag not in scalars:
        print(f"[WARN] Missing scalar: {tag} (rien à tracer)")
        return

    x = scalars[tag].steps
    y = scalars[tag].values

    plt.figure()
    plt.title("Epsilon schedule")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.plot(x, y, linewidth=2)
    _savefig(os.path.join(out_dir, "epsilon.png"))


def plot_eval_mean_return(
    scalars: Dict[str, ScalarSeries],
    out_dir: str,
) -> None:
    tag = "eval/mean_return"
    if tag not in scalars:
        print(f"[WARN] Missing scalar: {tag} (rien à tracer)")
        return

    x = scalars[tag].steps
    y = scalars[tag].values

    plt.figure()
    plt.title("Eval mean return vs step")
    plt.xlabel("Step")
    plt.ylabel("Mean return")
    plt.plot(x, y, marker="o", linewidth=2)
    _savefig(os.path.join(out_dir, "eval_mean_return.png"))


def plot_q_values_if_present(
    scalars: Dict[str, ScalarSeries],
    out_dir: str,
) -> None:
    candidates = ["train/q_mean", "train/q_max_mean", "train/q_sa_mean"]
    present = [t for t in candidates if t in scalars]
    if not present:
        print("[INFO] Aucun tag Q-value trouvé (train/q_mean, train/q_max_mean, ...).")
        return

    plt.figure()
    plt.title("Q-value statistics vs step")
    plt.xlabel("Step")
    plt.ylabel("Q-value")
    for t in present:
        plt.plot(scalars[t].steps, scalars[t].values, linewidth=2, label=t)
    plt.legend()
    _savefig(os.path.join(out_dir, "q_values.png"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="ex: runs/carracing_pixels ou runs/<run_name>")
    p.add_argument("--window", type=int, default=100, help="moving average window (episodes)")
    p.add_argument("--out-subdir", type=str, default="plots", help="subfolder created inside run-dir")
    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = os.path.join(run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    scalars = _load_scalars(run_dir)

    plot_rewards_episode_style(scalars, out_dir, window=args.window)
    plot_loss(scalars, out_dir)
    plot_epsilon(scalars, out_dir)
    plot_eval_mean_return(scalars, out_dir)
    plot_q_values_if_present(scalars, out_dir)

    print(f"Done. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
