import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

@dataclass
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def load_run_scalars(run_dir: str) -> Dict[str, ScalarSeries]:
    if not os.path.exists(run_dir):
        print(f"[WARN] Dossier introuvable : {run_dir}")
        return {}

    files = [f for f in os.listdir(run_dir) if "tfevents" in f]
    if not files:
        print(f"[WARN] Pas de fichier tfevents dans {run_dir}")
        return {}
    
    # On prend le plus récent ou le premier
    files.sort(key=lambda x: os.path.getmtime(os.path.join(run_dir, x)))
    event_path = os.path.join(run_dir, files[-1])
    print(f"Chargement : {event_path}")

    # Charge tout
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=np.float32)
        data[tag] = ScalarSeries(steps, values)
    
    return data

def merge_runs(run_dirs: List[str]) -> Dict[str, ScalarSeries]:
    merged_data = {}

    for run_dir in run_dirs:
        run_data = load_run_scalars(run_dir)
        
        for tag, series in run_data.items():
            if tag not in merged_data:
                merged_data[tag] = series
            else:
                current = merged_data[tag]
                
                # Concaténation
                new_steps = np.concatenate([current.steps, series.steps])
                new_values = np.concatenate([current.values, series.values])
                
                # Tri pour éviter les désordres si chevauchement
                sort_idx = np.argsort(new_steps)
                merged_data[tag] = ScalarSeries(new_steps[sort_idx], new_values[sort_idx])

    return merged_data


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
    print(f"Sauvegardé : {path}")


def plot_rewards_merged(series: ScalarSeries, out_dir: str, window: int = 100):
    x = series.steps
    y = series.values
    
    m, s = _rolling_mean_std(y, window=window)

    plt.figure()
    plt.title(f"Rewards per episode + moving average ({window})")
    plt.xlabel("Episode (approx / Step count)")
    plt.ylabel("Return")
    
    # Raw
    plt.plot(x, y, alpha=0.25, linewidth=1, label="Return (raw)")
    # Mean
    plt.plot(x, m, linewidth=2, label=f"Moving average ({window})")
    # Std zone
    plt.fill_between(x, m - s, m + s, alpha=0.2, label="± std (rolling)")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(out_dir, f"merged_rewards_ma{window}.png"))

def plot_loss_merged(series: ScalarSeries, out_dir: str, smooth_steps: int = 2000):
    x = series.steps
    y = series.values

    w = max(1, int(len(y) * (smooth_steps / max(1, x[-1]))))
    m, _ = _rolling_mean_std(y, window=max(1, w))

    plt.figure()
    plt.title("Training loss (Huber) vs step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y, alpha=0.25, linewidth=1, label="Loss (raw)")
    plt.plot(x, m, linewidth=2, label="Loss (smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(out_dir, "merged_loss.png"))

def plot_epsilon_merged(series: ScalarSeries, out_dir: str):
    x = series.steps
    y = series.values
    
    plt.figure()
    plt.title("Epsilon schedule")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.plot(x, y, linewidth=2)
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(out_dir, "merged_epsilon.png"))

def plot_eval_return_merged(series: ScalarSeries, out_dir: str):
    x = series.steps
    y = series.values
    
    plt.figure()
    plt.title("Eval mean return vs step")
    plt.xlabel("Step")
    plt.ylabel("Mean return")
    plt.plot(x, y, marker="o", linewidth=2)
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(out_dir, "merged_eval_mean_return.png"))

def plot_q_values_merged(merged_data: Dict[str, ScalarSeries], out_dir: str):
    candidates = ["train/q_mean", "train/q_max_mean", "train/q_sa_mean"]
    present = [t for t in candidates if t in merged_data]
    
    if not present:
        print("[INFO] Pas de Q-values trouvées.")
        return

    plt.figure()
    plt.title("Q-value statistics vs step")
    plt.xlabel("Step")
    plt.ylabel("Q-value")
    
    for tag in present:
        series = merged_data[tag]
        plt.plot(series.steps, series.values, linewidth=2, label=tag)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(out_dir, "merged_q_values.png"))


# --- 5. Main ---

def main():
    run_folders = [
        "runs/lunar_pixels_best_partie1",
        "runs/lunar_pixels_best_partie2"
    ]
    output_dir = "runs/merged_plots_full"

    print(f"--- Fusion de {len(run_folders)} runs ---")
    merged_data = merge_runs(run_folders)

    if "train/episode_return" in merged_data:
        plot_rewards_merged(merged_data["train/episode_return"], output_dir, window=100)
    
    if "train/loss" in merged_data:
        plot_loss_merged(merged_data["train/loss"], output_dir)
    
    if "train/epsilon" in merged_data:
        plot_epsilon_merged(merged_data["train/epsilon"], output_dir)
    
    if "eval/mean_return" in merged_data:
        plot_eval_return_merged(merged_data["eval/mean_return"], output_dir)

    plot_q_values_merged(merged_data, output_dir)

    print("Terminé.")

if __name__ == "__main__":
    main()