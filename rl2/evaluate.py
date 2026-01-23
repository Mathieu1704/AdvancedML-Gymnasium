from __future__ import annotations

import argparse
import os
import re
import json
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


from rl2.dqn import CNNQNetwork
from rl2.wrappers import make_pixels_only_env, PixelPipelineConfig, obs_to_numpy_u8


def _pick_latest_checkpoint(ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_step = -1
    best_path = None
    pat = re.compile(r"step_(\d+)\.pt$")

    for name in os.listdir(ckpt_dir):
        m = pat.search(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(ckpt_dir, name)

    if best_path is None:
        raise FileNotFoundError(f"No step_*.pt found in: {ckpt_dir}")
    return best_path


def _resolve_checkpoint_arg(ckpt_arg: str) -> str:
    if ckpt_arg.lower() == "latest":
        latest_txt = os.path.join("runs", "LATEST.txt")
        if not os.path.exists(latest_txt):
            raise FileNotFoundError("runs/LATEST.txt not found. Run training once first.")
        with open(latest_txt, "r", encoding="utf-8") as f:
            run_dir = f.read().strip()
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        return _pick_latest_checkpoint(ckpt_dir)

    if os.path.isdir(ckpt_arg):
        if os.path.basename(ckpt_arg) == "checkpoints":
            return _pick_latest_checkpoint(ckpt_arg)
        return _pick_latest_checkpoint(os.path.join(ckpt_arg, "checkpoints"))

    return ckpt_arg


def _parse_env_kwargs(s: str) -> dict[str, Any]:
    try:
        env_kwargs = json.loads(s)
        if not isinstance(env_kwargs, dict):
            raise ValueError
        return env_kwargs
    except Exception:
        raise ValueError('--env-kwargs doit être un JSON dict, ex: \'{"continuous": false}\'')


def _tiles_visited(env) -> float:
    uw = env.unwrapped

    if hasattr(uw, "tile_visited_count"):
        try:
            return float(getattr(uw, "tile_visited_count"))
        except Exception:
            pass

    # fallback: road tiles
    if hasattr(uw, "road"):
        try:
            road = getattr(uw, "road")
            if isinstance(road, (list, tuple)) and len(road) > 0:
                cnt = 0
                for t in road:
                    if hasattr(t, "visited") and bool(getattr(t, "visited")):
                        cnt += 1
                return float(cnt)
        except Exception:
            pass

    return float("nan")


def _running_mean(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=np.float32)
    s = 0.0
    for i, v in enumerate(x):
        s += float(v)
        out[i] = s / float(i + 1)
    return out


def _save_tiles_plot(tiles: list[float], out_dir: str) -> Optional[str]:
    tiles_arr = np.asarray([t for t in tiles if np.isfinite(t)], dtype=np.float32)
    if tiles_arr.size == 0:
        return None


    ep = np.arange(1, len(tiles_arr) + 1, dtype=np.int32)
    mean = float(np.mean(tiles_arr))
    run_mean = _running_mean(tiles_arr)

    plt.figure()
    plt.title("Tiles visited per episode (CarRacing)")
    plt.xlabel("Episode")
    plt.ylabel("Tiles visited")
    plt.plot(ep, tiles_arr, alpha=0.35, linewidth=1, label="Tiles (raw)")
    plt.plot(ep, run_mean, linewidth=2, label="Running mean")
    plt.axhline(mean, linewidth=2, linestyle="--", label=f"Mean = {mean:.1f}")
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tiles_visited.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, required=True)
    p.add_argument(
        "--env-kwargs",
        type=str,
        default="{}",
        help='JSON dict passed to gym.make (e.g. CarRacing: {"continuous": false})',
    )
    p.add_argument("--checkpoint", type=str, required=True, help='File, run dir, checkpoints dir, or "latest"')
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)

    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-folder", type=str, default="videos")
    p.add_argument("--video-prefix", type=str, default="eval")

    p.add_argument("--video-length", type=int, default=0)

    p.add_argument("--video-once", action="store_true")

    p.add_argument("--video-every", type=int, default=1)

    p.add_argument("--save-metrics", action="store_true", help="Save tiles_visited plot + metrics json")

    args = p.parse_args()

    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    cfg = PixelPipelineConfig(size=args.obs_size, frame_stack=args.frame_stack)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint_arg(args.checkpoint)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    obs_shape = tuple(int(x) for x in ckpt["obs_shape"])
    n_actions = int(ckpt["n_actions"])

    qnet = CNNQNetwork(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    qnet.load_state_dict(ckpt["model"])
    qnet.eval()

    # VIDEO
    episode_trigger = None
    step_trigger = None
    video_length = 0

    if args.record_video:
        video_length = int(args.video_length)
        if args.video_once:
            step_trigger = lambda step: step == 0
        else:
            n = max(1, int(args.video_every))
            episode_trigger = (lambda ep, n=n: (ep % n == 0))

    env = make_pixels_only_env(
        env_id=args.env_id,
        seed=args.seed,
        cfg=cfg,
        env_kwargs=env_kwargs,
        record_episode_stats=True,
        record_video=args.record_video,
        video_folder=args.video_folder,
        video_name_prefix=args.video_prefix,
        episode_trigger=episode_trigger,
        step_trigger=step_trigger,
        video_length=video_length,
    )

    returns: list[float] = []
    tiles: list[float] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            obs_u8 = obs_to_numpy_u8(obs)
            x = torch.from_numpy(obs_u8).to(device).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                q = qnet(x)
            action = int(torch.argmax(q, dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

        if "episode" in info:
            returns.append(float(info["episode"]["r"]))
        else:
            returns.append(float("nan"))

        tiles.append(_tiles_visited(env))

    env.close()

    returns_f = np.asarray([r for r in returns if np.isfinite(r)], dtype=np.float32)
    tiles_f = np.asarray([t for t in tiles if np.isfinite(t)], dtype=np.float32)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Episodes: {args.episodes}")
    if returns_f.size > 0:
        print(f"Mean return: {float(np.mean(returns_f)):.2f}")
        print(f"Std  return: {float(np.std(returns_f)):.2f}")
    else:
        print("Mean return: NaN")
        print("Std  return: NaN")

    if tiles_f.size > 0:
        print(f"Mean tiles visited: {float(np.mean(tiles_f)):.2f}")
        print(f"Std  tiles visited: {float(np.std(tiles_f)):.2f}")
    else:
        print("Mean tiles visited: NaN (metric not available for this env)")
        print("Std  tiles visited: NaN")

    # SAVE METRICS + PLOT
    if args.save_metrics:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = os.path.join(args.video_folder, f"{args.video_prefix}_metrics_{stamp}")
        os.makedirs(out_dir, exist_ok=True)

        plot_path = _save_tiles_plot(tiles, out_dir)

        metrics = {
            "checkpoint": ckpt_path,
            "env_id": args.env_id,
            "episodes": int(args.episodes),
            "mean_return": float(np.mean(returns_f)) if returns_f.size > 0 else None,
            "std_return": float(np.std(returns_f)) if returns_f.size > 0 else None,
            "mean_tiles_visited": float(np.mean(tiles_f)) if tiles_f.size > 0 else None,
            "std_tiles_visited": float(np.std(tiles_f)) if tiles_f.size > 0 else None,
            "tiles_per_episode": [float(x) if np.isfinite(x) else None for x in tiles],
            "returns_per_episode": [float(x) if np.isfinite(x) else None for x in returns],
            "plot_tiles_visited": plot_path,
        }

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if plot_path is not None:
            print(f"[OK] Tiles plot saved: {plot_path}")
        print(f"[OK] Metrics saved: {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
