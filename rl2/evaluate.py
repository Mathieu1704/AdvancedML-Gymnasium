from __future__ import annotations

import argparse
import os
import re
import json
import numpy as np
import torch

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, required=True)
    p.add_argument("--env-kwargs", type=str, default="{}", help="JSON dict passed to gym.make (e.g. CarRacing: {\"continuous\": false})")
    p.add_argument("--checkpoint", type=str, required=True, help='File, run dir, checkpoints dir, or "latest"')
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)

    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-folder", type=str, default="videos")
    p.add_argument("--video-prefix", type=str, default="eval")
    p.add_argument("--video-length", type=int, default=1000)
    p.add_argument("--video-once", action="store_true")

    args = p.parse_args()

    try:
        env_kwargs = json.loads(args.env_kwargs)
        if not isinstance(env_kwargs, dict):
            raise ValueError
    except Exception:
        raise ValueError("--env-kwargs doit être un JSON dict, ex: '{\"continuous\": false}'")

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

    step_trigger = None
    video_length = 0
    if args.record_video:
        video_length = int(args.video_length)
        if args.video_once:
            step_trigger = lambda step: step == 0  # clips fixes comparables

    env = make_pixels_only_env(
        env_id=args.env_id,
        seed=args.seed,
        cfg=cfg,
        env_kwargs=env_kwargs,
        record_episode_stats=True,
        record_video=args.record_video,
        video_folder=args.video_folder,
        video_name_prefix=args.video_prefix,
        step_trigger=step_trigger,
        video_length=video_length,
    )

    returns = []
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

        returns.append(float(info["episode"]["r"]) if "episode" in info else np.nan)

    env.close()

    returns = [r for r in returns if np.isfinite(r)]
    print(f"Checkpoint: {ckpt_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Std  return: {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
