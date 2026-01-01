from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from rl2.utils import set_global_seeds, make_run_dir, SafeSummaryWriter
from rl2.replay_buffer import ReplayBuffer
from rl2.dqn import CNNQNetwork, soft_update_
from rl2.wrappers import make_pixels_only_env, PixelPipelineConfig, obs_to_numpy_u8


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if t >= duration:
        return end
    return start + (end - start) * (t / duration)


@torch.no_grad()
def evaluate_pixels(env_id: str, qnet: CNNQNetwork, device: torch.device, episodes: int, seed: int,
                   cfg: PixelPipelineConfig, env_kwargs: dict) -> float:
    env = make_pixels_only_env(
        env_id=env_id,
        seed=seed,
        cfg=cfg,
        env_kwargs=env_kwargs,
        record_episode_stats=True,
        record_video=False,
    )

    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_u8 = obs_to_numpy_u8(obs)
            x = torch.from_numpy(obs_u8).to(device).unsqueeze(0).float() / 255.0
            q = qnet(x)
            action = int(torch.argmax(q, dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

        returns.append(float(info["episode"]["r"]) if "episode" in info else np.nan)

    env.close()
    returns = [r for r in returns if np.isfinite(r)]
    return float(np.mean(returns)) if returns else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--env-kwargs", type=str, default="{}", help="JSON dict passed to gym.make (e.g. CarRacing: {\"continuous\": false})")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--train-freq", type=int, default=4)

    p.add_argument("--target-update-freq", type=int, default=1_000)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)

    # IMPORTANT: pour éviter de "perdre du temps", tu peux mettre eval_every = total_steps
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=5)

    # NEW: checkpoints indépendants de l'évaluation
    p.add_argument("--save-every", type=int, default=50_000)

    p.add_argument("--run-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-tensorboard", action="store_true")

    p.add_argument("--double-dqn", action="store_true", default=True)
    p.add_argument("--no-double-dqn", action="store_false", dest="double_dqn")

    args = p.parse_args()

    try:
        env_kwargs = json.loads(args.env_kwargs)
        if not isinstance(env_kwargs, dict):
            raise ValueError
    except Exception:
        raise ValueError("--env-kwargs doit être un JSON dict, ex: '{\"continuous\": false}'")

    cfg = PixelPipelineConfig(size=args.obs_size, frame_stack=args.frame_stack)

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"dqn_pixels_{args.env_id}_seed{args.seed}"
    run_dir = make_run_dir(args.run_dir, run_name)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "env_kwargs_parsed": env_kwargs}, f, indent=2)

    writer = SafeSummaryWriter(run_dir, enabled=not args.no_tensorboard)

    env = make_pixels_only_env(
        env_id=args.env_id,
        seed=args.seed,
        cfg=cfg,
        env_kwargs=env_kwargs,
        record_episode_stats=True,
        record_video=False,
    )

    obs, info = env.reset(seed=args.seed)
    obs_u8 = obs_to_numpy_u8(obs)
    obs_shape = tuple(int(x) for x in obs_u8.shape)
    n_actions = int(env.action_space.n)

    qnet = CNNQNetwork(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    target = CNNQNetwork(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    optim = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, obs_shape=obs_shape)

    progress = trange(args.total_steps, desc="training", dynamic_ncols=True)

    for global_step in progress:
        eps = linear_schedule(args.eps_start, args.eps_end, args.eps_decay_steps, global_step)

        obs_u8 = obs_to_numpy_u8(obs)

        if rng.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.from_numpy(obs_u8).to(device).unsqueeze(0).float() / 255.0
                q = qnet(x)
                action = int(torch.argmax(q, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)

        done_for_reset = bool(terminated or truncated)
        terminated_flag = bool(terminated)  # IMPORTANT: truncated bootstrappe

        next_obs_u8 = obs_to_numpy_u8(next_obs)

        rb.add(
            obs_u8,
            action,
            float(reward),
            next_obs_u8,
            terminated_flag,
        )

        obs = next_obs

        if done_for_reset and "episode" in info:
            writer.add_scalar("train/episode_return", float(info["episode"]["r"]), global_step)
            writer.add_scalar("train/episode_len", float(info["episode"]["l"]), global_step)
            writer.add_scalar("train/epsilon", eps, global_step)
            obs, info = env.reset()

        # Learn
        if global_step >= args.learning_starts and (global_step % args.train_freq == 0) and len(rb) >= args.batch_size:
            batch = rb.sample(args.batch_size, rng)

            b_obs = torch.from_numpy(batch["obs"]).to(device).float() / 255.0
            b_next = torch.from_numpy(batch["next_obs"]).to(device).float() / 255.0
            b_actions = torch.from_numpy(batch["actions"]).to(device)
            b_rewards = torch.from_numpy(batch["rewards"]).to(device)
            b_terminated = torch.from_numpy(batch["terminated"]).to(device)

            q_sa = qnet(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if args.double_dqn:
                    next_actions = qnet(b_next).argmax(dim=1)
                    next_q = target(b_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q = target(b_next).max(dim=1).values

                target_q = b_rewards + args.gamma * next_q * (~b_terminated)

            loss = F.smooth_l1_loss(q_sa, target_q)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            optim.step()

            writer.add_scalar("train/loss", float(loss.item()), global_step)

        # Target update
        if global_step % args.target_update_freq == 0 and global_step > 0:
            if args.tau >= 1.0:
                target.load_state_dict(qnet.state_dict())
            else:
                soft_update_(target, qnet, tau=args.tau)

        # Save checkpoint (indépendant de l'éval)
        if (global_step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(run_dir, "checkpoints", f"step_{global_step+1}.pt")
            torch.save(
                {
                    "model": qnet.state_dict(),
                    "env_id": str(args.env_id),
                    "obs_shape": tuple(int(x) for x in obs_shape),
                    "n_actions": int(n_actions),
                    "global_step": int(global_step + 1),
                    "seed": int(args.seed),
                },
                ckpt_path,
            )

        # Eval (rare) + log
        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
            eval_return = evaluate_pixels(
                args.env_id, qnet, device, args.eval_episodes, seed=args.seed + 12345, cfg=cfg, env_kwargs=env_kwargs
            )
            writer.add_scalar("eval/mean_return", eval_return, global_step + 1)
            progress.set_postfix({"eps": f"{eps:.3f}", "eval": f"{eval_return:.1f}"})

    # Save final (au cas où total_steps n'est pas multiple de save_every)
    final_path = os.path.join(run_dir, "checkpoints", f"step_{args.total_steps}.pt")
    torch.save(
        {
            "model": qnet.state_dict(),
            "env_id": str(args.env_id),
            "obs_shape": tuple(int(x) for x in obs_shape),
            "n_actions": int(n_actions),
            "global_step": int(args.total_steps),
            "seed": int(args.seed),
        },
        final_path,
    )

    env.close()
    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
