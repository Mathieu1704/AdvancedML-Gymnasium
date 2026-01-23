from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Any
from datetime import datetime
import os
import math

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AddRenderObservation,
    GrayscaleObservation,
    ResizeObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)

try:
    from gymnasium.wrappers import FrameStackObservation as _FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStack as _FrameStack


@dataclass
class PixelPipelineConfig:
    size: Union[int, tuple[int, int]] = 84
    frame_stack: int = 4


def _is_image_space(space: gym.Space) -> bool:
    if not isinstance(space, gym.spaces.Box):
        return False
    if space.dtype != np.uint8:
        return False
    if space.shape is None:
        return False
    if len(space.shape) == 2:
        return True  # (H,W)
    if len(space.shape) == 3 and space.shape[-1] in (1, 3):
        return True  # (H,W,C)
    return False


def _safe_xy(x: Any) -> Optional[tuple[float, float]]:
    try:
        if hasattr(x, "x") and hasattr(x, "y"):
            return float(x.x), float(x.y)
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            return float(x[0]), float(x[1])
    except Exception:
        return None
    return None


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        terminated = False
        truncated = False
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            if done:
                break
        
        return obs, total_reward, terminated, truncated, info


class FireAndLifeLoss(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._lives = 0
        self._fire_action = 1 
        
        if hasattr(env.unwrapped, 'get_action_meanings'):
            meanings = env.unwrapped.get_action_meanings()
            if 'FIRE' in meanings:
                self._fire_action = meanings.index('FIRE')

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if "lives" in info:
            lives = info["lives"]
            if 0 < lives < self._lives:
                obs_f, r_f, term_f, trunc_f, info_f = self.env.step(self._fire_action)
                reward += r_f
                terminated = terminated or term_f
                truncated = truncated or trunc_f
                info = info_f
            self._lives = lives
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._lives = 0
        obs, info = self.env.reset(**kwargs)
        if "lives" in info:
            self._lives = info["lives"]
            
        self.env.step(self._fire_action)
        return obs, info


class WrongWayPenalty(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        penalty: float = 1.0,
        angle_threshold_deg: float = 150.0,
        speed_min: float = 8.0,
        backward_idx_tol: int = 5,
        confirm_steps: int = 8,
    ):
        super().__init__(env)
        self.penalty = float(penalty)
        self.angle_threshold_deg = float(angle_threshold_deg)
        self.speed_min = float(speed_min)
        self.backward_idx_tol = int(backward_idx_tol)
        self.confirm_steps = int(confirm_steps)

        self._track_xy: Optional[np.ndarray] = None
        self._track_alpha: Optional[np.ndarray] = None
        self._n: int = 0
        self._last_idx: Optional[int] = None
        self._wrong_count: int = 0
        self._cos_thr = math.cos(math.radians(self.angle_threshold_deg))

    def _parse_track(self) -> bool:
        uw = self.env.unwrapped
        if not hasattr(uw, "track"):
            return False
        track = getattr(uw, "track", None)
        if not isinstance(track, (list, tuple)) or len(track) < 10:
            return False

        xs, ys, alphas = [], [], []
        for item in track:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                try:
                    alpha = float(item[0])
                    x = float(item[2])
                    y = float(item[3])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)
                alphas.append(alpha)

        if len(xs) < 10:
            return False

        self._track_xy = np.stack(
            [np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)],
            axis=1,
        )
        self._track_alpha = np.asarray(alphas, dtype=np.float32)
        self._n = int(self._track_xy.shape[0])
        self._last_idx = None
        self._wrong_count = 0
        return True

    @staticmethod
    def _signed_cyclic_diff(new: int, old: int, n: int) -> float:
        d = new - old
        d = (d + n / 2.0) % n - n / 2.0
        return d

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._parse_track()
        return obs, info

    def _car_velocity_dir(self) -> Optional[tuple[float, float, float]]:
        uw = self.env.unwrapped
        if not hasattr(uw, "car"):
            return None
        try:
            v = uw.car.hull.linearVelocity
            vv = _safe_xy(v)
            if vv is None:
                return None
            vx, vy = float(vv[0]), float(vv[1])
            speed = float(math.hypot(vx, vy))
            if speed < 1e-6:
                return 0.0, 0.0, 0.0
            return vx / speed, vy / speed, speed
        except Exception:
            return None

    def _car_position(self) -> Optional[tuple[float, float]]:
        uw = self.env.unwrapped
        if not hasattr(uw, "car"):
            return None
        try:
            return _safe_xy(uw.car.hull.position)
        except Exception:
            return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.penalty > 0.0 and self._track_xy is not None and self._track_alpha is not None:
            vdir = self._car_velocity_dir()
            pos = self._car_position()

            if vdir is not None and pos is not None:
                vx_u, vy_u, speed = vdir
                if speed >= self.speed_min:
                    cx, cy = pos
                    dxy = self._track_xy - np.asarray([cx, cy], dtype=np.float32)
                    dist2 = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
                    idx = int(np.argmin(dist2))

                    alpha = float(self._track_alpha[idx])
                    tx, ty = math.cos(alpha), math.sin(alpha)
                    align = vx_u * tx + vy_u * ty

                    backward = False
                    if self._last_idx is not None and self._n > 0:
                        dd = self._signed_cyclic_diff(idx, self._last_idx, self._n)
                        backward = dd < -float(self.backward_idx_tol)

                    wrong_candidate = (align < self._cos_thr) or backward
                    if wrong_candidate:
                        self._wrong_count += 1
                    else:
                        self._wrong_count = 0

                    wrong_way = self._wrong_count >= self.confirm_steps
                    if wrong_way:
                        reward = float(reward) - self.penalty
                        info = dict(info)
                        info["wrong_way"] = True
                        info["wrong_way_penalty"] = float(self.penalty)

                    self._last_idx = idx

        return obs, reward, terminated, truncated, info


class HighSpeedSteerPenalty(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        penalty_scale: float = 0.0,
        speed_threshold: float = 18.0,
    ):
        super().__init__(env)
        self.penalty_scale = float(penalty_scale)
        self.speed_threshold = float(speed_threshold)

    def _speed(self) -> Optional[float]:
        uw = self.env.unwrapped
        if not hasattr(uw, "car"):
            return None
        try:
            v = uw.car.hull.linearVelocity
            vv = _safe_xy(v)
            if vv is None:
                return None
            return float(math.hypot(vv[0], vv[1]))
        except Exception:
            return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # CarRacing discret: 1=right, 2=left
        if self.penalty_scale > 0.0 and int(action) in (1, 2):
            sp = self._speed()
            if sp is not None and sp > self.speed_threshold:
                reward = float(reward) - self.penalty_scale * float(sp - self.speed_threshold)
                info = dict(info)
                info["highspeed_steer"] = True
                info["highspeed_steer_penalty"] = float(self.penalty_scale * (sp - self.speed_threshold))
        return obs, reward, terminated, truncated, info


class SteerOscillationPenalty(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = 0.0):
        super().__init__(env)
        self.penalty = float(penalty)
        self._prev_action: Optional[int] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.penalty > 0.0:
            a = int(action)
            pa = self._prev_action
            if pa is not None and ((pa == 1 and a == 2) or (pa == 2 and a == 1)):
                reward = float(reward) - self.penalty
                info = dict(info)
                info["steer_oscillation"] = True
        self._prev_action = a
        return obs, reward, terminated, truncated, info


class CenterlineBonus(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        bonus_scale: float = 0.0,
        sigma: float = 6.0,
        speed_min: float = 10.0,
        min_align: float = 0.3,
    ):
        super().__init__(env)
        self.bonus_scale = float(bonus_scale)
        self.sigma = float(sigma)
        self.speed_min = float(speed_min)
        self.min_align = float(min_align)
        self._track_xy: Optional[np.ndarray] = None
        self._track_alpha: Optional[np.ndarray] = None

    def _parse_track(self) -> bool:
        uw = self.env.unwrapped
        if not hasattr(uw, "track"):
            return False
        track = getattr(uw, "track", None)
        if not isinstance(track, (list, tuple)) or len(track) < 10:
            return False
        xs, ys, alphas = [], [], []
        for item in track:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                try:
                    alpha = float(item[0])
                    x = float(item[2])
                    y = float(item[3])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)
                alphas.append(alpha)
        if len(xs) < 10:
            return False
        self._track_xy = np.stack(
            [np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)],
            axis=1,
        )
        self._track_alpha = np.asarray(alphas, dtype=np.float32)
        return True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._parse_track()
        return obs, info

    def _car_position(self) -> Optional[tuple[float, float]]:
        uw = self.env.unwrapped
        if not hasattr(uw, "car"):
            return None
        try:
            return _safe_xy(uw.car.hull.position)
        except Exception:
            return None

    def _car_velocity_dir(self) -> Optional[tuple[float, float, float]]:
        uw = self.env.unwrapped
        if not hasattr(uw, "car"):
            return None
        try:
            v = uw.car.hull.linearVelocity
            vv = _safe_xy(v)
            if vv is None:
                return None
            vx, vy = float(vv[0]), float(vv[1])
            sp = float(math.hypot(vx, vy))
            if sp < 1e-6:
                return 0.0, 0.0, 0.0
            return vx / sp, vy / sp, sp
        except Exception:
            return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.bonus_scale > 0.0 and self._track_xy is not None and self._track_alpha is not None:
            pos = self._car_position()
            vdir = self._car_velocity_dir()
            if pos is not None and vdir is not None:
                vx_u, vy_u, sp = vdir
                if sp >= self.speed_min:
                    cx, cy = pos
                    dxy = self._track_xy - np.asarray([cx, cy], dtype=np.float32)
                    dist2 = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
                    idx = int(np.argmin(dist2))

                    alpha = float(self._track_alpha[idx])
                    tx, ty = math.cos(alpha), math.sin(alpha)
                    align = vx_u * tx + vy_u * ty

                    if align >= self.min_align:
                        nx, ny = -ty, tx
                        lateral = float(abs((cx - float(self._track_xy[idx, 0])) * nx + (cy - float(self._track_xy[idx, 1])) * ny))
                        bonus = self.bonus_scale * math.exp(-(lateral * lateral) / (2.0 * self.sigma * self.sigma))
                        reward = float(reward) + float(bonus)
                        info = dict(info)
                        info["center_bonus"] = float(bonus)
        return obs, reward, terminated, truncated, info


def make_pixels_only_env(
    env_id: str,
    seed: int,
    cfg: PixelPipelineConfig = PixelPipelineConfig(),
    env_kwargs: Optional[dict[str, Any]] = None,
    record_episode_stats: bool = True,
    record_video: bool = False,
    video_folder: str = "videos",
    video_name_prefix: str = "eval",
    episode_trigger: Optional[Callable[[int], bool]] = None,
    step_trigger: Optional[Callable[[int], bool]] = None,
    video_length: int = 0,
    skip_frames: int = 4,
    wrong_way_penalty: float = 0.0,
    wrong_way_angle_deg: float = 150.0,
    wrong_way_speed_min: float = 8.0,
    wrong_way_confirm_steps: int = 8,
    wrong_way_backward_idx_tol: int = 5,
    highspeed_steer_penalty_scale: float = 0.0,
    highspeed_steer_speed_threshold: float = 18.0,
    steer_oscillation_penalty: float = 0.0,
    center_bonus_scale: float = 0.0,
    center_bonus_sigma: float = 6.0,
    center_speed_min: float = 10.0,
    center_min_align: float = 0.3,
) -> gym.Env:
    if env_kwargs is None:
        env_kwargs = {}

    if episode_trigger is not None and step_trigger is not None:
        raise ValueError("RecordVideo: fournir soit episode_trigger soit step_trigger, pas les deux.")

    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

    if not _is_image_space(env.observation_space):
        if env.render_mode is None:
            env.close()
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        env = AddRenderObservation(env, render_only=True)
    
    target_ids = ["Pong", "Breakout", "SpaceInvaders", "Alien"]
    if any(x in env_id for x in target_ids):
        env = FireAndLifeLoss(env)

    if wrong_way_penalty and float(wrong_way_penalty) > 0.0:
        env = WrongWayPenalty(
            env,
            penalty=float(wrong_way_penalty),
            angle_threshold_deg=float(wrong_way_angle_deg),
            speed_min=float(wrong_way_speed_min),
            backward_idx_tol=int(wrong_way_backward_idx_tol),
            confirm_steps=int(wrong_way_confirm_steps),
        )

    if highspeed_steer_penalty_scale and float(highspeed_steer_penalty_scale) > 0.0:
        env = HighSpeedSteerPenalty(
            env,
            penalty_scale=float(highspeed_steer_penalty_scale),
            speed_threshold=float(highspeed_steer_speed_threshold),
        )

    if steer_oscillation_penalty and float(steer_oscillation_penalty) > 0.0:
        env = SteerOscillationPenalty(env, penalty=float(steer_oscillation_penalty))

    if center_bonus_scale and float(center_bonus_scale) > 0.0:
        env = CenterlineBonus(
            env,
            bonus_scale=float(center_bonus_scale),
            sigma=float(center_bonus_sigma),
            speed_min=float(center_speed_min),
            min_align=float(center_min_align),
        )

    if record_video:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        actual_folder = os.path.join(video_folder, f"{video_name_prefix}_{stamp}")
        os.makedirs(actual_folder, exist_ok=True)
        if episode_trigger is None and step_trigger is None:
            episode_trigger = lambda ep: ep == 0
        env = RecordVideo(
            env,
            video_folder=actual_folder,
            name_prefix=video_name_prefix,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=int(video_length),
        )

    if record_episode_stats:
        env = RecordEpisodeStatistics(env)

    if skip_frames > 0:
        env = SkipFrame(env, skip=skip_frames)

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and obs_space.shape is not None and len(obs_space.shape) == 3:
        env = GrayscaleObservation(env, keep_dim=False)

    shape = cfg.size if isinstance(cfg.size, tuple) else (cfg.size, cfg.size)
    env = ResizeObservation(env, shape)

    env = _FrameStack(env, cfg.frame_stack)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_numpy_u8(obs) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr