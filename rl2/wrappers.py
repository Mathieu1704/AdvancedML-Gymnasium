from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Any
from datetime import datetime
import os

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
    """Heuristique: une observation pixels est typiquement Box uint8 en 2D ou 3D."""
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
) -> gym.Env:
    """
    Construit un env "pixels-only" en utilisant Gymnasium.

    - Si l'environnement renvoie déjà une image (Box uint8), on l'utilise directement (pas de render()).
      Exemple: CarRacing-v3. :contentReference[oaicite:3]{index=3}
    - Sinon, on fabrique les pixels via AddRenderObservation (nécessite render_mode="rgb_array").

    Vidéo:
    - RecordVideo supporte episode_trigger OU step_trigger (pas les deux). :contentReference[oaicite:4]{index=4}
    - video_length > 0 permet des clips de longueur fixe (en steps).
    """
    if env_kwargs is None:
        env_kwargs = {}

    if episode_trigger is not None and step_trigger is not None:
        raise ValueError("RecordVideo: fournir soit episode_trigger soit step_trigger, pas les deux.")

    # Pour la vidéo, render_mode doit être rgb_array
    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

    # Si l'observation n'est PAS une image, on doit passer par render()
    if not _is_image_space(env.observation_space):
        if env.render_mode is None:
            env.close()
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

        env = AddRenderObservation(env, render_only=True)

    # Vidéo (après s'être assuré que render_mode est OK)
    if record_video:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        actual_folder = os.path.join(video_folder, f"{video_name_prefix}_{stamp}")
        os.makedirs(actual_folder, exist_ok=True)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = lambda ep: ep == 0  # défaut: 1er épisode

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

    # Grayscale uniquement si on est en (H,W,3) ou (H,W,1)
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box) and obs_space.shape is not None and len(obs_space.shape) == 3:
        env = GrayscaleObservation(env, keep_dim=False)  # -> (H,W)

    # Resize
    shape = cfg.size if isinstance(cfg.size, tuple) else (cfg.size, cfg.size)
    env = ResizeObservation(env, shape)

    # Frame stack: forme typique (stack, H, W). :contentReference[oaicite:5]{index=5}
    env = _FrameStack(env, cfg.frame_stack)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_numpy_u8(obs) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr
