from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """
    Replay buffer circulaire pour DQN: stocke (obs, action, reward, next_obs, terminated).

    IMPORTANT:
    - On stocke 'terminated' (vrai terminal MDP), PAS (terminated OR truncated).
    - Les épisodes 'truncated' (TimeLimit) doivent généralement bootstrapper.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]):
        self.capacity = int(capacity)
        self.obs_shape = obs_shape

        self._size = 0
        self._pos = 0

        self.obs = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        # terminated-only mask (True = terminal MDP => no bootstrap)
        self.terminated = np.empty((self.capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.obs[self._pos] = obs
        self.next_obs[self._pos] = next_obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.terminated[self._pos] = terminated

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        assert self._size >= batch_size
        idx = rng.integers(0, self._size, size=batch_size, endpoint=False)

        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "terminated": self.terminated[idx],
        }
