import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        """直接堆叠为 numpy 数组，避免 list of ndarrays -> tensor 的慢警告。"""
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        obs_np = np.stack(obs)
        next_obs_np = np.stack(next_obs)
        act_np = np.array(act, dtype=np.int64)
        rew_np = np.array(rew, dtype=np.float32)
        done_np = np.array(done, dtype=np.float32)
        return Transition(obs_np, act_np, rew_np, next_obs_np, done_np)
