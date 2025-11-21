import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
