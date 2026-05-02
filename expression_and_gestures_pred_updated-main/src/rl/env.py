from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


MOODS = ("happy", "angry", "neutral")
MOOD_IDX = {m: i for i, m in enumerate(MOODS)}

ACTIONS = ("up", "down", "left", "right")
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass
class Human:
    pos: tuple[int, int]
    mood: str


@dataclass
class GridConfig:
    size: int = 6
    start: tuple[int, int] = (0, 0)
    humans: list[Human] = field(default_factory=lambda: [
        Human(pos=(4, 4), mood="happy"),
        Human(pos=(2, 5), mood="angry"),
    ])
    max_steps: int = 50
    step_penalty: float = -0.1
    happy_reward: float = 10.0
    angry_penalty: float = -10.0
    neutral_bonus: float = 1.0


class HRIGridWorld:
    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.reset()

    @property
    def n_states(self) -> int:
        c = self.config
        return c.size * c.size * (len(MOODS) ** len(c.humans))

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def encode_state(self, pos: tuple[int, int] | None = None) -> int:
        c = self.config
        pos = pos or self.robot
        r, col = pos
        s = r * c.size + col
        moods_radix = len(MOODS) ** len(c.humans)
        mood_code = 0
        for h in c.humans:
            mood_code = mood_code * len(MOODS) + MOOD_IDX[h.mood]
        return s * moods_radix + mood_code

    def reset(self, humans: list[Human] | None = None) -> int:
        if humans is not None:
            self.config.humans = humans
        self.robot = self.config.start
        self.steps = 0
        self.done = False
        return self.encode_state()

    def _neighbor_moods(self) -> list[str]:
        hits = []
        for h in self.config.humans:
            if abs(h.pos[0] - self.robot[0]) + abs(h.pos[1] - self.robot[1]) <= 1:
                hits.append(h.mood)
        return hits

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self.done:
            return self.encode_state(), 0.0, True, {}

        dr, dc = ACTION_DELTAS[action]
        r, c = self.robot
        nr = min(max(r + dr, 0), self.config.size - 1)
        nc = min(max(c + dc, 0), self.config.size - 1)
        self.robot = (nr, nc)
        self.steps += 1

        reward = self.config.step_penalty
        reached_happy = False
        for mood in self._neighbor_moods():
            if mood == "happy" and self.robot == next(h.pos for h in self.config.humans if h.mood == "happy"):
                reward += self.config.happy_reward
                reached_happy = True
            elif mood == "angry":
                reward += self.config.angry_penalty
            elif mood == "neutral":
                reward += self.config.neutral_bonus

        self.done = reached_happy or self.steps >= self.config.max_steps
        return self.encode_state(), reward, self.done, {"pos": self.robot}

    def render(self) -> np.ndarray:
        c = self.config
        grid = np.zeros((c.size, c.size), dtype=np.int8)
        for h in c.humans:
            grid[h.pos] = {"happy": 1, "angry": 2, "neutral": 3}[h.mood]
        grid[self.robot] = 5
        return grid
