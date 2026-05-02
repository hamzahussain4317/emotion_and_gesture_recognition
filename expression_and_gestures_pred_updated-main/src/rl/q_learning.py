from __future__ import annotations

import json
import time

import numpy as np

from src.rl.env import GridConfig, HRIGridWorld, Human, MOODS


def train(
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 42,
    randomize_moods: bool = True,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    env = HRIGridWorld(GridConfig())
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)

    eps_decay = (eps_end / eps_start) ** (1.0 / episodes)
    eps = eps_start

    rewards = np.zeros(episodes, dtype=np.float32)
    steps_per_ep = np.zeros(episodes, dtype=np.int32)

    for ep in range(episodes):
        if randomize_moods:
            humans = [
                Human(pos=env.config.humans[0].pos, mood=rng.choice(MOODS)),
                Human(pos=env.config.humans[1].pos, mood=rng.choice(MOODS)),
            ]
        else:
            humans = None
        s = env.reset(humans=humans)
        total = 0.0
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(np.argmax(Q[s]))
            s_next, r, done, _ = env.step(a)
            total += r
            td_target = r + (0.0 if done else gamma * Q[s_next].max())
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
            if done:
                break
        rewards[ep] = total
        steps_per_ep[ep] = env.steps
        eps *= eps_decay

    stats = {
        "episodes": episodes,
        "final_avg_reward": float(rewards[-100:].mean()),
        "first_avg_reward": float(rewards[:100].mean()),
        "max_reward": float(rewards.max()),
        "alpha": alpha,
        "gamma": gamma,
    }
    return Q, {"rewards": rewards.tolist(), "steps": steps_per_ep.tolist(), **stats}


def rollout(Q: np.ndarray, humans: list[Human], max_steps: int = 50) -> dict:
    env = HRIGridWorld(GridConfig())
    s = env.reset(humans=humans)
    path = [env.robot]
    total = 0.0
    while True:
        a = int(np.argmax(Q[s]))
        s, r, done, info = env.step(a)
        path.append(env.robot)
        total += r
        if done:
            break
    return {"path": path, "total_reward": total, "steps": env.steps}
