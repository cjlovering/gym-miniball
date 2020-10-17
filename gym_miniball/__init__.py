from .ball import (
    BallEnv1,
    BallEnv2,
    BallEnv3,
    BallEnv4,
    BallEnv5,
    BallEnv6,
    BallEnv7,
    BallEnv8,
)

from gym.envs.registration import register

max_episode_steps = 10_000
register(
    id="MiniBall1-v0",
    entry_point="gym_miniball:BallEnv1",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall2-v0",
    entry_point="gym_miniball:BallEnv2",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall3-v0",
    entry_point="gym_miniball:BallEnv3",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall4-v0",
    entry_point="gym_miniball:BallEnv4",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall5-v0",
    entry_point="gym_miniball:BallEnv5",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall6-v0",
    entry_point="gym_miniball:BallEnv6",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall7-v0",
    entry_point="gym_miniball:BallEnv7",
    max_episode_steps=max_episode_steps,
)
register(
    id="MiniBall8-v0",
    entry_point="gym_miniball:BallEnv8",
    max_episode_steps=max_episode_steps,
)
