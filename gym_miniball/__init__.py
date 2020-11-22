from .ball import (
    BallEnv0,
    BallEnv1,
    BallEnv2,
    BallEnv3,
    BallEnv4,
    BallEnv5,
    BallEnv6,
    BallEnv7,
    BallEnv8,
    BallEnvStar,
)

from gym.envs.registration import register

reward_threshold = 5000
max_episode_steps = 5000

register(
    id="MiniBall0-v0",
    entry_point="gym_miniball:BallEnv0",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)

register(
    id="MiniBall1-v0",
    entry_point="gym_miniball:BallEnv1",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall2-v0",
    entry_point="gym_miniball:BallEnv2",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall3-v0",
    entry_point="gym_miniball:BallEnv3",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall4-v0",
    entry_point="gym_miniball:BallEnv4",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall5-v0",
    entry_point="gym_miniball:BallEnv5",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall6-v0",
    entry_point="gym_miniball:BallEnv6",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall7-v0",
    entry_point="gym_miniball:BallEnv7",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBall8-v0",
    entry_point="gym_miniball:BallEnv8",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)
register(
    id="MiniBallStar-v0",
    entry_point="gym_miniball:BallEnvStar",
    max_episode_steps=max_episode_steps,
    reward_threshold=reward_threshold,
)

