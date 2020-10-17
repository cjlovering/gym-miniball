from .ball import BallEnv, BallEnv2

from gym.envs.registration import register

register(id="MiniBall-v1", entry_point="gym_miniball:BallEnv", max_episode_steps=1000)
register(id="MiniBall-v2", entry_point="gym_miniball:BallEnv2", max_episode_steps=1000)
