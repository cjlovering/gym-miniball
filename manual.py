#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import gym_miniball
from gym_minigrid.window import Window


def redraw(img):
    img = env.render("rgb_array")
    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)
    # print("step=%s, reward=%.2f" % (env.step_count, reward))

    if done:
        print("done!")
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left" or event.key == "1":
        step(1)
        return
    if event.key == "right" or event.key == "2":
        step(2)
        return
    else:
        step(0)


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default="MiniBallStar-v0")
parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=-1
)
args = parser.parse_args()

env = gym.make(args.env)

window = Window("gym_miniball - " + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=False)
while 1:
    time.sleep(0.0000001)
    step(0)
