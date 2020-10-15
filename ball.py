import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import glob
import itertools
import math

from collections import ChainMap
from collections import Counter
from collections import defaultdict
from scipy.stats import zscore

import scipy.stats as stats
import dataclasses
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.metrics import ndcg_score

from gym.envs.classic_control import rendering
from gym import core, spaces
from gym.utils import seeding

𝛕 = 2 * math.pi
𝛑 = math.pi

NUM_ACTIONS = 3
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
WIDTH = 64
HEIGHT = 64
CHANNELS = 3

𝛑 = math.pi
𝛕 = 2 * math.pi
SPEED = 0.01
INTERNAL = 100
VARIANCE = 8
BALL_SPEED = 1 * SPEED
PLAYER_SPEED = 0.5 * SPEED
VISIBLE = True

DEFAULT_CONFIG = {
    "grid": {
        "height": 64,
        "width": 64
    },
    "player": {
        "y": 5,
        "length": 8
    },
    "balls": {
        "number": 1,
        "quadrant": "3",
    },
    "platform": {
        "number": 1,
        "quadrant": "1",
    }
}



def reflect_boundary_function(n_x_loc, n_y_loc, item):
    if (n_x_loc - item.radius <= 0) or (n_x_loc + item.radius >= WIDTH):
        item.direction = (3 * 𝛑 - item.direction) % 𝛕
        return True
    elif (n_y_loc - item.radius <= 0) or (n_y_loc + item.radius >= HEIGHT):
        item.direction = (𝛕 - item.direction) % 𝛕
        return True
    return False

def redirect_on_platform(n_x, n_y, radius, platform):
    item_x_boundary = n_x - radius, n_x + radius
    item_y_boundary = n_y - radius, n_y + radius

    platform_x_boundary = (
        platform.x - platform.length / 2,
        platform.x + platform.length / 2,
    )
    platform_y_boundary = (
        platform.y - platform.width / 2,
        platform.y + platform.width / 2,
    )

    if intersects(item_x_boundary, platform_x_boundary) and intersects(
        item_y_boundary, platform_y_boundary
    ):
        return True
    return False


def additive_movement_function(t, item):
    n_x_loc = item.x + item.v * math.cos(item.direction)
    n_y_loc = item.y + item.v * math.sin(item.direction)
    return n_x_loc, n_y_loc


def movement_function(item):
    n_x_loc = item.x + item.v * math.cos(item.direction)
    n_y_loc = item.y + item.v * math.sin(item.direction)
    return n_x_loc, n_y_loc


def point_inside(n_x, n_y, radius, point):
    return (n_x - radius <= point[0] and point[0] <= n_x + radius) and (
        n_y - radius <= point[1] and point[1] <= n_y + radius
    )
def intersects(a, b):
    return not (a[0] > b[1] or a[1] < b[0])

@dataclasses.dataclass
class Ball:
    x: float
    y: float
    v: float
    direction: float
    boundary_function: dataclasses.field() = reflect_boundary_function
    movement_function: dataclasses.field() = movement_function
    color: tuple = (0.1, 0.1, 0.1)  # r, b, g
    radius: int = 0.5
    t: int = 0

    def draw(self, viewer):
        transform = rendering.Transform(translation=(self.x, self.y))
        circ = viewer.draw_circle(self.radius)
        circ.set_color(*self.color)
        circ.add_attr(transform)

    def step(self, platforms):
        n_x, n_y = self.movement_function(self)
        at_boundary = self.boundary_function(n_x, n_y, self)
        direction = self.direction

        for p in platforms:
            intersected = redirect_on_platform(n_x, n_y, self.radius, p)
            if intersected:
                finished = False
                l, r, t, b = (
                    p.x + -p.length / 2,
                    p.x + p.length / 2,
                    p.y + p.width / 2,
                    p.y + -p.width / 2,
                )
                corners = [(l, b), (l, t), (r, t), (r, b)]
                for point in corners:
                    if point_inside(n_x, n_y, self.radius, point):
                        ball_dx = math.cos(self.direction)
                        ball_dy = math.sin(self.direction)
                        x = self.x - point[0]
                        y = self.y - point[1]
                        c = -2 * (ball_dx * x + ball_dy * y) / (x * x + y * y)
                        ball_dx = ball_dx + c * x
                        ball_dy = ball_dy + c * y
                        self.direction = math.atan2(ball_dx, ball_dy)
                        at_boundary = True
                        finished = True
                        break
                if not finished:
                    if (p.x - p.length / 2 <= self.x) and (
                        self.x <= p.x + p.length / 2
                    ):
                        self.direction = (𝛕 - self.direction) % 𝛕
                    else:
                        self.direction = (3 * 𝛑 - self.direction) % 𝛕

                # if (p.y - p.width / 2 < n_y) or (n_y > p.y + p.width / 2):
                # else:
                #     self.direction = (3 * 𝛑 - self.direction) % 𝛕
                at_boundary = True
                break

        # n_x, n_y = self.movement_function(self.t, self)
        # at_boundary = self.boundary_function(n_x, n_y, self)
        if not at_boundary:
            self.x = n_x
            self.y = n_y
        # else:
        #     n_x, n_y = self.movement_function(self)
        #     stuck = redirect_on_platform(n_x, n_y, self.radius, platforms[0])
        #     # print(stuck, n_x, n_y)
        #     # # if stuck:
        #     # #     print("STUCK!")
        #     if stuck:
        #         self.x = n_x
        #         self.y = n_y
        #     # pass
        #     pass

        # self.t += 1


@dataclasses.dataclass
class Platform:
    x: float
    y: float
    v: float
    direction: float
    boundary_function: dataclasses.field() = reflect_boundary_function
    movement_function: dataclasses.field() = additive_movement_function
    color: tuple = (0.8, 0.8, 0.8)  # r, b, g
    length: int = 16
    width: int = 4
    t: int = 0

    def draw(self, viewer):
        transform = rendering.Transform(translation=(self.x, self.y))
        l, r, t, b = -self.length / 2, self.length / 2, self.width / 2, -self.width / 2
        poly = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        poly.set_color(*self.color)
        poly.add_attr(transform)

    def step(self):
        pass
        # before_direction = self.direction
        # n_x, n_y = self.movement_function(self.t, self)
        # at_boundary = self.boundary_function(n_x, n_y, self)
        # if at_boundary:
        #     nn_x, nn_y = n_x, n_y
        #     n_x, n_y = self.movement_function(self.t, self)

        # self.x = n_x
        # self.y = n_y
        # self.t += 1


@dataclasses.dataclass
class PlayerPlatform:
    x: float
    y: float
    v: float
    direction: float
    boundary_function: dataclasses.field() = reflect_boundary_function
    movement_function: dataclasses.field() = additive_movement_function
    color: tuple = (0.8, 0.2, 0.2)  # r, b, g
    length: int = 16
    width: int = 4
    t: int = 0

    def draw(self, viewer):
        transform = rendering.Transform(translation=(self.x, self.y))
        l, r, t, b = -self.length / 2, self.length / 2, self.width / 2, -self.width / 2
        poly = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        poly.set_color(*self.color)
        poly.add_attr(transform)

    def step(self):
        n_x = self.x + self.v * math.cos(self.direction)
        n_y = self.y + self.v * math.sin(self.direction)
        if (n_x - (self.length / 2) <= 0) or (n_x + (self.length / 2) >= WIDTH):
            self.direction = (3 * 𝛑 - self.direction) % 𝛕
        else:
            self.x = n_x
            self.y = n_y


class BallEnv(core.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 2500}
    def __init__(self, items=None, platforms=None, config=None, internal_steps=INTERNAL, visible=VISIBLE):
        self.viewer = None
        self.internal_steps = internal_steps
        self.observation_space = spaces.Box(
            low=0, high=256, shape=(WIDTH, HEIGHT, CHANNELS), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.state = None
        self.visible = visible
        self.seed()

        self.config = config

        if config is not None:
            self.items, self.platforms = generate_items(self.config)
        else:
            self.items = items
            self.platforms = platforms

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.viewer.close()

    def reset(self):
        if self.config is None:
            assert False
        self.items, self.platforms = generate_items(self.config)
        return self.step(0)

    def step(self, a):
        for _ in range(self.internal_steps):
            for i in self.platforms:
                i.step()
            for i in self.items:
                i.step(self.platforms)

        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_HEIGHT, SCREEN_WIDTH)
            self.viewer.set_bounds(0, WIDTH, 0, HEIGHT)
        self.viewer.window.set_visible(self.visible)

        for i in self.items:
            i.draw(self.viewer)
        for i in self.platforms:
            i.draw(self.viewer)

        o = self.viewer.render(return_rgb_array=True)
        return o

def get_random_location(partition, width, height, delta=2):
    if partition == "1":
        x_left, x_right = width / 2, width
        y_bot, y_top = height * 3 / 4, height
    elif partition == "2":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height * 3 / 4, height
    elif partition == "3":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height / 2, height * 3 / 4
    elif partition == "4":
        x_left, x_right = width / 2, width
        y_bot, y_top = height / 2, height * 3 / 4 
    
    x = np.clip(np.random.normal((x_left + x_right)/2, VARIANCE), x_left+delta, x_right-delta)
    y = np.clip(np.random.normal((y_bot + y_top)/2, VARIANCE), y_bot+delta, y_top-delta)
    return x, y

def generate_items(config):
    items = []
    platforms = []

    grid_config = config["grid"]
    player_config = config["player"]
    platform_center = np.clip(
        np.random.normal(grid_config["width"] / 2, np.sqrt(grid_config["width"])),
        player_config["length"] / 2,
        grid_config["width"] - player_config["length"] / 2
    )

    direction = np.random.choice([0, 𝛑])    
    platforms.append(
        PlayerPlatform(platform_center, player_config["y"], SPEED,  direction, length=player_config["length"])
    )

    balls_config = config["balls"]
    for _ in range(balls_config["number"]):
        direction = np.random.uniform()
        x, y = get_random_location(balls_config["quadrant"], grid_config["width"], grid_config["height"])
        items.append(Ball(x, y, BALL_SPEED, direction * 𝛕))


    platform_config = config["platform"]
    for _ in range(balls_config["number"]):
        x, y = get_random_location(platform_config["quadrant"], grid_config["width"], grid_config["height"])
        platforms.append(Platform(x, y, 0, 0))

    # env = BallEnv(items, platforms, INTERNAL, VISIBLE)
    return items, platforms


if __name__ == "__main__":
    
    # start = time.time()
    env = BallEnv(config=DEFAULT_CONFIG)
    for _ in range(10):
        env.reset()
        for _ in range(100):
            env.step(0)

    # end = time.time()
    # print(1000 / (end - start))
    env.close()



    # import pyglet
    # import time

    # SPEED = 0.01
    # env = BallEnv(
    #     [
    #         Ball(10, 10, SPEED, 0.25 * 𝛑),
    #         Ball(10, 10, SPEED, 0.65 * 𝛑),
    #         Ball(25, 40, SPEED, 0.50 * 𝛑),
    #         Ball(10, 40, SPEED, 1.65 * 𝛑),
    #         Ball(10, 50, SPEED, 5.65 * 𝛑),
    #         Ball(10, 10, SPEED, 0.22 * 𝛑),
    #         Ball(10, 10, SPEED, 0.54 * 𝛑),
    #         Ball(25, 40, SPEED, 3.64 * 𝛑),
    #         Ball(10, 40, SPEED, 4.65 * 𝛑),
    #         Ball(10, 50, SPEED, 5.65 * 𝛑),
    #     ],
    #     [
    #         PlayerPlatform(32, 1, SPEED, 𝛕, length=8),
    #         Platform(25, 50, 0, 0),
    #         Platform(45, 25, 0, 0),
    #     ],
    #     visible=True,
    #     internal_steps=100
    # )
    # STEPS = 1000
    # start = time.time()
    # for _ in range(STEPS):
    #     env.step("n/a")
    # end = time.time()
    # print(STEPS / (end - start))
    # env.close()

