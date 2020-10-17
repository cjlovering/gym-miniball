import dataclasses
import math

import cv2 as cv
import numpy as np

from gym.envs.classic_control import rendering
from gym import core, spaces
from gym.utils import seeding

𝛕 = 2 * math.pi
𝛑 = math.pi

NUM_ACTIONS = 3
SCREEN_WIDTH = 100
SCREEN_HEIGHT = 100
WIDTH = 64
HEIGHT = 64
CHANNELS = 3

VISIBLE = True
SPEED = 0.02
INTERNAL = 100
VARIANCE = 4
PLAYER_SPEED = 0.75 * SPEED
BALL_SPEED = 1 * SPEED
MAX_BALL_SPEED = 1.5 * SPEED

BOTTOM_DANGER = True
DEFAULT_CONFIG = {
    "MiniBall-v1": {
        "grid": {"height": 64, "width": 64},
        "player": {"y": 5, "length": 8},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "5",},
    },
    "MiniBall-v2": {
        "grid": {"height": 64, "width": 64},
        "player": {"y": 5, "length": 8},
        "balls": {"number": 2, "quadrant": "3",},
        "platform": {"number": 2, "quadrant": "5",},
    },
}

SKIP_ACTION = 0
BOOST_LEFT_ACTION = 1
BOOST_RIGHT_ACTION = 2


@dataclasses.dataclass
class Ball:
    x: float
    y: float
    v: float
    direction: float
    color: tuple = (0.1, 0.1, 0.1)  # r, b, g
    radius: int = 1
    t: int = 0

    def draw(self, viewer):
        transform = rendering.Transform(translation=(self.x, self.y))
        circ = viewer.draw_circle(self.radius)
        circ.set_color(*self.color)
        circ.add_attr(transform)

    def step(self, platforms):
        n_x, n_y = movement_function(self)
        at_boundary = reflect_boundary_function(n_x, n_y, self)

        for p in platforms:
            intersected = redirect_on_platform(n_x, n_y, self.radius, p)
            if intersected:
                finished = False

                # First, we check if we hit a corner.
                l, r, t, b = (
                    p.x + -p.length / 2,
                    p.x + p.length / 2,
                    p.y + p.width / 2,
                    p.y + -p.width / 2,
                )
                corners = [(l, b), (l, t), (r, t), (r, b)]
                for point in corners:
                    if point_inside(n_x, n_y, self.radius, point):
                        # Adopted the logic for handling corners from stackexchange:
                        # https://gamedev.stackexchange.com/questions/10911/a-ball-hits-the-corner-where-will-it-deflects
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

                # If we did not hit a corner, check if we hit the top/bot or a side.
                if not finished:
                    # Top / bot side.
                    if (p.x - p.length / 2 <= self.x) and (
                        self.x <= p.x + p.length / 2
                    ):
                        self.direction = (𝛕 - self.direction) % 𝛕
                    # Left / right side
                    else:
                        self.direction = (3 * 𝛑 - self.direction) % 𝛕
                at_boundary = True
                break

        # If we're not about to hit something (and thus bounce), move.
        if not at_boundary:
            self.x = n_x
            self.y = n_y
        else:
            self.v = max(1.01 * self.v, MAX_BALL_SPEED)
            # If we're playing for keeps, the bottom is dangerous!
            if BOTTOM_DANGER:
                if n_y - self.radius <= 0:
                    return True

        # All is well.
        return False


@dataclasses.dataclass
class Platform:
    x: float
    y: float
    v: float
    direction: float
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


@dataclasses.dataclass
class PlayerPlatform:
    x: float
    y: float
    v: float
    direction: float
    color: tuple = (0.8, 0.2, 0.2)  # r, b, g
    length: int = 16
    width: int = 1
    t: int = 0

    def draw(self, viewer):
        transform = rendering.Transform(translation=(self.x, self.y))
        l, r, t, b = -self.length / 2, self.length / 2, self.width / 2, -self.width / 2
        poly = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        poly.set_color(*self.color)
        poly.add_attr(transform)

    def step(self):
        n_x = self.x + self.v * math.cos(self.direction)
        if (n_x - (self.length / 2) <= 0) or (n_x + (self.length / 2) >= WIDTH):
            self.direction = (3 * 𝛑 - self.direction) % 𝛕
        else:
            self.x = n_x


class BallEnv(core.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 2500}

    def __init__(self, config_name="MiniBall-v1"):
        internal_steps = INTERNAL
        visible = VISIBLE

        self.viewer = None
        self.internal_steps = internal_steps
        self.observation_space = spaces.Box(
            low=0, high=256, shape=(WIDTH, HEIGHT, CHANNELS), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.state = None
        self.visible = visible
        self.seed()

        self.config_name = config_name
        self.config = DEFAULT_CONFIG[self.config_name]
        self.items, self.platforms, self.player_platform = generate_items(
            DEFAULT_CONFIG[self.config_name]
        )

    def render(self, mode):
        self.visible = True
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_HEIGHT, SCREEN_WIDTH)
            self.viewer.set_bounds(0, WIDTH, 0, HEIGHT)
        self.viewer.window.set_visible(self.visible)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.viewer.close()

    def reset(self):
        if self.config is None:
            assert False
        self.items, self.platforms, self.player_platform = generate_items(
            DEFAULT_CONFIG[self.config_name]
        )
        return self.step(0)

    def step(self, action):
        if action == BOOST_LEFT_ACTION:
            self.player_platform.v = PLAYER_SPEED
            self.player_platform.direction = 𝛑
        elif action == BOOST_RIGHT_ACTION:
            self.player_platform.v = PLAYER_SPEED
            self.player_platform.direction = 0
        elif action == SKIP_ACTION:
            pass

        done = False
        for _ in range(self.internal_steps):
            for i in self.platforms:
                i.step()
            for i in self.items:
                done |= i.step(self.platforms)

        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_HEIGHT, SCREEN_WIDTH)
            self.viewer.set_bounds(0, WIDTH, 0, HEIGHT)
        self.viewer.window.set_visible(self.visible)

        for i in self.items:
            i.draw(self.viewer)
        for i in self.platforms:
            i.draw(self.viewer)

        observation = self.viewer.render(return_rgb_array=True)
        reward = 1 if not done else 0
        return observation, reward, done, {}


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
    elif partition == "5":
        x_left, x_right = width / 2, width
        y_bot, y_top = height * 1 / 4, height * 1 / 2
    elif partition == "6":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height * 1 / 4, height * 1 / 2

    x = np.clip(
        np.random.normal((x_left + x_right) / 2, VARIANCE),
        x_left + delta,
        x_right - delta,
    )
    y = np.clip(
        np.random.normal((y_bot + y_top) / 2, VARIANCE), y_bot + delta, y_top - delta
    )
    return x, y


def generate_items(config):
    items = []
    platforms = []

    grid_config = config["grid"]
    player_config = config["player"]
    platform_center = np.clip(
        np.random.normal(grid_config["width"] / 2, np.sqrt(grid_config["width"])),
        player_config["length"] / 2,
        grid_config["width"] - player_config["length"] / 2,
    )

    direction = np.random.choice([0, 𝛑])
    player_platform = PlayerPlatform(
        platform_center,
        player_config["y"],
        SPEED,
        direction,
        length=player_config["length"],
    )

    balls_config = config["balls"]
    for _ in range(balls_config["number"]):
        direction = np.random.uniform(0.15, 0.35)
        x, y = get_random_location(
            balls_config["quadrant"], grid_config["width"], grid_config["height"]
        )
        items.append(Ball(x, y, BALL_SPEED, direction * 𝛕))

    platform_config = config["platform"]
    for _ in range(balls_config["number"]):
        x, y = get_random_location(
            platform_config["quadrant"], grid_config["width"], grid_config["height"]
        )
        platforms.append(Platform(x, y, 0, 0))
    platforms.append(player_platform)

    return items, platforms, player_platform


class BallEnv2(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall-v2")

