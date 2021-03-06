import dataclasses
import math

import numpy as np

# from gym.envs.classic_control import rendering
from gym import core, spaces
from gym.utils import seeding
import cv2 as cv

# import time
# import pygame

NUM_ACTIONS = 3

SCREEN_WIDTH = 32  # 500
SCREEN_HEIGHT = 32  # 500
SPEED = 0.02
INTERNAL = 50
VISIBLE = False

WIDTH = 32
HEIGHT = 32
CHANNELS = 1

VARIANCE = 0.5
PLAYER_SPEED = 0.75 * SPEED
# INITIAL_BALL_SPEED = 0.25 * SPEED
BALL_SPEED = 1.0 * SPEED
# MAX_BALL_SPEED = 1.5 * SPEED
# MIN_BALL_SPEED = 0.5 * SPEED

PLAYER = 6

BOTTOM_DANGER = True
DEFAULT_CONFIG = {
    "MiniBall0-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 0, "quadrant": "1",},
    },
    "MiniBall1-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "1",},
    },
    "MiniBall2-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "2",},
    },
    "MiniBall3-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "4",},
        "platform": {"number": 1, "quadrant": "3",},
    },
    "MiniBall4-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "4",},
    },
    "MiniBall5-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "5",},
    },
    "MiniBall6-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "6",},
    },
    "MiniBall7-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "7",},
    },
    "MiniBall8-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "8",},
    },
    "MiniBallStar-v0": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 1, "quadrant": "3",},
        "platform": {"number": 1, "quadrant": "star",},
    },
    "MiniBall-v2": {
        "grid": {"height": HEIGHT, "width": WIDTH},
        "player": {"y": 5, "length": PLAYER},
        "balls": {"number": 2, "quadrant": "3",},
        "platform": {"number": 2, "quadrant": "5",},
    },
}

SKIP_ACTION = 0
BOOST_LEFT_ACTION = 1
BOOST_RIGHT_ACTION = 2

𝛕 = 2 * math.pi
𝛑 = math.pi


class OpenCvViewer:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.display = np.zeros((height, width, 1), dtype=np.uint8)
        self.viewer = None

    def clear(self):
        self.display.fill(0)

    def draw_circle(self, x, y, radius, color):
        cv.circle(
            self.display, (np.float32(x), np.float32(y)), radius, color, -1, cv.LINE_AA
        )

    def draw_rectangle(self, x, y, width, height, color):
        l, r, b, t = (x - width / 2, x + width / 2, y - height / 2, y + height / 2)
        cv.rectangle(
            self.display,
            pt1=(np.float32(l), np.float32(b)),
            pt2=(np.float32(r), np.float32(t)),
            color=color,
            thickness=-1,
        )

    def render(self, mode="human"):
        return self.display

    def close(self):
        cv.destroyAllWindows()


@dataclasses.dataclass
class Ball:
    x: float
    y: float
    v: float
    direction: float
    color: tuple = 1  # (0, 0, 255)  # BGR
    radius: int = 1
    t: int = 0
    was_at_boundary: int = 0

    def draw(self, viewer):
        # transform = rendering.Transform(translation=(self.x, self.y))
        # circ = viewer.draw_circle(self.radius)
        # circ.set_color(*self.color)
        # circ.add_attr(transform)
        viewer.draw_circle(self.x, self.y, radius=self.radius, color=self.color)

    def step(self, platforms, player_platform):
        n_x, n_y = movement_function(self)
        at_boundary = propel_boundary_function(n_x, n_y, self)
        intersects_player = intersects_platform(n_x, n_y, self.radius, player_platform)

        if not at_boundary:
            for p in platforms:
                intersected = intersects_platform(n_x, n_y, self.radius, p)
                if intersected:
                    finished = False

                    # First, we check if we hit a corner.
                    l, r, t, b = (
                        p.x + -p.width / 2,
                        p.x + p.width / 2,
                        p.y + p.height / 2,
                        p.y + -p.height / 2,
                    )
                    corners = [(l, b), (l, t), (r, t), (r, b)]
                    for point in corners:
                        # Is the corner inside the circle from the next location?
                        # This is not perfect logic... it could be true, but it would have hit the side first.
                        # If our step size was really small, this would be fine...
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
                        if intersects_player:
                            delta = (player_platform.x - self.x) / (
                                player_platform.width / 2
                            )
                            self.direction = (𝛕 - self.direction - delta) % 𝛕
                        else:
                            # Top / bot side.
                            if (p.x - p.width / 2 <= self.x) and (
                                self.x <= p.x + p.width / 2
                            ):
                                self.direction = (𝛕 - self.direction) % 𝛕
                            # Left / right side
                            else:
                                self.direction = (3 * 𝛑 - self.direction) % 𝛕
                    at_boundary = True
                    break

        # If we're not about to hit something move.
        if not at_boundary:
            self.x = n_x
            self.y = n_y
            # self.v = max(0.999 * self.v, MIN_BALL_SPEED)
            self.was_at_boundary = 0
        else:
            # self.v = min(1.01 * self.v, MAX_BALL_SPEED)
            # If we're playing for keeps, the bottom is dangerous!
            if BOTTOM_DANGER:
                if n_y - self.radius <= 1:
                    return True
            if self.was_at_boundary > 10:
                # game stuck
                return True
            self.was_at_boundary += 1

        # All is well.
        return False


@dataclasses.dataclass
class Platform:
    x: float
    y: float
    v: float
    direction: float
    color: tuple = 1  # (0, 0, 255)  # BGR
    width: int = 12
    height: int = 1
    t: int = 0

    def draw(self, viewer):
        viewer.draw_rectangle(self.x, self.y, self.width, self.height, color=self.color)
        # transform = rendering.Transform(translation=(self.x, self.y))
        # l, r, t, b = -self.width / 2, self.width / 2, self.width / 2, -self.width / 2
        # poly = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        # poly.set_color(*self.color)
        # poly.add_attr(transform)

    def step(self):
        pass


@dataclasses.dataclass
class PlayerPlatform:
    x: float
    y: float
    v: float
    direction: float
    color: tuple = 1  # (0, 0, 255)  # BGR
    width: int = 16
    height: int = 1
    t: int = 0

    def draw(self, viewer):
        viewer.draw_rectangle(self.x, self.y, self.width, self.height, color=self.color)
        # transform = rendering.Transform(translation=(self.x, self.y))
        # l, r, t, b = -self.width / 2, self.width / 2, self.width / 2, -self.width / 2
        # poly = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        # poly.set_color(*self.color)
        # poly.add_attr(transform)

    def step(self, items):
        n_x = self.x + self.v * math.cos(self.direction)
        n_y = self.y

        for i in items:
            # This makes it so the ball can't get stuck within the platform.
            if item_intersects_platform(
                i.x, i.y, n_x, n_y, i.radius, self.width, self.height
            ):
                self.v = 0
                return

        if (n_x - (self.width / 2) <= 0) or (n_x + (self.width / 2) >= WIDTH):
            self.direction = (3 * 𝛑 - self.direction) % 𝛕
        else:
            self.x = n_x


class BallEnv(core.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 2500}

    def __init__(self, config_name, mode="rgb_array"):
        internal_steps = INTERNAL
        visible = VISIBLE
        self.mode = mode

        self.viewer = OpenCvViewer(HEIGHT, WIDTH)
        self.internal_steps = internal_steps
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(WIDTH, HEIGHT, CHANNELS), dtype=np.uint8
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
        return self.viewer.render(mode).copy()

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
        return self.step(0)[0]

    def step(self, action):
        # before_obs = self.viewer.display
        if action == BOOST_LEFT_ACTION:
            self.player_platform.v = PLAYER_SPEED
            self.player_platform.direction = 𝛑
        elif action == BOOST_RIGHT_ACTION:
            self.player_platform.v = PLAYER_SPEED
            self.player_platform.direction = 0
        elif action == SKIP_ACTION:
            self.player_platform.v = 0.75 * self.player_platform.v

        done = False
        for _ in range(self.internal_steps):
            # Skip player platform.
            for i in self.platforms[:-1]:
                i.step()
            self.player_platform.step(self.items)
            for i in self.items:
                done |= i.step(self.platforms, self.player_platform)

        self.viewer.clear()
        for i in self.items:
            i.draw(self.viewer)
        for i in self.platforms:
            i.draw(self.viewer)

        reward = 1 if not done else -1
        obs = self.viewer.display

        return obs.copy(), reward, done, {}


def reflect_boundary_function(n_x_loc, n_y_loc, item):
    if (n_x_loc - item.radius <= 0) or (n_x_loc + item.radius >= WIDTH):
        item.direction = (3 * 𝛑 - item.direction) % 𝛕
        return True
    elif (n_y_loc - item.radius <= 0) or (n_y_loc + item.radius >= HEIGHT):
        item.direction = (𝛕 - item.direction) % 𝛕
        return True
    return False


def propel_boundary_function(n_x_loc, n_y_loc, item):
    if (n_x_loc - item.radius <= 1) or (n_x_loc + item.radius >= WIDTH - 1):
        delta = (n_y_loc - item.y) / (HEIGHT / 2)
        item.direction = (3 * 𝛑 - item.direction + delta) % 𝛕

        # prevent for-ever bouncing.
        updown = 1 if np.sin(item.direction) > 0 else -1
        if abs(item.direction - 0) < 0.15 or abs(item.direction - 𝛑) < 0.15:
            item.direction += 0.25 * updown

        return True
    elif (n_y_loc - item.radius <= 1) or (n_y_loc + item.radius >= HEIGHT - 1):
        delta = (n_x_loc - item.x) / (WIDTH / 2)
        item.direction = (𝛕 - item.direction + delta) % 𝛕
        return True
    return False


def intersects_platform(n_x, n_y, radius, platform):
    item_x_boundary = n_x - radius, n_x + radius
    item_y_boundary = n_y - radius, n_y + radius
    platform_x_boundary = (
        platform.x - platform.width / 2,
        platform.x + platform.width / 2,
    )
    platform_y_boundary = (
        platform.y - platform.height / 2,
        platform.y + platform.height / 2,
    )
    if intersects(item_x_boundary, platform_x_boundary) and intersects(
        item_y_boundary, platform_y_boundary
    ):
        return True
    return False


def item_intersects_platform(
    n_x, n_y, p_x, p_y, radius, platform_width, platform_height
):
    item_x_boundary = n_x - radius, n_x + radius
    item_y_boundary = n_y - radius, n_y + radius
    platform_x_boundary = (
        p_x - platform_width / 2,
        p_x + platform_width / 2,
    )
    platform_y_boundary = (
        p_y - platform_height / 2,
        p_y + platform_height / 2,
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


def get_random_location(quadrant, width, height, variance=VARIANCE):
    """
    Gets a random location in the given quadrant (or octant).

    Normally distributed from the center with var VARIANCE.

    Paramters
    ---------
    quadrant : ``str``
        Specifies which quadrant to sample from.
    width, height : ``int``
        Dimensions of the screen's underlying grid.


    ---------
    | 1 | 2 |
    | 3 | 4 |
    | 5 | 6 |
    | 7 | 8 |
    ---------
    """

    if quadrant == "1":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height * 3 / 4, height
    elif quadrant == "2":
        x_left, x_right = width / 2, width
        y_bot, y_top = height * 3 / 4, height
    elif quadrant == "3":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height / 2, height * 3 / 4
    elif quadrant == "4":
        x_left, x_right = width / 2, width
        y_bot, y_top = height / 2, height * 3 / 4
    elif quadrant == "5":
        x_left, x_right = 0, width / 2
        y_bot, y_top = height * 1 / 4, height * 1 / 2
    elif quadrant == "6":
        x_left, x_right = width / 2, width
        y_bot, y_top = height * 1 / 4, height * 1 / 2
    elif quadrant == "7":
        x_left, x_right = 0, width / 2
        y_bot, y_top = 0, height * 1 / 4
    elif quadrant == "8":
        x_left, x_right = width / 2, width
        y_bot, y_top = 0, height * 1 / 4
    elif quadrant == "star":
        # the ball can appear anywhere on the top portion of the game.
        x_left, x_right = 0, width
        y_bot, y_top = height * 3 / 4, height
        return (
            np.clip(np.random.uniform(x_left, x_right), x_left, x_right),
            np.clip(np.random.uniform(x_left, x_right), y_bot, y_top),
        )

    return (
        np.clip(np.random.normal((x_left + x_right) / 2, variance), x_left, x_right),
        np.clip(np.random.normal((y_bot + y_top) / 2, variance), y_bot, y_top),
    )


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
        0,
        direction,
        width=player_config["length"],
    )

    balls_config = config["balls"]
    for _ in range(balls_config["number"]):
        direction = np.random.uniform(0.15, 0.35)
        x, y = get_random_location(
            balls_config["quadrant"], grid_config["width"], grid_config["height"]
        )
        items.append(Ball(x, y, BALL_SPEED, direction * 𝛕))

    platform_config = config["platform"]
    for _ in range(platform_config["number"]):
        x, y = get_random_location(
            platform_config["quadrant"], grid_config["width"], grid_config["height"]
        )
        platforms.append(Platform(x, y, 0, 0))

    platforms.append(Platform(0, HEIGHT / 2, 0, 0, width=1, height=HEIGHT))
    platforms.append(Platform(WIDTH, HEIGHT / 2, 0, 0, width=1, height=HEIGHT))
    platforms.append(Platform(WIDTH / 2, HEIGHT, 0, 0, width=WIDTH, height=1))
    platforms.append(player_platform)

    return items, platforms, player_platform


class BallEnv0(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall0-v0")


class BallEnv1(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall1-v0")


class BallEnv2(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall2-v0")


class BallEnv3(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall3-v0")
        # Used for training distribution.


class BallEnv4(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall4-v0")
        # Used for test distribution.


class BallEnv5(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall5-v0")


class BallEnv6(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall6-v0")


class BallEnv7(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall7-v0")


class BallEnv8(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBall8-v0")


class BallEnvStar(BallEnv):
    def __init__(self):
        super().__init__(config_name="MiniBallStar-v0")

