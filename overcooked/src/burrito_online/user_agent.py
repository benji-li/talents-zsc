# this file allows a user to play with a greedy agent

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import argparse
import copy
import datetime
import json
import os
import shutil
from time import time
from typing import List

import numpy as np
import overcooked_ai_py.agents.agent as agent
import pygame
from moviepy.editor import ImageSequenceClip
from overcooked_ai_py.mdp.overcooked_mdp import (
    Action,
    Direction,
    ObjectState,
    PlayerState,
)
from overcooked_ai_py.static import TESTING_DATA_DIR
from overcooked_ai_py.utils import load_from_json
from pygame.locals import *

from .agents.steak_agent import SteakGreedyHumanModel
from .mdp.steakhouse_env import SteakhouseEnv
from .mdp.steakhouse_mdp import (
    SteakhouseGridworld,
    dishname2ingradient,
    ingradient2dishname,
)
from .visualization.state_visualizer import SteakhouseStateVisualizer

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 1000

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False
MAX_STEPS = 20000
USER_STUDY_LOG = os.path.join(os.getcwd(), "user_study/log")
TIMER = pygame.USEREVENT + 1
VIDEO_FPS = 10


class LocalOvercookedPygame:
    """
    Class to run the game in Pygame.
    Args:
        - env: Steakhouse env
        - agent1: First agent
        - agent2: Second agent
        - logger: Logger (instance of the class defined below)
        - game_time: Number of seconds the game should last (max 1000; default 30)
        - agents: List of agent IDs that are supposed to be greedy
        - ticks_per_ai_action: Time taken in ticks (each tick is 100ms) before the greedy
            agent takes the next action (default 4)
    """

    def __init__(
        self,
        env: SteakhouseEnv,
        agent1: agent.Agent,
        agent2: agent.Agent,
        logger,
        game_time: int = 30,
        agents: List[int] = None,
        ticks_per_ai_action: int = 4,
    ):
        self._running = True
        self.logger = logger
        self.env = env
        self.score = 0
        self.max_time = min(int(game_time), MAX_GAME_TIME)
        self.max_players = 2
        self.ticks_per_ai_action = ticks_per_ai_action
        self.agent1 = agent1
        self.agent2 = agent2
        self.init_time = time()
        self.player_1_action = Action.STAY
        self.player_2_action = Action.STAY

        self.agents = agents or []

    def on_init(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            (self.env.mdp.width * 30, self.env.mdp.height * 30 + 140), pygame.RESIZABLE
        )
        print(pygame.display.get_surface().get_size())
        # Initialize agents
        # self.agent1.set_agent_index(self.agent_idx)
        self.agent1.set_mdp(self.env.mdp)
        # self.agent2.set_agent_index(self.agent_idx+1)
        self.agent2.set_mdp(self.env.mdp)
        self.start_time = time()
        pygame.time.set_timer(TIMER, self.ticks_per_ai_action * 100)

        ds = load_from_json(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "config",
                "kitchen_config.json",
            )
        )
        test_dict = copy.deepcopy(ds)
        print(test_dict["config"])
        self.state_visualizer = SteakhouseStateVisualizer(**test_dict["config"])
        self._running = True

        self.logger.env = self.env

    def on_event(self, event):
        done = False

        player_1_action = Action.STAY
        player_2_action = Action.STAY

        # Players stay in place if no keypress are detected
        if event.type == TIMER:
            self.env.mdp.step_environment_effects(self.env.state)

            if 1 in self.agents:
                player_1_action, _ = self.agent1.action(self.env.state)
                self.player_1_action = player_1_action
            if 2 in self.agents:
                player_2_action, _ = self.agent2.action(self.env.state)
                self.player_2_action = player_2_action

        if event.type == pygame.KEYDOWN:
            pressed_keys = pygame.key.get_pressed()

            if 1 not in self.agents:
                if pressed_keys[pygame.K_UP]:
                    player_1_action = Direction.NORTH
                elif pressed_keys[pygame.K_RIGHT]:
                    player_1_action = Direction.EAST
                elif pressed_keys[pygame.K_DOWN]:
                    player_1_action = Direction.SOUTH
                elif pressed_keys[pygame.K_LEFT]:
                    player_1_action = Direction.WEST
                elif pressed_keys[pygame.K_SPACE]:
                    player_1_action = Action.INTERACT

            if 2 not in self.agents:
                if pressed_keys[pygame.K_w]:
                    player_2_action = Direction.NORTH
                elif pressed_keys[pygame.K_d]:
                    player_2_action = Direction.EAST
                elif pressed_keys[pygame.K_s]:
                    player_2_action = Direction.SOUTH
                elif pressed_keys[pygame.K_a]:
                    player_2_action = Direction.WEST
                elif pressed_keys[pygame.K_f]:
                    player_2_action = Action.INTERACT

            # check if action is valid
            if (
                player_1_action in Action.ALL_ACTIONS
                and player_2_action in Action.ALL_ACTIONS
            ):
                self.player_1_action = player_1_action
                self.player_2_action = player_2_action

        if event.type == pygame.QUIT or done:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

    def on_loop(self):
        self.logger.env = self.env
        time_now_in_milisecond = round(time() * 1000 - self.init_time * 1000)
        self.env.state.timestep = time_now_in_milisecond // 100

        ## change onloop to update game at 10fps,60 fps, apply joint action, update logger
        ## step environment every 0.1s/100ms,
        # 1 second = 1000ms
        if time_now_in_milisecond % 100 == 0:
            # print(time_now_in_milisecond)
            joint_action = (self.player_1_action, self.player_2_action)
            done = self._human_step_env(self.player_1_action, self.player_2_action)
            # log user behavior to json
            log = {
                "state": self.env.state.to_dict(),
                "joint_action": joint_action,
                "score": self.score,
            }
            self.logger.episode.append(log)

            # reinitialize action
            self.player_1_action = Action.STAY
            self.player_2_action = Action.STAY

            if self.logger.video_record:
                frame_name = self.logger.img_name(time_now_in_milisecond / 1000)
                pygame.image.save(self.screen, frame_name)

            if done:
                self._running = False

    def on_render(self):
        kitchen = self.state_visualizer.render_state(
            self.env.state,
            self.env.mdp.terrain_mtx,
            hud_data=self.state_visualizer.default_hud_data(
                self.env.state,
                time_left=self.env.horizon - self.env.state.timestep,
                # time_left=round(max(self.max_time - (time() - self.start_time), 0)),
                score=self.score,
            ),
        )
        self.screen.blit(kitchen, (0, 0))
        pygame.display.flip()

    def on_cleanup(self):
        self.logger.save_log_as_pickle()
        if self.logger.video_record:
            self.logger.create_video()
        pygame.quit()

    def on_execute(self):
        self.on_init()
        while self._running and not self._time_up():
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def _time_up(self):
        return time() - self.start_time > self.max_time

    def _human_step_env(self, human1_action, human2_action):
        joint_action = (human1_action, human2_action)
        prev_state = self.env.state
        self.state, info = self.env.mdp.get_state_transition(prev_state, joint_action)

        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action, joint_agent_action_info=[{"1"}, {"2"}]
        )
        return done

    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(self.max_time - (time() - self.start_time), 0)
        print(f"time left: {state_dict['time_left']}")
        return state_dict


class StudyConfig:
    def __init__(self, args):
        self.participant_id = args.participant_id
        self.layout_name = args.layout
        layout_file_name = self.layout_name + ".layout"

        self.log_file_name = args.log_file_name
        self.record_video = args.record_video

        # Copy layout to Overcooked AI code base
        path_from = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "layout",
            layout_file_name,
        )
        path_to = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "overcooked_ai",
            "src",
            "overcooked_ai_py",
            "data",
            "layouts",
            layout_file_name,
        )
        shutil.copy(path_from, path_to)

        if args.order_list:
            self.order_list = args.order_list
            start_all_orders = [dishname2ingradient(dish) for dish in args.order_list]
            self.start_all_orders = start_all_orders
            self.world_mdp = SteakhouseGridworld.from_layout_name(
                self.layout_name,
                start_all_orders=self.start_all_orders,
                order_list=self.order_list,
            )
        else:
            self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)

        self.total_time = args.total_time
        self.base_env = SteakhouseEnv.from_mdp(
            self.world_mdp, horizon=args.total_time * 10
        )  # horizon * 10 since each frame is 0.1s
        self.ticks_per_ai_action = args.ticks_per_ai_action


class Logger:
    def __init__(self, config, filename, agent1=None, agent2=None, video_record=False):
        self.participant_id = config.participant_id
        self.json_filename = filename + ".json"
        self.filename = filename
        self.video_record = config.record_video

        # create log folder
        self.log_folder = os.path.join(
            USER_STUDY_LOG,
            f"{self.participant_id}_"
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}",
        )
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.makedirs(self.log_folder)
        self.img_dir = os.path.join(self.log_folder, "img")
        os.makedirs(self.img_dir)
        self.img_name = lambda timestep: f"{self.img_dir}/{int(timestep*10):05d}.png"

        # game info
        # self.env = config.base_env
        self.layout_name = config.layout_name
        self.agent1 = agent1
        self.agent2 = agent2
        self.episode = []

    def save_log_as_pickle(self):
        with open(os.path.join(self.log_folder, self.json_filename), "w") as file:
            json.dump(
                {
                    "layout_name": self.layout_name,
                    "participant_id": self.participant_id,
                    "total_time": self.env.state.timestep,
                    "episode": self.episode,
                },
                file,
            )
        print(f"Pickle log saved to {self.json_filename}")

    def create_video(self, fps=10):
        image_files = sorted(
            [
                img
                for img in os.listdir(self.img_dir)
                if img.endswith(".jpg") or img.endswith(".png")
            ]
        )
        clips = [os.path.join(self.img_dir, img) for img in image_files]
        clip = ImageSequenceClip(clips, fps=fps)
        clip.write_videofile(
            os.path.join(self.log_folder, self.filename + ".mp4"), codec="libx264"
        )
        shutil.rmtree(self.img_dir)


def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description="Initialize configurations for a human study."
    )

    ### Args for the game setup ###
    parser.add_argument(
        "--layout",
        type=str,
        default="steak",
        help="List of tasks to be performed in the study",
    )
    parser.add_argument(
        "--order_list",
        type=str,
        nargs="+",
        help="List of dishes (steak_dish, chicken_dish, steak_onion_dish, boilded_chicken_onion_dish) to serve",
    )
    parser.add_argument(
        "--total_time",
        type=int,
        default=70,
        help="Total time to given to complete the game",
    )
    parser.add_argument(
        "--ticks_per_ai_action",
        type=int,
        default=4,
        help="Time taken in ticks (each tick is 100ms) before the greedy agent takes the next action (default 4)",
    )

    ### Args for the study ###
    parser.add_argument(
        "--participant_id", type=int, help="ID of participants in the study", default=0
    )
    parser.add_argument("--log_file_name", type=str, default="", help="Log file name")
    parser.add_argument(
        "--record_video",
        dest="record_video",
        action="store_true",
        help="Record video during replay",
    )
    parser.add_argument(
        "--no-record_video",
        dest="record_video",
        action="store_false",
        help="Do not record video during replay",
    )

    args = parser.parse_args()

    if args.log_file_name == "":
        args.log_file_name = "-".join([str(args.participant_id), args.layout])

    return StudyConfig(args)


if __name__ == "__main__":
    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    print(f"Participant ID: {study_config.participant_id}")
    print(f"Layout: {study_config.layout_name}")
    print(study_config.base_env.mdp.terrain_mtx)
    print("Orders:")
    for i, task in enumerate(study_config.base_env.mdp.order_list, start=1):
        print(f"{i}. {task}")

    # Initialize human agent
    agent1 = agent.Agent()
    agent1.set_agent_index(0)

    # Intialize greedy agent
    human_model_config = {
        "mlam_params": {
            "start_orientations": False,
            "wait_allowed": True,
            "same_motion_goals": False,
        },
        "hl_boltzmann_rational": False,
        "ll_boltzmann_rational": False,
        "hl_temp": 1,
        "ll_temp": 1,
        "auto_unstuck": False,
        "mdp": study_config.world_mdp,
    }
    agent2 = SteakGreedyHumanModel(**human_model_config)
    agent2.set_agent_index(1)

    # Initialize logging
    logger = Logger(
        study_config, study_config.log_file_name, agent1=agent1, agent2=agent2
    )
    gameapp = LocalOvercookedPygame(
        study_config.base_env,
        agent1,
        agent2,
        logger,
        game_time=study_config.total_time,
        agents=[2],
        ticks_per_ai_action=study_config.ticks_per_ai_action,
    )
    gameapp.on_execute()
