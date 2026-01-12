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

from .mdp.steakhouse_env import SteakhouseEnv
from .mdp.steakhouse_mdp import SteakhouseGridworld, dishname2ingradient
from .planners.steak_planner import SteakMediumLevelActionManager
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
TIMER, t = pygame.USEREVENT + 1, 1000
VIDEO_FPS = 10
NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}


class OvercookedPygame:
    """
    Class to run the game in Pygame.
    Args:
        - env: Steakhouse env.
        - agents: List of agents.
        - logger: Logger (instance of the class defined below)
        - game_time: Number of seconds the game should last (max 1000; default 30)
    """

    def __init__(
        self,
        env: SteakhouseEnv,
        agents: List[agent.Agent],
        logger,
        game_time: int = 30,
    ):
        self._running = True
        self.logger = logger
        self.env = env
        self.score = 0
        self.max_time = min(int(game_time), MAX_GAME_TIME)
        self.num_players = len(agents)
        self.ticks_per_ai_action = 1
        self.agents = agents
        self.init_time = time()
        self.prev_timestep = 0
        self.player_actions = [Action.STAY] * self.num_players

    def on_init(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            (self.env.mdp.width * 30, self.env.mdp.height * 30 + 140), pygame.RESIZABLE
        )

        # Initialize agents
        for agent in self.agents:
            agent.set_mdp(self.env.mdp)

        self.start_time = time()
        pygame.time.set_timer(TIMER, t)

        ds = load_from_json(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "config",
                "kitchen_config.json",
            )
        )
        test_dict = copy.deepcopy(ds)
        self.state_visualizer = SteakhouseStateVisualizer(**test_dict["config"])
        self._running = True

        self.logger.env = self.env

    def on_event(self, event):
        done = False
        player_actions = self.player_actions

        if event.type == TIMER:
            self.env.mdp.step_environment_effects(self.env.state)

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict["key"]

            if pressed_key == pygame.K_UP:
                player_actions[0] = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                player_actions[0] = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                player_actions[0] = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                player_actions[0] = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                player_actions[0] = Action.INTERACT

            if pressed_key == pygame.K_w:
                player_actions[1] = Direction.NORTH
            elif pressed_key == pygame.K_d:
                player_actions[1] = Direction.EAST
            elif pressed_key == pygame.K_s:
                player_actions[1] = Direction.SOUTH
            elif pressed_key == pygame.K_a:
                player_actions[1] = Direction.WEST
            elif pressed_key == pygame.K_f:
                player_actions[1] = Action.INTERACT

            if pressed_key == pygame.K_i and self.num_players == 3:
                player_actions[2] = Direction.NORTH
            elif pressed_key == pygame.K_l and self.num_players == 3:
                player_actions[2] = Direction.EAST
            elif pressed_key == pygame.K_k and self.num_players == 3:
                player_actions[2] = Direction.SOUTH
            elif pressed_key == pygame.K_j and self.num_players == 3:
                player_actions[2] = Direction.WEST
            elif pressed_key == pygame.K_SEMICOLON and self.num_players == 3:
                player_actions[2] = Action.INTERACT

            # check if action is valid
            if all([x in Action.ALL_ACTIONS for x in player_actions]):
                self.player_actions = player_actions

        if event.type == pygame.QUIT:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

    def on_loop(self, fps=10):
        self.logger.env = self.env
        time_step = round((time() - self.init_time) * fps)
        self.env.state.timestep = time_step

        ## change onloop to update game at 10fps, 60 fps, apply joint action, update logger
        ## step environment every 0.01s/10ms,
        # 1 second = 1000ms
        if time_step > self.prev_timestep:
            self.prev_timestep = time_step

            done = self._human_step_env(self.player_actions)
            joint_action = self.player_actions.copy()

            # log user behavior to json
            log = {
                "state": self.env.state.to_dict(),
                "joint_action": joint_action,
                "score": self.score,
            }
            self.logger.episode.append(log)

            # reinitialize action
            for i in range(self.num_players):
                self.player_actions[i] = Action.STAY

            if self.logger.video_record:
                frame_name = self.logger.img_name(time_step / fps)
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
        # on top of the kitchen, render fog
        for agent in self.agents:
            self.state_visualizer.render_fog(kitchen, self.env, agent)

        self.screen.blit(kitchen, (0, 0))
        pygame.display.flip()

    def on_cleanup(self):
        self.logger.save_log_as_pickle()
        if self.logger.video_record:
            self.logger.create_video()
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        while self._running and not self._time_up():
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def _time_up(self):
        return time() - self.start_time > self.max_time

    def _human_step_env(self, joint_action):
        prev_state = self.env.state
        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action, joint_agent_action_info=[{"1"}, {"2"}, {"3"}]
        )

        self.state = next_state
        curr_reward = sum(info["sparse_r_by_agent"])
        self.score += curr_reward

        return done

    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(self.max_time - (time() - self.start_time), 0)
        return state_dict


class StudyConfig:
    def __init__(self, args):
        self.participant_id = args.participant_id
        self.layout_name = (
            args.layout
            if args.num_players == 2
            else "_".join([args.layout, str(args.num_players) + "p"])
        )
        layout_file_name = self.layout_name + ".layout"
        self.num_players = args.num_players
        self.total_time = args.total_time

        # Log info
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

        self.base_env = SteakhouseEnv.from_mdp(
            self.world_mdp, horizon=args.total_time * 10
        )  # horizon * 10 since each frame is 0.1s


class Logger:
    def __init__(self, config, filename, agents=[], video_record=False):
        self.participant_id = config.participant_id
        self.json_filename = filename + ".json"
        self.filename = filename
        self.video_record = config.record_video
        self.total_time = config.total_time

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
        self.env = config.base_env
        self.layout_name = config.layout_name
        self.agents = agents
        self.horizon = config.base_env.horizon
        self.episode = []

    def save_log_as_pickle(self):
        with open(os.path.join(self.log_folder, self.json_filename), "w") as file:
            json.dump(
                {
                    "layout_name": self.layout_name,
                    "participant_id": self.participant_id,
                    "horizon": self.horizon,
                    "total_time": self.env.state.timestep,
                    #    "time_left": round( max(self.env.horizon - (time() - self.env.), 0)),
                    #    "time_elapsed": round(time() - self.start_time),
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
        default=300,
        help="Total seconds given to complete the game",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=2,
        help="Number of players in the game. Can be either 2 or 3 players.",
    )

    # The following game config options are still undergoing construction
    # parser.add_argument('--served_in_order', type=bool, help='Complete the order list in order')
    # parser.add_argument('--single_player', type=bool, help='Single player mode: one human controlled agent collaborating with a modeled greedy agent')

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
        args.log_file_name = "-".join(
            [str(args.participant_id), args.layout, str(args.num_players) + "p"]
        )

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
    print(study_config.world_mdp.order_list)

    # Initialize two human agent
    mlam = SteakMediumLevelActionManager(study_config.world_mdp, NO_COUNTERS_PARAMS)

    agents = []
    for i in range(study_config.num_players):
        tmp_agent = agent.Agent()
        tmp_agent.set_agent_index(i)
        agents.append(tmp_agent)

    # Initialize logging
    logger = Logger(study_config, study_config.log_file_name, agents=agents)
    gameapp = OvercookedPygame(
        study_config.base_env, agents, logger, game_time=study_config.total_time
    )
    gameapp.on_execute()
