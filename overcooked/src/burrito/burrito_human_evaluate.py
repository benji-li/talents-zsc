import argparse
import copy
import datetime
import json
import os
import sys
import shutil
import time as tm
from time import time
from typing import List
import traceback

import random
import string
import yaml
import pickle
import zlib

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
# from .agents.steak_agent import SteakGreedyHumanModel
from burrito_rl.agents.burrito_agent import DummyAgent, RLlibAgent, get_agent

from burrito.encoding.env_enums import (
    counter_mapping,
    ingredient_mapping,
    kitchen_tool_mapping,
    held_item_mapping,
    recipe_mapping,
    orientation_mapping,
    action_mapping,
    KitchenToolType,
)

# from .mdp.burrito_env import SteakhouseEnv
from burrito.mdp.burrito_env import BurritoEnv
from burrito.mdp.burrito_mdp import (
    BurritoGridworld,
    BurritoState,
    dishname2ingradient,
)
# from .planners.steak_planner import SteakMediumLevelActionManager
# from .visualization.state_visualizer import SteakhouseStateVisualizer


## TODO: figure out how to not need this for mac
ffmpeg_path = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "ffmpeg")
os.environ["FFMPEG_BINARY"] = ffmpeg_path

import uuid
from html import escape

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 240
n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False
MAX_STEPS = 20000
USER_STUDY_LOG = os.path.join(os.getcwd(), "user_study/log")
TIMER, t = pygame.USEREVENT + 1, 200
VIDEO_FPS = 10
NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}
COLORS = ["red", "blue", "yellow", "green", "purple"]
completedGames = 0


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
        self.agents = args.agents
        self.players = []

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
            "..",
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
            self.all_recipes = args.all_recipes
            start_all_orders = [dishname2ingradient(dish) for dish in args.order_list]
            self.start_all_orders = start_all_orders
            self.world_mdp = BurritoGridworld.from_layout_name(
                self.layout_name,
                start_all_orders=self.start_all_orders,
                order_list=self.order_list,
                all_recipes=self.all_recipes,
            )
        else:
            self.world_mdp = BurritoGridworld.from_layout_name(self.layout_name)
        # start_state = SteakhouseState(
        #     [P((8, 1), s), P((1, 1), s)]
        # )
        # TODO: we basically modify here for our burrito mdp and environment
        self.base_env = BurritoEnv.from_mdp(
            self.world_mdp, horizon=args.total_time * 10
        )  # horizon * 10 since each frame is 0.1s

        shutil.copy(path_from, path_to)


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
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_"
            f"{uuid.uuid4()}",
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

    def save_log_as_pickle(self, game_id, info=None):
        # Optimize by using asynchronous I/O operations
        #import asyncio

        #async def async_save_log():
        if not info:
            with open(os.path.join(self.log_folder, self.json_filename), "w") as fh:
                fh.write(
                    json.dumps(
                        {
                            "layout_name": self.layout_name,
                            "participant_id": self.participant_id,
                            "horizon": self.horizon,
                            "total_time": self.env.state.timestep,
                            "episode": self.episode,
                        },
                        indent=4,
                    )
                )
        else:
            with open(os.path.join(self.log_folder, "info.json"), "w") as fh:
                fh.write(json.dumps(info, indent=4))

        print(f"Log saved to {self.json_filename}")

        #asyncio.run(async_save_log())


def initialize_config_from_args(settings):
    parser = argparse.ArgumentParser(
        description="Initialize configurations for a human study."
    )

    ### Args for the game setup ###
    parser.add_argument(
        "--layout", type=str, default=settings["layout"], help="Layout name"
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=settings["num_players"],
        help="Number of players",
    )
    parser.add_argument(
        "--total_time", type=int, default=settings["total_time"], help="Total game time"
    )
    parser.add_argument(
        "--order_list", type=list, default=settings["order_list"], help="List of orders"
    )

    parser.add_argument(
        "--agents", type=list, default=settings["agents"], help="List of agents"
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
            [
                str(args.participant_id),
                settings["layout"],
                str(settings["num_players"]) + "p",
            ]
        )

    return args


from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    url_for,
    make_response,
    has_app_context,
    redirect,
)
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import os
import threading
import time as tmodule
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, cpu_count
import multiprocessing
from enum import Enum
from queue import Empty


class OvercookedWebApp:
    def __init__(
        self,
    ):
        self._initialized = False
        self._main_loop_running = False
        self._players = []
        self.internal_agents = 0
        self.humans = []
        self._running = False
        self._waiting = True
        self.player_actions = []

        print("%" * 100)
        print("Overcooked Web App Constructed")
        print("%" * 100)

    def setPlayer(self, player_id):
        for player in self._players:
            if (
                player.get("client_id") in [None, player_id]
                and player.get("type") == "HUMAN"
            ):
                player["client_id"] = player_id
                player["active"] = True
                return player["client_id"], self.url_for_static(
                    "static",
                    filename=f"./chefs/chef-{'SOUTH'}-{player['color']}.png",
                )

    def deletePlayer(self, player_id):
        for player in self._players:
            if player.get("client_id") == player_id:
                player["client_id"] = None
                player["active"] = False
                if hasattr(player["agent"], "event_queue"):
                    player["agent"].event_queue.put(
                        {
                            "event": "remove_player",
                        }
                    )

    def __del__(self):
        print("DESTROYED")
        print(os.getpid())

    def call_repeatedly(self, interval, func):
        """Call a function repeatedly at a specified interval (in seconds)."""

        def wrapper():
            next_call = time()
            while self._running or self._waiting:
                with self.lock:
                    try:
                        # Check for commands from the queue
                        start_time = time()  # Record start time
                        while not self.command_queue.empty():
                            command = self.command_queue.get()
                            if command["event"] == "start_game":
                                self.kickoff_game()
                            elif command["event"] == "player_update":
                                client, chef_img = self.setPlayer(command["data"])
                                self.event_queue.put(
                                    {
                                        "event": "added_player",
                                        "data": {
                                            "message": f"{chef_img}",
                                            "client": client,
                                        },
                                        "room": str(self.game_id),
                                    }
                                )
                                self.kickoff_game()
                            elif command["event"] == "remove_player":
                                self.deletePlayer(command["data"])
                                self.event_queue.put(
                                    {
                                        "event": "deleted_player",
                                        "data": {
                                            "client": client,
                                        },
                                        "room": str(self.game_id),
                                    }
                                )
                            elif (
                                command["event"] == "key_press"
                                and self._players is not None
                            ):
                                player_index = next(
                                    i
                                    for i, player in enumerate(self._players)
                                    if player.get("client_id") == command["cid"]
                                )
                                if self._accept_input[player_index]:
                                    self.handle_key_press(
                                        command["cid"], command["key"]
                                    )
                                    self._accept_input[player_index] = False
                        # Handle actual game-logic only after both players joined
                        if self._main_loop_running:
                            func()  # Call the function
                            # Prepare the function call for the next iteration
                            self._accept_input = [
                                True for _ in range(game_settings["num_players"])
                            ]
                        end_time = time()  # Record end time
                        func_execution_time = (
                            end_time - start_time
                        )  # Calculate execution time
                    except Exception as e:
                        print(f"An error occurred --- : {e}")
                        traceback.print_exc()  # This will print the full traceback

                # Update the next call time and calculate remaining time to sleep
                next_call += interval
                sleep_time = next_call - time()  # Calculate remaining sleep time
                if sleep_time > 0:
                    if sleep_time < 0.05:
                        print("Sleep Time:", sleep_time)
                    socketio.sleep(sleep_time)
            self.on_cleanup()

        wrapper()

    def on_init(self):
        for i, player in enumerate(self._players):
            if (player["agent"] is None) and player["type"] == "GREEDY":
                player["agent"] = init_greedy_agent(self.world_mdp)
                player["agent"].set_mdp(self.env.mdp)
                player["agent"].set_agent_index(i)
            if (player["agent"] is None) and player["type"] == "HIERARCHICAL":
                player["agent"] = init_hierarchical_agent(self.world_mdp)
                player["agent"].set_mdp(self.env.mdp)
                player["agent"].set_agent_index(i)
            elif player["type"] == "HUMAN":
                player["agent"].set_mdp(self.env.mdp)
                player["agent"].set_agent_index(i)

        # Wait until all remote agents processes initialized
        print("Initializing agents...")
        to_initialize = 0
        for player in self._players:
            if hasattr(player['agent'], "command_queue"):
                to_initialize += 1
        while to_initialize > 0:
            for player in self._players:
                if hasattr(player['agent'], "command_queue"):
                    message = player["agent"].command_queue.get()
                    if message.get("event") == "setup":
                        to_initialize -= 1
                    player['agent'].event_queue.put(
                        {
                            "event": "waiting",
                        }
                    )
        print("Agents initialized!")

        self.start_time = time()

        ds = load_from_json(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "config",
                "kitchen_config.json",
            )
        )
        test_dict = copy.deepcopy(ds)
        # self.state_visualizer = SteakhouseStateVisualizer(**test_dict["config"])
        self._running = True

        self.logger.env = self.env
        if any(player["type"] == "HUMAN" for player in self._players):
            self.event_queue.put(
                {
                    "event": "game_initialized",
                    "data": {"message": f"Initialization complete"},
                    "room": str(self.game_id),
                }
            )
            # else:
            # self.command_queue.put({"event": "start_game"})

    def _make_this(self, args):
        study_config = StudyConfig(args)
        # Make Game stuff
        print("Study Configuration Initialized:")
        print(f"Participant ID: {study_config.participant_id}")
        print(f"Layout: {study_config.layout_name}")
        print(study_config.base_env.mdp.terrain_mtx)
        print("Orders:")
        for i, task in enumerate(study_config.base_env.mdp.order_list, start=1):
            print(f"{i}. {task}")

        # Initialize logging
        logger = Logger(
            study_config, study_config.log_file_name, agents=study_config.agents
        )

        # Initializing agents
        agents = []
        player_types = []
        for i, agent_type in enumerate(game_settings["agents"]):
            if agent_type == "HUMAN":
                p = {
                    "name": f"player-{i}",
                    "agent": agent.Agent(),
                    "type": "HUMAN",
                    "active": False,
                    "client_id": None,
                    "color": COLORS[i],
                }
                agents.append(p)
                player_types.append('H')
            
            # TODO: create non-human agent processes
            elif agent_type == "RLlib":
                # Load a RLlib policy from a rllib trainer and specified policy model path
                # Hard code to be PPO policy (PPOPolicy for now)
                # TODO: modify to initialize with trainer which can accomodate various policies
                assert game_settings["agents_config"][i] != "None"
                if not "burrito_rl.config.config_loader" in sys.modules:
                    from burrito_rl.config.config_loader import ConfigLoader
                    _config = ConfigLoader.load_config(game_settings["agents_config"][i]["config_name"])['BASE_CONFIG']
                    _config["num_gpus"] = 0 # disable cuda when playing games
                    _config["num_workers"] = 0 # reduce overhead of initalization RLlib agent
                    _config["num_envs_per_worker"] = 1
                p = {
                    "name": f"player-{i}",
                    "type": "RLLIB",
                    "active": True,
                    "client_id": f"{agent_type}-{i}",
                    "color": COLORS[i],
                    "agent": get_agent("rllib", i, 
                                       event_queue = Queue(), 
                                       command_queue = Queue(),
                                       mdp = study_config.world_mdp,
                                       timeout = 120,
                                       config = _config,
                                       #model_path = _config["pretrained_model_path_BR"]) #game_settings["agents_config"][i]["model_path"])
                                       model_path = game_settings["agents_config"][i]["model_path"])
                }
                p["agent"].start() # Start the agent process
                agents.append(p)
                player_types.append('A')
            elif agent_type == "Dummy":
                assert game_settings["agents_config"][i] != "None"
                if not "burrito_rl.config.config_loader" in sys.modules:
                    from burrito_rl.config.config_loader import ConfigLoader
                    _config = ConfigLoader.load_config(game_settings["agents_config"][i]["config_name"])['BASE_CONFIG']
                p = {
                    "name": f"player-{i}",
                    "type": "DUMMY",
                    "active": True,
                    "client_id": f"{agent_type}-{i}",
                    "color": COLORS[i],
                    "agent": get_agent("dummy", i, 
                                       event_queue = Queue(), 
                                       command_queue = Queue(),
                                       mdp = study_config.world_mdp,
                                       timeout = 120,
                                       config = _config)
                }
                p["agent"].start() # Start the agent process
                agents.append(p)
                player_types.append('A')
            else:
                p = {
                    "name": f"player-{i}",
                    "agent": None,
                    "type": agent_type,
                    "active": True,
                    "client_id": f"{agent_type}-{i}",
                    "color": COLORS[i],
                }
                agents.append(p)
                player_types.append('A')

        study_config.base_env.setup_planner(study_config.layout_name, player_types, restrict_capability=True)

        return (
            study_config.base_env,
            logger,
            study_config.total_time,
            study_config.world_mdp,
            agents,
        )


    # TODO: entrance of a single game
    # Modify from here to initialize agent processes
    def startup_game(self, command_queue, event_queue, game_id, args, fps=10):
        (
            env,
            logger,
            game_time,
            world_mdp,
            agents,
        ) = self._make_this(args)
        # Former init stuff
        self.event_queue = event_queue # Send event to a game server
        self.env = env
        self._players = agents

        self._running = True
        self.logger = logger
        self.score = 0
        print("Game Time:", game_time)
        print("Max Game Time:", MAX_GAME_TIME)
        self.max_time = min(int(game_time), MAX_GAME_TIME)
        self.num_players = len(agents)
        self.ticks_per_ai_action = 1
        self.agents = agents
        self.init_time = time()
        self.prev_timestep = 0
        # self.player_actions = [Action.ACTION_TO_INDEX[Action.STAY] for i in range(len(self.num_players))]
        self.player_actions = [Action.ACTION_TO_INDEX[Action.STAY]] * self.num_players
        # self.player_actions = [Action.STAY] * self.num_players

        self.lock = threading.Lock()
        self._waiting = True
        self._accept_input = [True for _ in range(self.num_players)]
        self._last_grid = {}
        self._online_grid = {}
        self._last_text = {}
        self._current_env_grid = np.zeros(
            (self.env.mdp.height, self.env.mdp.width, 11), dtype=np.int32
        )
        self._env_grids = []
        self._env_grids_dic = {}
        self._recording_images = []
        self._hud_data = {}
        self.game_id = game_id
        self.world_mdp = world_mdp

        # Actual stuff :)
        self.command_queue = command_queue
        self.on_init()
        self._initialized = True
        print("-----> Startup for game", game_id, "complete", self)
        try:
            tm.sleep(2)
            self.call_repeatedly(1 / fps, self.handle_game_logic)
        except Exception as e:
            print(f"An error occurred in call repeatedly: {e}")
            traceback.print_exc()

    def kickoff_game(self):
        # if not all humans active, return
        if not all(
            player.get("active")
            for player in self._players
            if player.get("type") == "HUMAN"
        ):
            return
        print("kicking off", self.game_id)
        print(self._main_loop_running)
        if not self._main_loop_running:
            print("setting main loop to run")
            self.logger.save_log_as_pickle(self.game_id, info=self._make_game_info())
            # Start background thread...
            self._main_loop_running = True
            self.start_time = time()
            self.init_time = time()
            self._running = True
            self._waiting = False

    def from_state_to_grid():
        # This is the current map
        cnt = 0
        data = {}
        online_data = {}
        text_data = {}
        # Precompute static file paths

    def on_loop(self, fps=10):
        self.logger.env = self.env
        time_step = round((time() - self.init_time) * fps)
        self.env.state.timestep = time_step

        # change onloop to update game at 10fps, 60 fps, apply joint action, update logger
        # step environment every 0.01s/10ms,
        # 1 second = 1000ms
        if time_step > self.prev_timestep:
            self.prev_timestep = self.env.state.timestep
            try:  # This is a hacky way to handle the fact that the game might end before the time is up
                # joint_action = tuple(
                #     tuple(action) if isinstance(action, tuple) else (action)
                #     for action in self.player_actions
                # )
                joint_action = self.player_actions
                done, info = self._human_step_env(joint_action)
            except Exception as e:
                print(f"Exception occurred!: {e}")
                traceback.print_exc()  # This will print the full traceback
                done = True

            if done:
                self._running = False
                self.waiting = False
            else:
                joint_action = self.player_actions.copy()
                # log user behavior to json
                log = {
                    "state": self.env.state.to_dict(),
                    "joint_action": joint_action,
                    "score": self.score,
                }
                # log my state here
                self.logger.episode.append(log)
                img = self._get_current_image()
                self.generate_grid(joint_action)

                # reinitialize action
                for i in range(self.num_players):
                    self.player_actions[i] = Action.ACTION_TO_INDEX[Action.STAY]
                return img, info
        return None, None

    def generate_grid(self, joint_action):
        # Generate the grid for the current state
        state = self.env.state
        self.convert_to_layer(state.players, state.objects)
        meta = self.get_metadata(joint_action)
        self._env_grids_dic[f"{len(self._env_grids_dic)}"] = {
            "state": np.array([self._current_env_grid]),
            "meta": np.array(meta),
        }
        self._current_env_grid = np.zeros(
            (self.env.mdp.height, self.env.mdp.width, 11), dtype=np.int32
        )  # Reset the current grid

    def on_render(self):
        result = {
            "orders": [r for r in self.env.state.order_list],
            "score": self.score,
            "time_left": self.env.horizon - self.env.state.timestep,
            "plates_available": self.env.state.num_plates,
        }
        return result

    def save_frames_to_video(self, fps=10):
        """
        Save a list of PIL image frames into a video file.

        :param frames: List of PIL.Image objects representing each frame.
        :param output_path: The path where the resulting video will be saved.
        :param fps: Frames per second for the output video.
        """
        # Convert PIL images to numpy arrays for moviepy
        numpy_frames = [np.array(frame) for frame in self._recording_images]

        print("frames in video:", len(numpy_frames))

        # Save the environment grid to a file
        env_grid_path = os.path.join(self.logger.log_folder, "env_grid.npy")
        states_array = np.array([v["state"] for v in self._env_grids_dic.values()])
        np.save(env_grid_path, states_array)

        print("Number of matrices: ", len(self._env_grids))

        # print(self._env_grids_dic)

        # Save the env_grid_dic to a file

        env_grid_dic_path = os.path.join(self.logger.log_folder, "env_grid.npz")
        np.savez(env_grid_dic_path, **self._env_grids_dic)
        env_grid_dic_path = os.path.join(self.logger.log_folder, "env_grid.json")

        with open(env_grid_dic_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        k: {"state": v["state"].tolist(), "meta": v["meta"].tolist()}
                        for k, v in self._env_grids_dic.items()
                    }
                )
            )

        print(f"Environment grid saved to {env_grid_dic_path}")
        # Create a video clip from the numpy arrays
        clip = ImageSequenceClip(numpy_frames, fps=fps)

        # Save each frame as a separate image file
        for i, frame in enumerate(numpy_frames):
            frame_image = Image.fromarray(frame)
            frame_image.save(
                os.path.join(self.logger.log_folder, "img", f"frame_{i:05d}.png")
            )

        # Get the video output path
        output_path = os.path.join(self.logger.log_folder, "video.mp4")

        # Write the video file
        clip.write_videofile(output_path, codec="libx264")
        print(f"Video saved to {output_path}")
        self._running = False

    def save_matrix_states(self):
        # Save the environment grid to a file
        import asyncio

        async def save_matrix():
            env_grid_path = os.path.join(self.logger.log_folder, "env_grid.npy")
            states_array = np.array([v["state"] for v in self._env_grids_dic.values()])
            np.save(env_grid_path, states_array)

            print("Number of matrices: ", len(self._env_grids))

            # print(self._env_grids_dic)

            # Save the env_grid_dic to a file

            env_grid_dic_path = os.path.join(self.logger.log_folder, "env_grid.npz")
            np.savez(env_grid_dic_path, **self._env_grids_dic)
            env_grid_dic_path = os.path.join(self.logger.log_folder, "env_grid.json")

            with open(env_grid_dic_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            k: {
                                "state": v["state"].tolist(),
                                "meta": v["meta"].tolist(),
                            }
                            for k, v in self._env_grids_dic.items()
                        }
                    )
                )
            print(f"Environment grid saved to {env_grid_dic_path}")

        asyncio.run(save_matrix())

    def on_cleanup(self):
        print("Running On Cleanup")
        id = self.game_id
        newGameId = generate_random_characters()
        self.event_queue.put(
            {
                "event": "update_title",
                "data": {"message": f"Game Complete!-{newGameId}"},
                "room": str(self.game_id),
            }
        )
        # self.socketio.emit("update_title", {"message": "Game Complete!"}, room=str(self.game_id))

        # NOTE: commented because of a logging issue
        # self.logger.save_log_as_pickle(self.game_id)
        self.event_queue.put({"event": "game_status", "data": "completed"})
        # if self.logger.video_record:
        #     # self.logger.create_video()
        #     self.save_frames_to_video()
        # NOTE: also commented bc of logging issue
        # self.save_matrix_states()

        # TODO: what does it mean on the finish here?
        # CLEAR UP AGENT PROCESSES
        for player in self._players:
            if hasattr(player["agent"], "event_queue"):
                player["agent"].event_queue.put(
                    {
                        "event": "remove_player",
                }
            )

        # START A NEW GAME!
        clients = {player["client_id"] for player in self._players}
        self.event_queue.put(
            {
                "event": "make_new_game",
                "data": {"room": newGameId, "num": 1, "clients": clients},
            }
        )

        self.command_queue.close()
        self.event_queue.close()

    def on_execute(self):
        pass

    def _time_up(self):
        return time() - self.start_time > self.max_time

    def _human_step_env(self, joint_action):
        try:
            # print("try to step env:")
            # print(joint_action)
            next_state, timestep_sparse_reward, done, info = self.env.step(
                joint_action,
                joint_agent_action_info=[{f"{i + 1}"} for i in range(self.num_players)],
            )
            self.state = next_state
            self.score += timestep_sparse_reward
            return done, info
        except Exception as e:
            print(f"Exception occurred xxxxx: {e}")
            traceback.print_exc()  # This will print the full traceback
            return True, {}

    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(self.max_time - (time() - self.start_time), 0)
        return state_dict

    def handle_game_logic(self):
        for i, player in enumerate(self._players):
            if hasattr(player["agent"], "command_queue"):
                # print("===============")
                try:
                    action = player["agent"].command_queue.get(block=False)
                except:
                    action = Action.ACTION_TO_INDEX[Action.STAY]
                self.player_actions[i] = action

            elif player["type"] != "HUMAN":
                # print("?????????????????????")
                action = player["agent"].action(self.env.state)
                self.player_actions[i] = action

        # TODO: a game image is in the online data
        # Substitute with state / obs that agents need
        online_data, info = self.on_loop()
        self._hud_data = self.on_render()
        state = self.env.state
        for i, player in enumerate(self._players):
            if hasattr(player["agent"], "event_queue"):
                # print(f"Try to send message to {player['type']}")
                player["agent"].event_queue.put(
                    {
                        "event": "playing",
                        "stats": (state.deepcopy(), info)
                    }
                )

        # Batch updates to minimize the number of network requests
        updates = {
            "update_tiles": {"images": online_data, "text": self._last_text},
            "update_hud": {"hud": self._hud_data},
        }
        self.event_queue.put(
            {
                "event": "batch_update",
                "data": updates,
                "room": str(self.game_id),
            }
        )

    def _make_game_info(self):
        return {
            "game_id": self.game_id,
            "players": [player.get("client_id") for player in self._players],
            "names": [player.get("name") for player in self._players],
        }

    def handle_key_press(self, cid, key):
        # get the client index in the self._players array
        player_index = next(
            i
            for i, player in enumerate(self._players)
            if player.get("client_id") == cid
        )

        # Handle logic
        if key == "up":
            self.player_actions[player_index] = Action.ACTION_TO_INDEX[Direction.NORTH]
        elif key == "right":
            self.player_actions[player_index] = Action.ACTION_TO_INDEX[Direction.EAST]
        elif key == "down":
            self.player_actions[player_index] = Action.ACTION_TO_INDEX[Direction.SOUTH]
        elif key == "left":
            self.player_actions[player_index] = Action.ACTION_TO_INDEX[Direction.WEST]
        elif key == "space":
            self.player_actions[player_index] = Action.ACTION_TO_INDEX[Action.INTERACT]

    def url_for_static(self, context, filename):
        target = "none"
        if has_app_context():
            target = url_for(context, filename=filename)
        else:
            target = f"/assets/{filename}"
        return target

    def get_current_images(self, cid):
        with self.lock:
            return self._last_grid, self._last_text

    def get_held_object(self, agent):
        held_object = str(agent.held_object)
        if "@(" in held_object:
            return held_object.split("@")[0]
        return held_object

    def convert_to_layer(self, players, dynamic_objects):
        terrain_mtx = self.env.mdp.terrain_mtx

        for row_idx, row in enumerate(terrain_mtx):
            for col_idx, cell in enumerate(row):
                self.handle_cell(row_idx, col_idx, cell)

        self.handle_dynamic_objects(dynamic_objects)
        self.handle_players(players)

    def handle_cell(self, row, col, cell):
        self._current_env_grid[row, col, 0] = counter_mapping.get(cell, 0)
        self._current_env_grid[row, col, 1] = ingredient_mapping.get(cell, 0)
        self._current_env_grid[row, col, 3] = kitchen_tool_mapping.get(cell, 0)

        # if cell == KitchenToolType.DIRTY_DISHES.value:
        #     self._current_env_grid[row, col, 4] = self.env.state.num_plates

    def handle_dynamic_objects(self, dynamic_objects):
        for obj in dynamic_objects.values():
            row, col = obj.position
            name = obj.name
            is_burnt = getattr(obj, "is_burnt", False)
            _is_extinguished = getattr(obj, "_is_extinguished", False)
            _cooking_tick = getattr(obj, "_cooking_tick", None)
            _warning_tick = getattr(obj, "_warning_tick", None)
            _waiting_tick = getattr(obj, "_waiting_tick", None)

            # print("Object Name", name)
            if _cooking_tick:
                # print("Cooking Tick", _cooking_tick)
                self._current_env_grid[col, row, 4] = _cooking_tick
            if _waiting_tick:
                # print("Waiting Tick", _waiting_tick)
                self._current_env_grid[col, row, 5] = _waiting_tick
            if _warning_tick:
                # print("Warning Tick", _warning_tick)
                self._current_env_grid[col, row, 6] = _warning_tick

            existing = self._current_env_grid[col, row, 1]

            self._current_env_grid[col, row, 2] = (
                held_item_mapping.get(name, None) or existing
            )

            self._current_env_grid[col, row, 7] = (
                1 if is_burnt else (2 if _is_extinguished else 0)
            )

    def convert_orientation(self, orientation):
        return (
            "SOUTH"
            if orientation[1] == 1
            else (
                "NORTH"
                if orientation[1] == -1
                else ("EAST" if orientation[0] == 1 else "WEST")
            )
        )

    def convert_action(self, action):
        if action == Action.ACTION_TO_INDEX[Action.INTERACT]:
            return "INTERACT"
        if action == Action.ACTION_TO_INDEX[Action.STAY]:
            return "NONE"
        return (
            "UP"
            if action == Action.ACTION_TO_INDEX[Direction.NORTH]
            else (
                "DOWN"
                if action == Action.ACTION_TO_INDEX[Direction.SOUTH]
                else ("LEFT" if action == Action.ACTION_TO_INDEX[Direction.WEST] else "RIGHT")
            )
        )

    def handle_players(self, players):
        for i, player in enumerate(players):
            index = i + 1
            row, col = player.position
            held_object = getattr(player.held_object, "name", "None")
            orientation = self.convert_orientation(player.orientation)
            self._current_env_grid[col, row, 8] = index  # PLAYER POSITION
            obj_map = held_item_mapping.get(held_object)
            # print("Held Object", held_object)
            # print("Object Map", obj_map)
            self._current_env_grid[col, row, 9] = held_item_mapping.get(
                held_object, 0
            )  # PLAYER HELD OBJECT - can be held item or ingredient
            self._current_env_grid[col, row, 10] = orientation_mapping.get(
                orientation, 0
            )  # PLAYER ORIENTATION

    def get_metadata(self, joint_action):
        max_recipes = 4
        game_info = 4
        arry_length = (
            game_info + max_recipes * 3 + len(joint_action)
        )  # + 1 just for 0 based indexing
        result = np.zeros((arry_length), dtype=np.int32)

        # game info
        result[0] = self.score
        result[1] = self.env.state.timestep
        result[2] = self.max_time
        result[3] = self.env.state.num_plates

        # order list
        index = game_info
        for i, order in enumerate(self.env.state.order_list[:max_recipes]):

            result[index] = recipe_mapping.get(order[0], 0)
            result[index + 1] = order[2] - order[1]
            result[index + 2] = order[2]
            index += 3

        # player actions
        for i, action in enumerate(joint_action):
            # print("Action", action)
            # print("converted action", self.convert_action(action))
            # print("action mapping", action_mapping.get(self.convert_action(action), 0))
            result[game_info + max_recipes * 3 + i] = action_mapping.get(
                self.convert_action(action), 0
            )
        return result

    def _get_current_image(self):
        # This is the current map
        cnt = 0
        data = {}
        text_data = {}
        terrain_mtx = self.env.mdp.terrain_mtx
        width = self.env.mdp.width

        # Precompute static file paths
        # static_file_paths = {
        #     "X": "counter.png",
        #     " ": "floor.png",
        #     "S": "deliver.png",
        #     "D": "dishes.png",
        #     "J": "dirty_plate.png",
        #     "K": "clean_plate.png",
        #     "W": "sink.png",
        #     "M": "meat-dispenser.png",
        #     "P": "pot.png",
        #     "B": "cutting_board.png",
        #     "G": "grill.png",
        #     "T": "tortilla-dispenser.png",
        #     "Z": "mushroom-dispenser.png",
        #     "U": "trash.png",
        #     "R": "rice-dispenser.png",
        # }

        # Precompute static file paths
        static_file_paths = {
            "X": "counter.png",
            " ": "floor.png",
            "S": "deliver.png",
            "D": "dishes.png",
            "J": "dirty_plate.png",
            "K": "clean_plate.png",
            "W": "sink.png",
            "M": "meat.png",
            "P": "pot.png",
            "B": "cutting_board.png",
            "G": "grill.png",
            "T": "tortillas.png",
            "Z": "mushroom.png",
            "U": "trash.png",
            "R": "rice.png",
        }

        for row in terrain_mtx:
            for cell in row:
                if cell in static_file_paths:
                    if cell == "D":
                        text_data[cnt] = {"plates": self.env.state.num_plates}
                    fileName = static_file_paths[cell]
                    data[cnt] = self.url_for_static("static", filename=fileName)
                    data[cnt] = [fileName, 0]
                else:
                    print(f"Unknown cell: {cell}")
                cnt += 1

        # Now place the players
        for i, agent in enumerate(self.env.state.players):
            x, y = agent.position
            color = COLORS[i]
            orientation = (
                "SOUTH"
                if agent.orientation[1] == 1
                else (
                    "NORTH"
                    if agent.orientation[1] == -1
                    else ("EAST" if agent.orientation[0] == 1 else "WEST")
                )
            )
            original_held_object = self.get_held_object(agent)

            if original_held_object == "None":
                fileName = f"{orientation}-{color}.png"
            else:
                held_object = (
                    original_held_object.replace("{✓", "clean_plate")
                    .replace("{","dirty_plate")
                    .replace("{!✓", "steak-dish")
                    .replace("steak_onion", "steak-onion-dish")
                    .replace("{%✓", "chopped_steak-plate")
                    .replace("{R✓", "boiled_rice-plate")
                    .replace("{Rx", "charcoal")
                    .replace("{@✓", "chicken-dish")
                    .replace(
                        "boiled_chicken_onion", "chicken-onion-dish"
                    )  ## TODO CHANGE FILE NAMES
                    .replace("{^✓", "fried_mushroom-plate")
                    .replace("{^%", "chopped_steak-plate")
                    # .replace("steak_burrito_dish", "steak_burrito")
                    # .replace("mushroom_burrito_dish", "mushroom_burrito")
                )
                fileName = f"{orientation}-{held_object}-{color}.png"

            data[y * width + x] = [fileName, 1]

        dynamic_objects = []
        # Place the dynamic objects
        for obj in self.env.state.objects.values():
            original_name = obj.name
            name = obj.name
            x, y = obj.position
            if name == "boiled_rice":
                if obj.is_warning:
                    name = "warning_pot"
                elif obj.is_burnt:
                    name = "burning_pot"
            elif name == "fried_mushroom" or name == "chopped_steak":
                if obj.is_warning:
                    name = "warning_grill"
                elif obj.is_burnt:
                    name = "burning_grill"
            elif name == "mushroom_burrito":
                name = "mushroom_burrito_dish"
            elif name == "steak_burrito":
                name = "steak_burrito_dish"
            fileName = f"{name}.png"
            # data[y * width + x] = self.url_for_static("static", filename=f"{fileName}")
            data[y * width + x] = [fileName, 0]
            if hasattr(obj, "is_burnt") and obj.is_burnt:
                dynamic_objects.append((obj.position, original_name, True))
            else:
                dynamic_objects.append((obj.position, original_name, False))

        # Draw some text on this...
        for obj in self.env.state.objects.values():
            x_pos, y_pos = obj.position
            grid = terrain_mtx
            text = ""

            if ((obj.name == "chopped_steak" and grid[y_pos][x_pos] == "G")
                or (obj.name == "fried_mushroom" and grid[y_pos][x_pos] == "G")
                or (obj.name == "boiled_rice" and grid[y_pos][x_pos] == "P")
            ): 
                if obj._warning_tick != -1:
                    text = {"burning": str((obj._warn_time - obj._warning_tick) / 10)}
                elif obj._cooking_tick != -1:
                    text = {"cooking": str((obj._cook_time - obj._cooking_tick) / 10)}
            if (
                (obj.name == "garnish" and grid[y_pos][x_pos] == "B")
                or (obj.name == "dirty_plate" and grid[y_pos][x_pos] == "W")
                or (obj.name == "chopped_meat" and grid[y_pos][x_pos] == "B")
                or (obj.name == "chopped_mushroom" and grid[y_pos][x_pos] == "B")
            ):
                if obj._cooking_tick != -1 and (
                    obj._cooking_tick <= obj.cook_time or True
                ):
                    text = {"cooking": str(obj._cook_time - obj._cooking_tick)}

            if grid[y_pos][x_pos] == "D":
                text = {"dishes": str(self.env.state.num_plates)}

            if text:
                text_data[y_pos * width + x_pos] = text

        self._last_text = text_data
        self._last_grid = data

        return data

    def is_running(self):
        return self._running or self._waiting

    def getGrid(self):
        if not self._initialized:
            return {}
        with self.lock:
            return self._last_grid


app = Flask(__name__, template_folder="html", static_folder="html/assets")
socketio = SocketIO(app, ping_timeout=60000, ping_interval=25000)
lock = threading.RLock()
# global gameapp

# DEFAULTS
# TODO: load our own config
games = {}
game_settings = {
    "num_players": 2,
    "layout": "open",
    "total_time": 240,  # 240 = 4 mins
    "order_list": [],
    "agent_types": [
        "HUMAN",
        "Dummy",
        "RLlib",
    ],  # Agents types available to select
    "agents": [
        "HUMAN",
        "RLlib",
    ],  # Agents that are actually playing in the game
    "agents_config": [
        "None",
        {
            "config_name": "eval_vaebr_open_bp",
            #"model_path": "/Users/benjili/Research/burrito_cooperation/overcooked/src/burrito_rl/policy_params/hallway_clusterbr-1/0000/checkpoint_000521/policies/polBR/policy_state.pkl"
            "model_path": "/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/FINAL_vaebr_gpuoptim_open_bp_norw-1/0000/checkpoint_001563/policies/polBR/policy_state.pkl"
        },
    ]
}
client_rooms = {}
clientIds = {}


def get_client_game():
    game_lock = request.cookies.get("game_lock")
    if not game_lock:
        return ""
    return game_lock


def get_client_id():
    """Generate or retrieve a unique client ID from cookies."""
    client_id = request.cookies.get("client_id")
    if not client_id:
        client_id = str(uuid.uuid4())  # Generate a new unique ID
    return client_id


def get_client_index(game_id):
    """Retrieve the client index from the client ID."""
    client_id = get_client_id()
    connected_clients = games[game_id]["gameapp"]
    return list(connected_clients.keys()).index(client_id)


def handle_game_joining(game_id, client_id):
    """Shared logic to check game state and manage client connections."""
    if game_id not in games:
        return "Game not found", 404

    with lock:
        game = games[game_id]
        if game["status"] == "completed":
            game["gameapp"]._running = False
            game["gameapp"]._waiting = False
            return "Game completed", 403

        players = game["connected_clients"]

        print(f"Players in {game_id}", players)

        if client_id not in players and len(players) >= game_settings["num_players"]:
            return "Game is full", 403

        game["command_queue"].put({"event": "player_update", "data": client_id})
        game["connected_clients"].update({client_id: datetime.datetime.now()})
        return None


@app.before_request
def limit_players():
    if request.endpoint != "game_index":
        return

    game_id = request.view_args.get("game_id")
    client_id = get_client_id()

    # Use shared function to handle game logic
    error = handle_game_joining(game_id, client_id)
    if error:
        if error == "Game completed":
            return make_response(render_template("gamecomplete.html"), 403)
        elif error == "Game is full":
            return make_response(render_template("fullgame.html"), 403)


@app.route("/sprite_sheets", methods=["GET"])
def fetch_sprite_sheets():
    # """Create a new game with a random ID."""
    asset_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "html", "assets"
    )
    all_chef_sprites_path = os.path.join(asset_path, "all_chef_sprites.json")
    sprite_overcooked_path = os.path.join(asset_path, "object_sprites.json")

    with open(all_chef_sprites_path, "r") as f:
        all_chef_sprites = json.load(f)

    with open(sprite_overcooked_path, "r") as f:
        sprite_overcooked = json.load(f)

    return jsonify(
        {
            "all_chef_sprites": all_chef_sprites,
            "object_sprites": sprite_overcooked,
        }
    )


@app.route("/create", methods=["POST"])
def create_game():
    # """Create a new game with a random ID."""
    game_id = generate_random_characters()
    makeGame(game_id=game_id, isPrivate=True)
    return redirect(url_for("game_index", game_id=game_id))


@app.route("/random", methods=["POST"])
def join_random():
    with lock:
        # Find a game with available slots
        for game_id, game_data in games.items():
            if (
                game_data["status"] != "completed"
                and game_data["isPrivate"] is False
                and len(game_data["connected_clients"]) < game_settings["num_players"]
            ):
                return redirect(url_for("game_index", game_id=game_id))
        # If no available game is found, create a new game
        game_id = generate_random_characters()
        makeGame(game_id=game_id, isPrivate=False)
        return redirect(url_for("game_index", game_id=game_id))


@app.route("/")
def index_list():
    """Display all active games and allow joining."""
    client_id = get_client_id()
    completed = {
        key: value for key, value in games.items() if value["status"] == "completed"
    }
    response = make_response(
        render_template("index.html", games=games, count=len(completed))
    )
    response.set_cookie("client_id", client_id)
    return response


@app.route("/admin")
def admin():
    # TODO put these layouts elsewhere.
    layouts = ["hallway_without_chicken", "hallway", "hallway_with_chicken_and_onion"]
    completed = {
        key: value for key, value in games.items() if value["status"] == "completed"
    }
    response = make_response(
        render_template(
            "admin.html",
            games=games,
            count=len(completed),
            layouts=layouts,
            settings=game_settings,
        )
    )
    return response


@app.route("/submit", methods=["POST"])
def submit():
    form_data = request.form

    sanitized_data = {key: escape(value) for key, value in form_data.items()}
    players = {
        key: value for key, value in sanitized_data.items() if key.startswith("player_")
    }
    print(players)

    admin_pin = sanitized_data.get("admin_pin")
    num_players = len(players)
    layout = sanitized_data.get("layout")

    # TODO: put these elsewhere
    available_layouts = [
        "hallway",
        "hallway_without_chicken",
        "hallway",
        "hallway_with_chicken_and_onion",
    ]

    # TODO: better validation here

    # Validate number of players
    num_players = int(num_players)
    if num_players < 1 or num_players > 5:
        return jsonify({"message": "Number of players must be between 1 and 5"})

    if layout not in available_layouts:
        return jsonify({"message": "Invalid layout"})

    if admin_pin == "1211":
        game_settings["num_players"] = num_players
        game_settings["layout"] = layout
        game_settings["agents"] = list(players.values())
        print(game_settings)
        return jsonify(
            {
                "message": "Success",
                "settings": {
                    "num_players": game_settings["num_players"],
                    "layout": game_settings["layout"],
                    "agents": game_settings["agents"],
                },
            }
        )

    else:
        return jsonify({"message": "Invalid pin"})


@app.route("/game/<game_id>")
def game_index(game_id):
    """Serve the main game page for a specific game."""
    if game_id not in games:
        return redirect(url_for("index_list"))

    if not games[game_id]["gameapp"].is_running():
        return render_template("gamecomplete.html")

    response = make_response(render_template("game.html", game_id=game_id))
    response.set_cookie("game_lock", f"{game_id.split('_')[0]}_")
    # Set the client ID as a cookie
    return response


# SocketIO event handlers
@socketio.on("connect")
def handle_connect():
    if not background_task_started.is_set():
        # Start the background task
        socketio.start_background_task(target=handle_all_game_events)
        background_task_started.set()  # Set the flag to indicate the task has started
        print("Background task started.")


@socketio.on("key_press")
def on_key_press(data):
    game_id = data["game_id"]
    key = data["key"]
    cid = get_client_id()
    games[game_id]["command_queue"].put({"event": "key_press", "key": key, "cid": cid})


@socketio.on("join_game")
def on_join(data):
    game_id = data["game_id"]
    client_id = data["client_id"]  # Assuming the client sends a unique identifier

    # Use shared function to handle game logic
    error = handle_game_joining(game_id, client_id)
    if error:
        # Send the appropriate error message back to the client
        emit("join_rejected", {"message": error})
        return

    # Join the client to the game room after checks pass
    join_room(game_id)
    sid = request.sid
    client_rooms[sid] = game_id
    emit("join_accepted", {"message": "Successfully joined the game."})

    # If the game is starting, notify all clients in the room
    emit("update_title", {"message": "Waiting for another player 👾"}, room=game_id)
    # print(f"Client {client_id} joined game {game_id}. All roosm: {rooms()}")
    # Sent the initial state
    emit("update_tiles", {"images": games[game_id]["gameapp"].getGrid()}, room=game_id)


@socketio.on("join_room")
def on_join(data):
    game_id = data["game_id"]
    client_id = data["client_id"]  # Assuming the client sends a unique identifier

    print(games)
    # Join the client to the game room after checks pass
    join_room(game_id)
    sid = request.sid
    client_rooms[sid] = game_id
    emit("update_tiles", {"images": games[game_id]["gameapp"].getGrid()}, room=game_id)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid  # Get the session ID of the disconnecting client
    # Check if the session ID exists in client_rooms to find the associated game ID
    game_id = client_rooms.get(sid, None)

    if game_id:
        del client_rooms[sid]  # Remove the session ID from the client_rooms dictionary
        client_id = get_client_id()  # Get the client ID
        client = games[game_id]["connected_clients"].get(client_id, None)
        print("time", (datetime.datetime.now() - client).total_seconds())
        if client and (datetime.datetime.now() - client).total_seconds() > 0.5:
            # Remove client_id from players in games[game_id]["gameapp"].players
            print("removing player : ", client_id)
            print(games[game_id])
            games[game_id]["command_queue"].put(
                {"event": "remove_player", "data": client_id}
            )
            leave_room(game_id)  # Leave the game room
            # TODO: remove client
            # TODO: remove all agents in the game room
            for i, player in games[game_id]["gameapp"]._players:
                if hasattr(player["agent"], "event_queue"):
                    player["agent"].event_queue.put(
                        {
                            "event": "remove_player",
                        }
                    )
            # games[game_id]["connected_clients"].pop(client_id)


def make_game(game_id, settings):
    # Merge settings with the result of initialize_config_from_args
    study_config = initialize_config_from_args(settings)
    return study_config


def handle_all_game_events():
    while True:
        cgame_data = []
        for game_id in games:
            queue = games[game_id]["event_queue"]
            while not queue.empty():
                event = queue.get()
                if event["event"] == "game_status":
                    games[game_id]["status"] = event["data"]
                elif event["event"] == "make_new_game":
                    cgame_data.append(
                        (
                            event["data"]["room"],
                            event["data"]["num"],
                            event["data"]["clients"],
                        )
                    )
                elif event["event"] == "deleted_player":
                    try:
                        games[game_id]["connected_clients"].pop(event["data"]["client"])
                    except KeyError:
                        pass
                else:
                    # Batch updates to minimize the number of network requests
                    if event["event"] == "batch_update":
                        for update_event, update_data in event["data"].items():
                            json_data = json.dumps(update_data)
                            compressed_data = zlib.compress(json_data.encode("utf-8"))
                            socketio.emit(
                                update_event, compressed_data, room=event["room"]
                            )
                    else:
                        socketio.emit(event["event"], event["data"], room=event["room"])

        for cg in cgame_data:
            makeGame(game_id=cg[0], clients=cg[2], isPrivate=True)
        cgame_data = []
        socketio.sleep(0.1)


# Global flag to track if the background task has started
background_task_started = threading.Event()


def generate_random_characters():
    length = 5

    # Define punctuation characters, excluding the underscore
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))

#TODO: We don't need the multiple game process here
# But can just keep this feature
def makeGame(game_id, clients=None, isPrivate=None):
    if isPrivate is None:
        isPrivate = False

    dgame_id = game_id if game_id else generate_random_characters()
    args = make_game(game_id, game_settings)

    games[dgame_id] = {
        "connected_clients": (
            {client: datetime.datetime.now() for client in clients} if clients else {}
        ),
        "status": "waiting",
        "start_time": time(),
        "gameapp": OvercookedWebApp(),
        "command_queue": Queue(),
        "event_queue": Queue(),
        "isPrivate": isPrivate,
    }

    process = multiprocessing.Process(
        target=games[game_id]["gameapp"].startup_game,
        args=(
            games[game_id]["command_queue"],
            games[game_id]["event_queue"],
            game_id,
            args,
        ),
    )
    games[game_id]["process"] = process
    process.start()
    # Wait for the process to start
    return dgame_id


def init_greedy_agent(world_mdp):
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
        "mdp": world_mdp,
    }
    # greedyAgent = SteakGreedyHumanModel(**human_model_config)
    return None #greedyAgent


def init_hierarchical_agent(world_mdp):
    # Intialize hierarchical agent
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "manager",
        "hierarchical_agent.yml",
    )
    config = read_yml_file(file_path)

    pickle_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "qd-simulacra",
        "burrito_2p",
        "archive",
        "archive_134.pkl",
    )

    pickle = read_pickle_file(pickle_file_path)
    _config = {
        "model": config["model"],
        "hidden_size": config["hidden_size"],
        "counter_locations": config["counter_locations"],
        "stochastic": config["stochastic"],
        "max_depth": config["max_depth"],
        "replan_on_stuck": config["replan_on_stuck"],
        "mdp": world_mdp,
    }

    # hierarchicalAgent = HierarchicalAgent(**_config)
    # hierarchicalAgent.from_numpy(pickle["solution"][-1, :])
    # random_index = random.randint(0, 236)
    # hierarchicalAgent.from_numpy(pickle["solution"][random_index, :])

    return None # hierarchicalAgent


def read_pickle_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def read_yml_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

import signal
def cleanup_and_exit(signum, frame):
    print("Shutting down...")
    for game_id, game_data in games.items():
        if "process" in game_data and game_data["process"].is_alive():
            game_data["process"].terminate()
            game_data["process"].join(timeout=2)
    sys.exit(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    socketio.run(app, host="0.0.0.0", port=27000, debug=True)
