import pygame
from pygame.locals import *
import copy
from overcooked_ai_py.utils import generate_temporary_file_path, load_from_json
from overcooked_ai_py.static import TESTING_DATA_DIR
import os
import pickle
import shutil
import json
import cv2
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import argparse
import numpy as np
from overcooked_ai_py.agents.agent import (
    AgentPair,
    GreedyHumanModel,
    RandomAgent,
)
from mdp.steakhouse_mdp import SteakhouseGridworld, dishname2ingradient, ingradient2dishname
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedEnvPettingZoo
from mdp.steakhouse_env import SteakhouseEnv
import overcooked_ai_py.agents.agent as agent 
from agents.steak_agent import SteakLimitVisionHumanModel
from  planners.steak_planner import SteakMediumLevelActionManager
import overcooked_ai_py.planning.planners as planners
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file
from visualization.state_visualizer import SteakhouseStateVisualizer
from time import time

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 1000

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False
MAX_STEPS = 20000
USER_STUDY_LOG = os.path.join(os.getcwd(), 'user_study/log')
TIMER, t = pygame.USEREVENT+1, 1000
VIDEO_FPS = 10
NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

class OvercookedPygame():
    """     
    Class to run the game in Pygame.
    Args:
      - max_time: Number of seconds the game should last
    """

    def __init__(
            self,
            env,
            agent1,
            agent2,
            logger,
            gameTime = 30,
    ):
        self._running = True
        self.logger = logger
        self.env = env
        self.score = 0 
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.agent1 = agent1
        self.agent2 = agent2
        self.init_time = time()
        self.player_1_action = Action.STAY
        self.player_2_action = Action.STAY

    def on_init(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            (self.env.mdp.width * 30, self.env.mdp.height * 30 + 140), pygame.RESIZABLE)
        print(pygame.display.get_surface().get_size())
        # Initialize agents
        # self.agent1.set_agent_index(self.agent_idx)
        self.agent1.set_mdp(self.env.mdp)
        # self.agent2.set_agent_index(self.agent_idx+1)
        self.agent2.set_mdp(self.env.mdp)
        self.start_time = time()
        pygame.time.set_timer(TIMER, t)

        ds = load_from_json(os.path.join(
            "data", "config", "kitchen_config.json"))
        test_dict = copy.deepcopy(ds)
        print(test_dict["config"])
        self.state_visualizer = SteakhouseStateVisualizer(
            **test_dict["config"])
        self._running = True

        self.logger.env = self.env

    def on_event(self, event):
        done = False
        player_1_action = Action.STAY
        player_2_action = Action.STAY
        # Players stay in place if no keypress are detected
        if event.type == TIMER:
            self.env.mdp.step_environment_effects(self.env.state)

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']

            if pressed_key == pygame.K_UP:
                player_1_action = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                player_1_action = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                player_1_action = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                player_1_action = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                player_1_action = Action.INTERACT
            
            elif pressed_key == pygame.K_w:
                player_2_action = Direction.NORTH
            elif pressed_key == pygame.K_d:
                player_2_action = Direction.EAST
            elif pressed_key == pygame.K_s:
                player_2_action = Direction.SOUTH
            elif pressed_key == pygame.K_a:
                player_2_action = Direction.WEST
            elif pressed_key == pygame.K_f:
                player_2_action = Action.INTERACT

            # check if action is valid
            if player_1_action in Action.ALL_ACTIONS and player_2_action in Action.ALL_ACTIONS:
                self.player_1_action = player_1_action
                self.player_2_action = player_2_action
                print(f"player2:{self.agent2.get_knowledge_base(self.env.state)} ")
        if event.type == pygame.QUIT:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

    def on_loop(self):
        self.logger.env = self.env
        time_now_in_milisecond = round(time() * 1000 - self.init_time *1000)
        # self.env.state.timestep = float('%.1f'%(time_now_in_milisecond/1000))

        ## change onloop to update game at 10fps,60 fps, apply joint action, update logger
        ## step environment every 0.1s/100ms,
        # 1 second = 1000ms
        if(time_now_in_milisecond%10 == 0):
            # print(time_now_in_milisecond)
            joint_action = (self.player_1_action, self.player_2_action)
            done = self._human_step_env(self.player_1_action, self.player_2_action)
            # log user behavior to json
            log = {"state":self.env.state.to_dict(),"joint_action":joint_action ,"score": self.score}
            self.logger.episode.append(log)

            # reinitialize action
            self.player_1_action = Action.STAY
            self.player_2_action = Action.STAY

            if self.logger.video_record:
                frame_name = self.logger.img_name(time_now_in_milisecond/1000)
                pygame.image.save(self.screen, frame_name)

            if done:
                self._running = False



    def on_render(self):
        
        kitchen = self.state_visualizer.render_state(
            self.env.state, self.env.mdp.terrain_mtx, hud_data=self.state_visualizer.default_hud_data(
                self.env.state,time_left=round( max(self.max_time - (time() - self.start_time), 0))
            )
        )
        #on top of the kitchen, render fog
        for agent in [self.agent1, self.agent2]:
            self.state_visualizer.render_fog(kitchen,self.env, agent)
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

    def _human_step_env(self, human1_action, human2_action):
        joint_action = (human1_action, human2_action)
        prev_state = self.env.state
        self.state, info = self.env.mdp.get_state_transition(
                prev_state, joint_action
                )

        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward
        
        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action, joint_agent_action_info =[{"1"},{"2"}])
        return done
    
    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
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
            os.getcwd(), "data", "layout", layout_file_name)
        path_to = os.path.join("..", "overcooked_ai", "src",
                               "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)

        if args.order_list:
            self.order_list = args.order_list
            start_all_orders = [dishname2ingradient(
                dish) for dish in args.order_list]
            self.start_all_orders = start_all_orders
            self.world_mdp = SteakhouseGridworld.from_layout_name(
                self.layout_name, start_all_orders=self.start_all_orders, order_list=self.order_list)
        else:
            self.world_mdp = SteakhouseGridworld.from_layout_name(
                self.layout_name)

        self.base_env = SteakhouseEnv.from_mdp(
            self.world_mdp, horizon=args.total_time)


class Logger:
    def __init__(self, config, filename, agent1=None, agent2=None, video_record=False):
        self.participant_id = config.participant_id
        self.json_filename = filename+'.json'
        self.filename = filename
        self.video_record = config.record_video

        # create log folder
        self.log_folder = os.path.join(
            USER_STUDY_LOG, str(self.participant_id))
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.makedirs(self.log_folder)
        self.img_dir = os.path.join(self.log_folder, 'img')
        os.makedirs(self.img_dir)
        self.img_name = lambda timestep: f"{self.img_dir}/{int(timestep*10):05d}.png"

        # game info
        self.env = config.base_env
        self.layout_name = config.layout_name
        self.agent1 = agent1
        self.agent2 = agent2
        self.horizon = config.base_env.horizon
        self.episode =[]

    def save_log_as_pickle(self):
        with open(os.path.join(self.log_folder, self.json_filename), 'w') as file:
            json.dump({"layout_name": self.layout_name,
                       "participant_id": self.participant_id,
                       "horizon": self.horizon,
                    #    "time_left": round( max(self.env.horizon - (time() - self.env.), 0)),
                    #    "time_elapsed": round(time() - self.start_time),
                      "episode": self.episode}, file)
        print(f"Pickle log saved to {self.json_filename}")
    
    def create_video(self):
        images = [img for img in os.listdir(self.img_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.img_dir, images[0]))
        height, width, layers = frame.shape
        video_name = '{}{}.mp4'.format(self.log_folder+'/', self.filename)

        video = cv2.VideoWriter(video_name, 0, VIDEO_FPS, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(self.img_dir, image)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.img_dir)
def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description='Initialize configurations for a human study.')

    ### Args for the game setup ###
    parser.add_argument('--layout', type=str, default='steak',
                        help='List of tasks to be performed in the study')
    parser.add_argument('--order_list', type=str, nargs='+',
                        help='List of dishes (steak_dish, chicken_dish, steak_onion_dish, boilded_chicken_onion_dish) to serve')
    parser.add_argument('--total_time', type=int, default=MAX_STEPS,
                        help='Total time to given to complete the game')

    # The following game config options are still undergoing construction
    # parser.add_argument('--served_in_order', type=bool, help='Complete the order list in order')
    # parser.add_argument('--single_player', type=bool, help='Single player mode: one human controlled agent collaborating with a modeled greedy agent')

    ### Args for the study ###
    parser.add_argument('--participant_id', type=int,
                        help='ID of participants in the study', default=0)
    parser.add_argument('--log_file_name', type=str,
                        default='', help='Log file name')
    parser.add_argument('--record_video', dest='record_video',
                        action='store_true', help='Record video during replay')
    parser.add_argument('--no-record_video', dest='record_video',
                        action='store_false', help='Do not record video during replay')

    args = parser.parse_args()

    if args.log_file_name == '':
        args.log_file_name = '-'.join([str(args.participant_id), args.layout])

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
    # switch to limit vision
    VISION_LIMIT = True
    #degree of vision
    VISION_BOUND = 120
    # whether other agent knows visionlimitagent is vision limited
    VISION_LIMIT_AWARE = True
    EXPLORE = False
    # medium level A* search depth,
    SEARCH_DEPTH = 5
    # low level exploaration depth, 
    KB_SEARCH_DEPTH = 1
    KB_UPDATE_DELAY = 3

    # Initialize two human agent
    mlam = SteakMediumLevelActionManager(study_config.world_mdp,NO_COUNTERS_PARAMS)


    agent1 = SteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, debug=False)
    agent2 =SteakLimitVisionHumanModel(mlam, study_config.base_env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, debug=False)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)

    agent1.init_knowledge_base(study_config.base_env.state)
    agent2.init_knowledge_base(study_config.base_env.state)

    # Initialize logging
    logger = Logger(study_config, study_config.log_file_name,
                    agent1=agent1, agent2=agent2)
    gametime = 300
    gameapp = OvercookedPygame(study_config.base_env, agent1, agent2, logger,gameTime=gametime)
    gameapp.on_execute()
