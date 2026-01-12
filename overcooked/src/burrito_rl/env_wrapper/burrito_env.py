import numpy as np
from gym.spaces import Box, Discrete, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from ray.rllib.utils import override
from burrito.mdp.burrito_mdp import EVENT_TYPES, BurritoGridworld 
from burrito.planners.burrito_planner import BurritoPlanner, HighLevelActions
from burrito.mdp.burrito_env import BurritoEnv

class BurritoRLLibWrapper(MultiAgentEnv):
    """
    Class used to wrap the Burrito Environment in an Rllib compatible multi-agent environment
    """
    def __init__(self, config):
        super().__init__()
        # self.mdp = OvercookedGridworld(**config["mdp_config"])
        self.config = config
        self.layout = config.get("layout")
        self.env = BurritoEnv.from_mdp(
            mdp = BurritoGridworld.from_layout_name(layout_name = config.get("layout"), 
                                                       rew_shaping_params = config.get("rew_shaping_params")), 
            horizon=config.get("max_steps"))

        self.n_agents = self.env.mdp.num_players
        self.agents = [i for i in range(self.n_agents)]
        self._agent_ids = self.agents

        self.action_level = self.config["action_level"]
        self.n_actions =  len(list(HighLevelActions)) if self.action_level=='high' else len(Action.ALL_ACTIONS)
        if self.action_level == 'high':
            self.env.setup_planner(self.layout, player_types=['A' for _ in range(self.env.mdp.num_players)], restrict_capability=config.get("restrict_capability")) 
        
        # NOTE: set this True in the env to train with cluster_id
        self.cluster_br = config.get("cluster_br", False)
        self.num_clusters = config.get("num_clusters")
        self.cluster_id = 0 # set as default
        print("==============================")
        print("self.cluster_br", self.cluster_br)
        print("self.num_clusters", self.num_clusters)

        self.env.start_state_fn = self.env.mdp.get_random_start_position_fn() #self.env.mdp.get_standard_start_state #self.env.mdp.get_random_start_position_fn() #self.env.mdp.get_fc_start_position_fn(self.cluster_id) #self.env.mdp.get_random_start_position_fn() #self.env.mdp.get_standard_start_state #self.env.mdp.get_random_start_position_fn()#self.env.mdp.get_standard_start_state #self.env.mdp.get_random_start_position_fn() #get_standard_start_state  # TODO: implement a randomized start state get_random_start_position_fn()
        self.obs_encoder = self._setup_encoder() # TODO: Add this option in the config
        self.dummy_state = self.env.mdp.get_standard_start_state() # TODO: change to random start state
        # store all orders with a fixed sequence, used for observation encoding
        self.action_space = self._setup_action_space(self.agents)
        self.observation_space = self._setup_observation_space(self.agents)
        #print('OBS_SPACE',self.observation_space)
        self.annealing_reward_shaping = config.get("annealing_reward_shaping", True)
        self._initial_reward_shaping_factor = config.get("reward_shaping_factor")
        self.reward_shaping_factor = self._initial_reward_shaping_factor
        self.reward_shaping_horizon = config.get("reward_shaping_horizon")
        self.use_phi = config.get("use_phi")

        #### tricks to get evaulation work
        self.start_state_fn = self.env.start_state_fn
        self.horizon = self.env.horizon


    def copy(self):
        return BurritoRLLibWrapper(self.config)

    def _setup_action_space(self, agents):
        action_sp = {}
        for agent in agents:
            action_sp[agent] = Discrete(self.n_actions)
        return Dict(action_sp)

    def _setup_observation_space(self, agents):
        # TODO: include more dict fields into obs space for high-level action decision making
        dummy_obs = self.obs_encoder(self.dummy_state)
        self._postprocess_encoder(dummy_obs,self.dummy_state)
        obs_shape = dummy_obs[0].shape
        high = np.ones(obs_shape) * 65535
        low = np.zeros(obs_shape) * (-65535)
        observation_space = Box(
            np.int32(low),np.int32(high),dtype=np.int32
        )
        action_mask_space = Box(0.0,1.0,shape=(self.n_actions,), dtype=np.float32)
        action_stats_space = Box(0.0,1.0,shape=(self.n_actions,), dtype=np.float32)
        order_list_space = Box(0,1, shape=(8,), dtype=np.int8)
        prev_agent_actions_space = Box(0,1,shape=(self.n_actions,), dtype=np.float32)

        ob_space = {}
        if self.cluster_br:
            cluster_id_space = Box(0,1, shape=(self.num_clusters,), dtype=np.int8)
            for agent in agents:
                ob_space[agent] = Dict({
                    'image': observation_space, 
                    'action_mask': action_mask_space, 
                    'action_stats': action_stats_space, 
                    'order_list': order_list_space,
                    'cluster_id': cluster_id_space,
                    "prev_action": prev_agent_actions_space
                })
        else:
            for agent in agents:
                ob_space[agent] = Dict({
                    'image': observation_space, 
                    'action_mask': action_mask_space, 
                    'action_stats': action_stats_space, 
                    'order_list': order_list_space,
                    "prev_action": prev_agent_actions_space
                })
        return Dict(ob_space)

    def _setup_encoder(self):
        self.encoder_option = self.config.get("obs_encoder")
        if self.encoder_option is None or self.encoder_option == "lossless":
            return self.env.mdp.lossless_state_encoding
        elif self.encoder_option == "onehot":
            return self.env.mdp.onehot_state_encoding
        elif self.encoder_option == "nopos":
            return self.env.mdp.nopos_state_encoding
        else:
            raise NotImplementedError # return the specified encoder_option
        
    def _postprocess_encoder(self, img, burrito_state):
        # Generate an ego layer to map policy to the correct agent
        if self.encoder_option != 'nopos':

            #print("PLAYERS" , burrito_state.players)
            for agent in self.agents:
                #print("AGENT", agent)
                row, col = burrito_state.players[agent].position
                H, W, C = img[agent].shape
                ego_layer = np.zeros((H, W, 1), dtype=img[agent].dtype)
                ego_layer[col, row, :] = 1
                img[agent] = np.concatenate([ego_layer, img[agent]], axis=-1)

    
    def get_obs(self, burrito_state, info=None):
        # TODO: include more dict fields into obs space for high-level action decision making
        img = self.obs_encoder(burrito_state)
        self._postprocess_encoder(img, burrito_state)
        if info is None:
            info = self.env.env_info
        if self.action_level == "high":
            action_mask = self.get_highlevel_action_mask(burrito_state)
        else:
            action_mask = self.get_lowlevel_action_mask(burrito_state)
        
        # print("INFO",info)
        action_stats = self.get_action_stats(info)

        # Get order list
        order_encoding = np.zeros((4,2), dtype=np.int8)
        for i, order in enumerate(burrito_state.order_list):
            if "mushroom" in order[0]:
                order_encoding[i, 1] = 1
            elif "steak" in order[0]:
                order_encoding[i, 0] = 1
        order_encoding = order_encoding.flatten()

        prev_agent_actions = {}
        # Get previous action
        for agent_idx in range(self.env.mdp.num_players):
            stats = np.zeros(self.n_actions, dtype=np.float32)
            # default to stay (4) if first timestep
            if info is None or info.get("action_status", None) is None:
                stats[4] = 1
            else:
                prev_action = info["action_status"][agent_idx]["prev_action"]
                stats[prev_action] = 1
                #print("PREV AGENT, PREV ACTION", agent_idx, prev_action)
            prev_agent_actions[agent_idx] = stats

            if info is not None:
                if "human_action" in info.keys():
                    if info["human_action"] == -1:
                        prev_agent_actions[agent_idx] = np.zeros(self.n_actions,dtype=np.float32) #no op
                    else:
                        s = np.zeros(self.n_actions, dtype=np.float32)
                        s[info["human_action"]] = 1
                        if info["human_action"] == 25:
                            s[26] = 1
                        prev_agent_actions[agent_idx] = s

        #print("PREV AGENT ACTIONS", prev_agent_actions)
        if self.cluster_br:
            cluster_encoding = np.zeros((self.num_clusters,), dtype=np.int8)
            cluster_encoding[self.cluster_id] = 1
            cluster_encoding = cluster_encoding.flatten()
            return {i: {
                "image": img[i],
                "action_mask": action_mask[i],
                "action_stats": action_stats[i],
                "order_list": order_encoding.copy(),
                "cluster_id": cluster_encoding,
                "prev_action": prev_agent_actions[1-i]} # gets the partners
                for i in self.agents
                }
        else:
            return {i: {
                "image": img[i], 
                "action_mask": action_mask[i], 
                "action_stats": action_stats[i], 
                "order_list": order_encoding.copy(),
                "prev_action": prev_agent_actions[1-i]} # NOTE: changed from 1-i to i for 3 agent training
                for i in self.agents}

    def get_lowlevel_action_mask(self, burrito_state):
        action_mask = {}
        for agent_idx in self.agents:
            agent = burrito_state.players[agent_idx]
            mask = np.ones(self.n_actions, dtype=np.float32)
            # 1. If the agent is holding nothing, it cannot serve, so mask 'serve' action.
            agent_position = agent.position
            near_terrain = False
            for direction in Direction.ALL_DIRECTIONS:
                new_pos = Action.move_in_direction(agent_position, direction)
                if new_pos in self.env.mdp.terrain_pos_dict:  # 'X' typically represents a wall or impassable terrain
                    mask[direction] = 1  # Mask this direction as it leads to a wall
                    near_terrain = True
            if not near_terrain:
                mask[Action.ACTION_TO_INDEX[Action.INTERACT]] = 1
            action_mask[agent_idx] = mask
        return action_mask
    
    def get_highlevel_action_mask(self, burrito_state):
        action_mask = {}
        grid_distances = self.env.planner.grid_distances
        terrain_pos_dict = self.env.mdp.terrain_pos_dict
        terrain_mtx = self.env.planner.terrain_mtx

        for agent_idx in self.agents:
            mask = np.ones(self.n_actions, dtype=np.float32)
            for action_idx, action_init in enumerate(HighLevelActions):
                action_cls = action_init.action_class
                action_kwargs = action_init.action_kwargs
                action = action_cls(agent_idx, grid_distances, terrain_mtx, terrain_pos_dict, burrito_state, **action_kwargs)
                is_valid = action.valid_on_step_start(agent_idx, terrain_mtx, burrito_state)
                mask[action_idx] = int(is_valid)
            action_mask[agent_idx] = mask
        # print(action_mask)
        return action_mask

    def get_action_stats(self, info=None):
        """
        :return: dict of (n_actions) dimension arrays, one-hot encoding the ongoing action for each agent
        """
        stats_obs = {}
        for agent_idx in range(self.env.mdp.num_players):
            stats = np.zeros(self.n_actions, dtype=np.float32)
            if self.action_level=='low' or info is None or info.get("action_status", None) is None:
                stats = np.ones(self.n_actions, dtype=np.float32)
            else:
                prev_action = info["action_status"][agent_idx]["prev_action"]
                action_done = info["action_status"][agent_idx]["status"]
                if not action_done:
                    stats[prev_action] = 1
                else:
                    stats = np.ones(self.n_actions,dtype=np.float32)
            stats_obs[agent_idx] = stats
                
        return stats_obs
    
    @override(MultiAgentEnv)
    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy

        """
        action = [int(action_dict[key]) for key in action_dict] # convert from float to int
        #print(action)
        assert all(
            self.action_space[agent].contains(action[agent])
            for agent in action_dict
        ), "%r (%s) invalid" % (action, type(action))

        # take a step in the current base environment
        assert not self.env.is_done()
        next_state, timestep_sparse_reward, done, info = self.env.step(action)

        # # NOTE: debugging for eval pipeline
        # if self.env.state.timestep > 2380:
        #     print(self.env.state.timestep)

        sparse_reward = info["sparse_r_by_agent"]
        shaped_reward = info["shaped_r_by_agent"]
        # return (next_state, timestep_sparse_reward, done, env_info)
        if self.use_phi:
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = [potential for _ in self.agents]
        else:
            dense_reward = info["shaped_r_by_agent"] #[0]*self.n_agents #info["shaped_r_by_agent"]

        obs = self.get_obs(next_state, info)
        rewards = {i: sparse_reward[i] + self.reward_shaping_factor*shaped_reward[i] for i in self.agents}

        # NOTE: debugging stuff for eval pipeline
        # if rewards != {i: 0.0 for i in self.agents}:
        #     print(self.env.state.timestep, "REWARD STEP")
        #     print("ACTIONS", action)
        #     print("ACTION STATUS", info["action_status"])
        #     print("SPARSE REWARD", sparse_reward, "SHAPED REWARD", shaped_reward, "COMBINED REWARD", rewards)
        #     print(info["sparse_r_by_agent"], info["shaped_r_by_agent"])

        dones = {"__all__": done}

        if done:
            self._add_episode_info(info)
        
        infos = {i: info for i in self.agents}
        return obs, rewards, dones, infos
    

    def _add_episode_info(self, env_info):
        env_info["episode"]["reward_shaping_factor"] = self.reward_shaping_factor

    @override(MultiAgentEnv)
    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.env.reset(regen_mdp) # state initialized from class OvercookedState

        # self.curr_agents = self._populate_agents()
        #print("RESETTING ENVIRONMENT")

        #print(self.env.state.players,"PLAYERS during reset")
        obs = self.get_obs(self.env.state)
        for agent_id, agent_obs in obs.items():
            space = self.observation_space[agent_id]
            if not space.contains(agent_obs):
                print(f"\n❌ Agent {agent_id} OBS does NOT match its space")
                for k, v in agent_obs.items():
                    subspace = space.spaces[k]
                    ok = subspace.contains(v)
                    print(f"  • Field `{k}`: contains? {ok}")
                    if not ok:
                        print(f"      – dtype: {v.dtype}, shape: {v.shape}")
                        print(f"      – expected: {subspace}")
        events_dict = { k : [ [] for _ in range(self.env.mdp.num_players) ] for k in EVENT_TYPES }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0.] * self.env.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0.] * self.env.mdp.num_players)
        }
        self.env.game_stats = {**events_dict, **rewards_dict}
        return obs

    def anneal_reward_shaping_factor(self, timesteps = 0):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(
            self._initial_reward_shaping_factor,
            timesteps,
            self.reward_shaping_horizon,
        )
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def set_action_dist(self, action_dist):
        self.last_action_dist = action_dist
