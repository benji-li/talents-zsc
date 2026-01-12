import numpy as np
from overcooked_ai_py.mdp.overcooked_env import MAX_HORIZON, OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS

from burrito.mdp.burrito_mdp import EVENT_TYPES, BurritoGridworld 
from burrito.planners.burrito_planner import BurritoPlanner, HighLevelActions

import time
import tqdm
import traceback
from overcooked_ai_py.utils import append_dictionaries, mean_and_std_err
from overcooked_ai_py.mdp.overcooked_trajectory import DEFAULT_TRAJ_KEYS
from overcooked_ai_py.mdp.actions import Action

class BurritoEnv(OvercookedEnv):
    def __init__(
        self,
        mdp_generator_fn,
        start_state_fn=None,
        horizon=MAX_HORIZON, # del?
        mlam_params=NO_COUNTERS_PARAMS, # del?
        info_level=0,
        num_mdp=1,
        initial_info={},
        planner = None
    ):
        self.planner = planner 
        self.num_mdp = num_mdp
        self.variable_mdp = num_mdp > 1
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self._mlam = None
        self._mp = None
        self.mlam_params = mlam_params
        self.start_state_fn = start_state_fn
        self.info_level = info_level
        self.reset(outside_info=initial_info)
        if self.horizon >= MAX_HORIZON and self.info_level > 0:
            print (
                "Environment has (near-)infinite horizon and no terminal states. \
                Reduce info level of OvercookedEnv to not see this message."
            )

        self.env_info = None
        self.human_study = False

    @property
    def mp(self):
        return None

    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=1,
        num_mdp=None,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, BurritoGridworld)
        if num_mdp is not None:
            assert num_mdp == 1
        mdp_generator_fn = lambda _ignored: mdp
        return BurritoEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
        )

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return BurritoEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp,
            planner = self.planner
        )

    # NOTE: Maps human action into HL action for the user study
    def map_event_to_planner_index(self, event_name):
        # Create a mapping from event types to planner indices based on the provided table
        event_to_index = {
            # GO_TO_PAN_AND_FRY_INGREDIENT (0)
            "chopped_steak_cooking": 0,
            "fried_mushroom_cooking": 0,
            "steak_cooking": 0,
            
            # GO_TO_POT_AND_BOIL_RICE (1)
            "potting_rice": 1,
            
            # GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT (2)
            "meat_chopping": 2,
            "onion_chopping": 2,
            
            # GO_TO_PUT_OUT_FIRE (3)
            "extinguish_fire": 3,
            
            # STAY (4) - No specific events
            
            # GO_TO_SINK_TO_WASH_PLATE (5)
            "plate_rinsing": 5,
            
            # GO_TO_TRASH_AND_THROW (6)
            "object_in_trash": 6,
            
            # GO_TO_SERVE_DISH (7)
            "dish_delivery": 7,
            
            # GRAB_MEAT (8)
            "meat_pickup": 8,
            
            # GRAB_MUSHROOM (9)
            "mushroom_pickup": 9,
            
            # GRAB_TORTILLA (10)
            "tortilla_pickup": 10,
            
            # GRAB_RICE (11)
            "rice_pickup": 11,
            
            # GRAB_DIRTY_PLATE (12)
            "dirty_plate_pickup": 12,
            
            # GRAB_CLEAN_PLATE (13)
            "clean_plate_pickup": 13,
            
            # GRAB_FIRE_EXT (14)
            "fire_ext_pickup": 14,
            
            # GRAB_CHOPPED_MUSHROOM (15)
            "chopped_mushroom_pickup": 15,
            
            # GRAB_CHOPPED_MEAT (16)
            "chopped_meat_pickup": 16,
            
            # GRAB_ONE_INGREDIENT_PLATE (17)
            "boiled_rice-plate_pickup": 17,
            "tortilla-plate_pickup": 17,
            "chopped_steak-plate_pickup": 17,
            "fried_mushroom-plate_pickup": 17,
            
            # GRAB_TWO_INTREDIENT_PLATE (18)
            "chopped_steak-boiled_rice-plate_pickup": 18,
            "fried_mushroom-boiled_rice-plate_pickup": 18,
            "boiled_rice-tortilla-plate_pickup": 18,
            "fried_mushroom-tortilla-plate_pickup": 18,
            "chopped_steak-tortilla-plate_pickup": 18,
            
            # GRAB_MURHSOOM_BURRITO_PLATE (19)
            "mushroom_burrito_pickup": 19,
            
            # GRAB_STEAK_BURRITO_PLATE (20)
            "steak_burrito_pickup": 20,
            
            # GET_MURHSOOM_FROM_GRILLER (21)
            "fried_mushroom-plate_grill_pickup": 21,
            "fried_mushroom-tortilla-plate_grill_pickup": 21,
            "fried_mushroom-boiled_rice-plate_grill_pickup": 21,
            "mushroom_burrito_grill_pickup": 21,
            
            # GET_STEAK_FROM_GRILLER (22)
            "chopped_steak-plate_grill_pickup": 22,
            "chopped_steak-tortilla-plate_grill_pickup": 22,
            "chopped_steak-boiled_rice-plate_grill_pickup": 22,
            "steak_burrito_grill_pickup": 22,
            
            # GET_BOILED_RICE (23)
            "boiled_rice-plate_pot_pickup": 23,
            "chopped_steak-boiled_rice-plate_pot_pickup": 23,
            "fried_mushroom-boiled_rice-plate_pot_pickup": 23,
            "boiled_rice-tortilla-plate_pot_pickup": 23,
            "mushroom_burrito_pot_pickup": 23,
            "steak_burrito_pot_pickup": 23,
            
            # GRAB_CHARCOAL (24)
            "charcoal_pickup": 24,
            
            # PASS_OBJECT/PUT_DOWN_OBJECT (25/26)
            "mushroom_drop": 25,  # Could also be 26
            "chopped_mushroom_drop": 25,
            "rice_drop": 25,
            "boiled_rice-plate_drop": 25,
            "tortilla_drop": 25,
            "tortilla-plate_drop": 25,
            "fire_ext_drop": 25,
            "charcoal_drop": 25,
            "chopped_steak-plate_drop": 25,
            "meat_drop": 25,
            "chopped_meat_drop": 25,
            "chopped_steak-boiled_rice-plate_drop": 25,
            "fried_mushroom-boiled_rice-plate_drop": 25,
            "boiled_rice-tortilla-plate_drop": 25,
            "fried_mushroom-plate_drop": 25,
            "fried_mushroom-tortilla-plate_drop": 25,
            "chopped_steak-tortilla-plate_drop": 25,
            "chopped_steak_drop": 25,
            "dish_drop": 25,
            "steak_drop": 25,
            "mushroom_burrito_drop": 25,
            "steak_burrito_drop": 25,
            "dirty_plate_drop": 25,
        }
        
        return event_to_index.get(event_name, -1)  # Return -1 if event not found

    ###################
    # BASIC ENV LOGIC #
    ###################
    def setup_planner(self, layout, player_types, restrict_capability=True):
        """
        Set up the burrito planner of high-level actions
        """
        dummy_env_state = self.state
        self.planner = BurritoPlanner(layout, dummy_env_state, player_types, restrict_capability)
    
    def step(self, joint_action, joint_agent_action_info=None, display_phi=False):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        try:
            if self.is_done():
                return self.state, 0, True, {}

            # assert not self.is_done()
            if joint_agent_action_info is None:
                joint_agent_action_info = [{} for _ in range(self.mdp.num_players)]
            if self.planner:
                joint_action_array, action_done, solution_found = self.planner.define_plan(self.state, joint_action)
            else:
                joint_action_array = [Action.INDEX_TO_ACTION[action] for action in joint_action]

            next_state, mdp_infos = self.mdp.get_state_transition(
                self.state, joint_action_array, display_phi
            ) # there's a self.mp arg after display_phi for motion planning, but we can omit for now

            # Update game_stats
            self._update_game_stats(mdp_infos)
            #print(joint_action,"joint action")

            # Update state and done
            self.state = next_state
            done = self.is_done()
            env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)
            if self.planner:
                env_info['action_status'] = action_done
                env_info['solution_found'] = solution_found
            #print("ENV STEP")

            # NOTE: we need to update the action mapping here for when playing w human
            if self.human_study:
                env_info['human_action'] = -1  # Default no-op value, assuming human is always index 0
                for event, status in mdp_infos["event_infos"].items():
                    if status[0]:  
                        #print(event,"EVENT")
                        planner_index = self.map_event_to_planner_index(event)
                        if planner_index != -1:
                            env_info['human_action'] = planner_index
                            #print('hl act', planner_index)
                            break  # Assuming only one action can be active at a time

            if done:
                self._add_episode_info(env_info)

            timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
            self.env_info = env_info
            return (next_state, timestep_sparse_reward, done, env_info)
        except Exception as e:
            print(f"Exception in envrionment step: {e}")
            traceback.print_exc()

    def lossless_state_encoding_mdp(self, state):
        """
        Wrapper of the mdp's lossless_encoding
        """
        return self.mdp.lossless_state_encoding(state, self.horizon)


    def reset(self, regen_mdp=True, outside_info={}):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0.0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0.0] * self.mdp.num_players),
        }
        self.game_stats = {**events_dict, **rewards_dict}
        self.env_info = None
        if self.planner:
            self.planner.reset()

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)

    def potential(self, mlam, state=None, gamma=0.99):
        """
        Return the potential of the environment's current state, if no state is provided
        Otherwise return the potential of `state`
        args:
            mlam (MediumLevelActionManager): the mlam of self.mdp
            state (OvercookedState): the current state we are evaluating the potential on
            gamma (float): discount rate
        """
        state = state if state else self.state
        return self.mdp.potential_function(state, mp=self.mp, gamma=gamma)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {
            "agent_infos": [
                joint_agent_action_info[agent_idx]
                for agent_idx in range(self.mdp.num_players)
            ]
        }
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["phi_s"] = (
            mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        )
        env_info["phi_s_prime"] = (
            mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        )
        return env_info

    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            # "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(
                self.game_stats["cumulative_sparse_rewards_by_agent"]
            ),
            "ep_shaped_r": sum(
                self.game_stats["cumulative_shaped_rewards_by_agent"]
            ),
            "ep_sparse_r_by_agent": self.game_stats[
                "cumulative_sparse_rewards_by_agent"
            ],
            "ep_shaped_r_by_agent": self.game_stats[
                "cumulative_shaped_rewards_by_agent"
            ],
            "ep_length": self.state.timestep,
            "ep_delivery": sum([len(self.game_stats['dish_delivery'][agent]) for agent in range(self.mdp.num_players)]),
            "ep_correct_delivery": sum([len(self.game_stats['correct_dish_delivery'][agent]) for agent in range(self.mdp.num_players)]),
            "ep_in_order_delivery": sum([len(self.game_stats['in_order_dish_delivery'][agent]) for agent in range(self.mdp.num_players)])
        }
        return env_info

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(
            infos["sparse_reward_by_agent"]
        )
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(
            infos["shaped_reward_by_agent"]
        )

        for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(
                        self.state.timestep
                    )

    def run_agents(
            self,
            agent_pair,
            include_final_state=False,
            display=False,
            dir=None,
            display_phi=False,
            display_until=np.Inf,
        ):
            """
            Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
            """
            assert (
                self.state.timestep == 0
            ), "Did not reset environment before running agents"
            trajectory = []
            done = False
            # default is to not print to file
            fname = None

#            if dir != None:
#                import os
#                os.makedirs(dir, exist_ok=True)
#                fname = dir + "/roll_out_" + str(time.time()) + ".txt"
#                with open(fname, "w+", encoding='utf-8') as f:
#                    print(self, file=f)
            info = None
            while not done:
                s_t = self.state

                # Getting actions and action infos (optional) for both agents
                joint_action_and_infos = agent_pair.joint_action((s_t, info))
                a_t, a_info_t = zip(*joint_action_and_infos)
                assert all(type(a_info) is dict for a_info in a_info_t)

                s_tp1, r_t, done, info = self.step(a_t, a_info_t, display_phi)
                trajectory.append((s_t, a_t, r_t, done, info))

                if display and self.state.timestep < display_until:
                    self.print_state_transition(a_t, r_t, info, fname, display_phi)

            assert len(trajectory) == self.state.timestep, "{} vs {}".format(
                len(trajectory), self.state.timestep
            )

            # Add final state
            if include_final_state:
                trajectory.append((s_tp1, (None, None), 0, True, None))

            total_sparse = sum(
                self.game_stats["cumulative_sparse_rewards_by_agent"]
            )
            total_shaped = sum(
                self.game_stats["cumulative_shaped_rewards_by_agent"]
            )
            return (
                np.array(trajectory, dtype=object),
                self.state.timestep,
                total_sparse,
                total_shaped,
            )



    # NOTE: porting this over because the AgentEvaluator stuff at the end is incompatible with burrito
    # in the future might be good to push the change to overcooked_ai, since they were looking to do that anyway
    def get_rollouts(
        self,
        agent_pair,
        num_games,
        display=False,
        dir=None,
        final_state=False,
        display_phi=False,
        display_until=np.Inf,
        metadata_fn=None,
        metadata_info_fn=None,
        info=True,
    ):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: this is the standard trajectories format used throughout the codebase
        """
        trajectories = {k: [] for k in DEFAULT_TRAJ_KEYS}
        metadata_fn = (lambda x: {}) if metadata_fn is None else metadata_fn
        metadata_info_fn = (
            (lambda x: "") if metadata_info_fn is None else metadata_info_fn
        )
        range_iterator = (
            tqdm.trange(num_games, desc="", leave=True)
            if info
            else range(num_games)
        )
        for i in range_iterator:
            agent_pair.set_mdp(self.mdp)

            rollout_info = self.run_agents(
                agent_pair,
                display=display,
                dir=dir,
                include_final_state=final_state,
                display_phi=display_phi,
                display_until=display_until,
            )
            (
                trajectory,
                time_taken,
                tot_rews_sparse,
                _tot_rews_shaped,
            ) = rollout_info
            obs, actions, rews, dones, infos = (
                trajectory.T[0],
                trajectory.T[1],
                trajectory.T[2],
                trajectory.T[3],
                trajectory.T[4],
            )
            trajectories["ep_states"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))

            # we do not need to regenerate MDP if we are trying to generate a series of rollouts using the same MDP
            # Basically, the FALSE here means that we are using the same layout and starting positions
            # (if regen_mdp == True, resetting will call mdp_gen_fn to generate another layout & starting position)
            self.reset(regen_mdp=False)
            agent_pair.reset()

            if info:
                mu, se = mean_and_std_err(trajectories["ep_returns"])
                description = "Avg rew: {:.2f} (std: {:.2f}, se: {:.2f}); avg len: {:.2f}; ".format(
                    mu,
                    np.std(trajectories["ep_returns"]),
                    se,
                    np.mean(trajectories["ep_lengths"]),
                )
                description += metadata_info_fn(trajectories["metadatas"])
                range_iterator.set_description(description)
                range_iterator.refresh()

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = append_dictionaries(
            trajectories["metadatas"]
        )
        return trajectories


    ##################
    #   RENDERING    #
    ##################

    def render(self, mode="human"):
        time_step_left = self.horizon - self.t if self.horizon != MAX_HORIZON else None
        time_passed = (
            time.time() - self.start_time if self.start_time is not None else 0
        )
        self.mdp.render(
            self.state,
            mode,
            time_step_left=time_step_left,
            time_passed=time_passed,
        )


