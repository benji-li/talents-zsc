from overcooked_ai_py.agents.agent import Agent, AgentPair, AgentGroup
import numpy as np
import random
from overcooked_ai_py.mdp.actions import Action
import multiprocessing as mp
from burrito.planners.burrito_planner import HighLevelActions

import time
import random

import traceback

class DummyAgent(Agent):
    """
    Class for testing the implemented evaluation agent-human game pipeline
    """
    def __init__(self, agent_index, mdp, **kwargs):
        self.config = kwargs.get("config", None)
        self.agent_index = agent_index
        self.mdp = mdp
        self.featurize = None
        self.generated_action = None
        self.setup_agent()

        self.action_queue = [[],
                             [4,4,4,4,4,4,4,4,8,9,9,25,9,25,12,5,5,5,5,13,25,12,5,5, 8,13],  #[4 for _ in range(500)]+ [12, 21], # [12, 8, 15, 0, 11, 13] + [4 for _ in range(100)] + [21],
                             [4,4,4,9,4,4,4,4,8,4,4,4,9,4,4,4]] #4,4,4,9,26,4,4,4,4,4,4,4,9,26,8,25]]
                            #  [11,6,9,2,0,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,11,22,9,22,10,23,21,6]]
                            #  [11, 1, 34, 18, 0, 37, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4,\
                            #   16, 32, 43, 17, 6, 41, 15, 5, 40, 12, 1, 35, 19, 3, 38, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4,\
                            #   16, 31, 43, 
                            #   ],
                             # grab meat, go to chop1, go to serve2, go to chop1, chop meat, grab chopped meat
                             # go to pan2, go to pot1, go to pan2, fry meat, ..stay...
                             # grab clean plate, get fried steak, go to trash, 
                             # throw, grab fire ext, go to trash, throw, grab mushroom, go to chop1, go to serve2, 
                             # go to chop1, chop mushroom, go to sink, grab chopped mushroom, go to pan1, go to trash, throw
                            #  [14, 9, 36, 15, 5, 40, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   21, 33, 13, 7, 42, \
                            #   14, 9, 36, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                            #   20, 33, 13, 8, 42]
                        #    [8, 2, 0, 12, 5, 11, 1, 13, 10, \
                        #     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                        #    19, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 7],
                        #    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, \
                        #     9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 15, 0, 11, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                        #     12, 5, 13, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,\
                        #    19, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 7 ]
#                             # grab rice, go to serve1, go to pot1, boil rice, grab dirty plate, go to trash, go to sink, wash plate,
#                             # go to serve1, go to pot2, ...stay..., get boiled rice, go to serve1,
#                             # serve dish, grab mushroom, go to chop2, chop mushroom, grab chopped mushroom, go to pot1
                            # ]

        
    def setup_agent1(self):
        return
        assert self.config
        from ray.rllib.algorithms.ppo import PPOConfig
        from burrito_rl.mdp.burrito_env import BurritoRLLibWrapper
        rllib_config = PPOConfig(self.config).environment(env=self.config.get("env"), env_config=self.config.get("env_config")).multi_agent(**self.config.get("multiagent"))
        dummy_rllib_env = BurritoRLLibWrapper(rllib_config.env_config)
        self.featurize = dummy_rllib_env.get_obs
    def setup_agent(self):
    # Remove the early return to allow proper setup if needed
        if self.config:
            try:
                from burrito_rl.env_wrapper.burrito_env import BurritoRLLibWrapper
                dummy_rllib_env = BurritoRLLibWrapper(self.config.get("env_config", {}))
                self.featurize = dummy_rllib_env.get_obs
                self.n_actions = dummy_rllib_env.n_actions
                print(f"DummyAgent setup complete. Action space size: {self.n_actions}")
            except Exception as e:
                print(f"DummyAgent setup failed: {e}")
                # Continue without featurize function
 
    
    def action(self, stats):
        """
        Return a random action index from 0~5
        Sleep for some time
        """
        try:
            # if self.agent_index == 2:
            #     agent_action = 4
            #     return agent_action, {}
            
            obs, info = stats
            action_status = info.get("action_status", None)
            if action_status is not None:
                action_done = action_status[self.agent_index]["status"]
                action = action_status[self.agent_index]["prev_action"]
            else:
                action_done = True
            
            if not action_done:
                return action
            else:
                action = self.action_queue[self.agent_index][0] if len(self.action_queue[self.agent_index]) > 0 else 4
                if not len(self.action_queue[self.agent_index]) == 0:
                    self.action_queue[self.agent_index].pop(0)
                # if self.agent_index == 1:
                #     action = np.random.choice([1, 3, 7, 9]) # chop1, pan1, serve1, pot1
                # else:
                #     action = np.random.choice([0, 2, 8, 10]) # chop2, pan2, serve2, pot2
                # if self.agent_index == 2:
                #     action = 4
                agent_action = action
                # agent_action = np.random.randint(24)

                return agent_action, {}
        except Exception as e:
            print(f"An error occurred --- : {e}")
            traceback.print_exc()
            return 4, {}


from collections import deque
class RLlibAgent(Agent):
    """
    Class for wrapping a trained Policy object into an Overcooked compatible Agent
    Currently support RLlib policy objects
    """

    def __init__(self, agent_index, mdp=None, **kwargs):
        self.agent_index = agent_index
        self.mdp = mdp
        self.config = kwargs.get("config", None)
        self.model_path = kwargs.get("model_path", None)
        self.policy = kwargs.get("policy", None)
        self.featurize = kwargs.get("featurize_fn",None)
        self.action_history = deque(maxlen=kwargs.get("history_length",20))
        self.human_study = True
        self.setup_agent()
        self.reset()


    def setup_agent(self):
        if self.policy is not None and self.featurize is not None:
            return
        assert (self.config and self.model_path)
        from burrito_rl.algorithms.trainer_util import get_trainer
        # from burrito_rl.algorithms.agent_ppo_policy import PPOPolicy
        # from ray.rllib.algorithms.ppo import PPOConfig
        # from burrito_rl.model.actor_critic import ActorCritic
        # from ray.rllib.models import ModelCatalog
        from burrito_rl.env_wrapper.burrito_env import BurritoRLLibWrapper
        # from ray.tune import register_env

        # register_env("burrito", lambda env_config: BurritoRLLibWrapper(env_config))
        # ModelCatalog.register_custom_model(
        #     "cc_model",
        #     ActorCritic
        # )
        import burrito_rl.infrastructure # register for envs and models
        trainer = get_trainer(self.config['alg'])(config = self.config)
        self.config = trainer.config
        dummy_rllib_env = BurritoRLLibWrapper(self.config.env_config)
        self.n_actions = dummy_rllib_env.n_actions
        self.policy = trainer.get_policy("polBR")
        self.policy.model.eval()
        self.policy._load_pretrained_model(self.model_path)
        self.featurize = dummy_rllib_env.get_obs

    def reset(self):
        # reset some policy params that may be persistent with incorrect batch sizes
        if hasattr(self.policy, 'prev_decodes'):
            self.policy.prev_decodes = {}
            print("resetting prev decodes")
        if hasattr(self.policy, 'cumulative_regret'):
            self.policy.cumulative_regret = None
        if hasattr(self.policy, 'expert_weights'):
            self.policy.expert_weights = None

        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy.model, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.model.get_initial_state()
            ]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.get_initial_state()
            ]
        else:
            self.rnn_state = []

    def action_probabilities(self, state):
        """
        Arguments:
            - state (burrito_mdp.BurritoState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]
        
        # Compute non-normalized log probabilities from the underlying model
        logits = self.policy.compute_actions(
            np.array([my_obs]), self.rnn_state
        )[2]["action_dist_inputs"]

        # Softmax in numpy to convert logits to normalized probabilities
        return self._softmax(logits)

    def action(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state
        try:
            if type(state) == tuple:
                # When evaluating agent with human
                # input state comes from environment message, containing a tuple of state and info
                state, info = state
                obs = self.featurize(state, info)
            else:
                # When evaluating multiagent performance with evaluator
                # self.featurize_fn = evaluator.env_wrapper.get_obs
                obs = self.featurize(state)
            my_obs = obs[self.agent_index]
            # Use Rllib.Policy class to compute action argmax and action probabilities
            # The first value is action_idx, which we will recompute below so the results are stochastic
            # _, rnn_state, info = self.policy.compute_actions(
            #     my_obs, self.rnn_state
            # )
            for key, array in my_obs.items():
                my_obs[key] = np.expand_dims(array, axis=0)
            [action], rnn_state, info = self.policy.compute_actions_from_input_dict(
                {'obs': my_obs},
                timestep=0
            )
            # Softmax in numpy to convert logits to normalized probabilities
            logits = info["action_dist_inputs"]
            action_probabilities = self._softmax(logits)

            if self.human_study:
                if len(self.action_history) >= self.action_history.maxlen * 0.8:
                    unique_actions = set(self.action_history)
                    unique_actions_without_4 = unique_actions.copy()
                    if 4 in unique_actions_without_4:
                        unique_actions_without_4.remove(4)
                    if len(unique_actions_without_4) <= 3:
                        sorted_actions = np.argsort(-action_probabilities[0])

                        if action in unique_actions:
                            for alt_action in sorted_actions:
                                if alt_action not in unique_actions and alt_action != 4:
                                    action = alt_action
                                    #print(action_probabilities[0])
                                    #print("BREAKING CYCLE, CHOOSING", action)
                                    
                                    break
            self.action_history.append(action)
            #print(self.action_history)

            # The original design is stochastic across different games,
            # Though if we are reloading from a checkpoint it would inherit the seed at that point, producing deterministic results
            # TODO: refer to correct high-level actions here
            # [action_idx] = random.choices(
            #     list(range(len(HighLevelActions.__members__))), action_probabilities[0]
            #     #list(range(self.n_actions)), action_probabilities[0]
            # )
            # agent_action = HighLevelActions.INDEX_TO_ACTION[action_idx]
            agent_action_info = {"action_probs": action_probabilities}
            self.rnn_state = rnn_state
        except Exception as e:
            print(f"An error occurred --- : {e}")
            traceback.print_exc()

        return action, agent_action_info

    def _softmax(self, logits):
        e_x = np.exp(logits.T - np.max(logits))
        return (e_x / np.sum(e_x, axis=0)).T
    



class WrappedAgentBase(mp.Process):
    """
    A top-level base class that can wrap any agent_class
    and be started as a Process with the 'spawn' method.
    """
    def __init__(self, agent_class, agent_index, event_queue, command_queue, mdp, timeout=60, **kwargs):
        mp.Process.__init__(self)
        self.agent_class_str = agent_class   # e.g. "rllib" or "dummy"
        self.agent_index = agent_index
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.event = None
        self.timeout = timeout
        self.last_activity_time = time.time()
        self._initialized_agent = None
        self.mdp = mdp
        self.kwargs = kwargs


    def _init_agent(self):
        """Dynamically initialize the agent (rllib, dummy, etc.)"""
        if self.agent_class_str == "rllib":
            # Import your RLlibAgent here or at the top
            self._initialized_agent = RLlibAgent(self.agent_index, self.mdp, **self.kwargs)
        elif self.agent_class_str == "dummy":
            self._initialized_agent = DummyAgent(self.agent_index, self.mdp, **self.kwargs)
        else:
            raise Exception(f"Agent class {self.agent_class_str} is not supported")
        
        self.command_queue.put({
            "event": "setup",
        })

    def run(self):
        # Make sure the agent is initialized in the child process
        try:
            self._init_agent()
        except Exception as e:
            print(f"Exception in agent __init__: {e}")
            traceback.print_exc()
            return

        print(f"run {self.agent_class_str}")

        while self.event != "remove_player":
            try:
                message = self.event_queue.get(timeout=0.5)
                # Drain any remaining messages in the queue
                while not self.event_queue.empty():
                    message = self.event_queue.get(timeout=0.5)
                self.event = message["event"]
                self.last_activity_time = time.time()

                # Let the actual agent do something
                # e.g. forward pass, or some RLlib logic
                # self._initialized_agent.action(...)
                if self.event == "playing":
                    agent_action, _ = self._initialized_agent.action(message["stats"])
                    # Send the result back
                    self.command_queue.put(agent_action)

            except Exception as e:
                waittime = time.time() - self.last_activity_time
                if waittime > self.timeout:
                    print(f"No incoming message in {waittime} second, shutting down ...")
                    break
                else:
                    # print(f"{self.agent_class_str} agent get exception {e}")
                    # traceback.print_exc()
                    continue

        self.event_queue.close()
        self.command_queue.close()
        print(f"{self.agent_class_str} agent shutdown")
    
def get_agent(agent_class: str,
              agent_index: int,
              event_queue: mp.Queue,
              command_queue: mp.Queue,
              mdp,
              timeout=60,
              **kwargs):
    """
    Wrap any agents into a Process object compatible with real-time human interaction.
    :param agent_class: A class str indicating from which class agent instances will be initialized
    :param agent_index: The index of the agent
    :param event_queue: Event queue from the environment
    :param commend_queue: Command queue back to the environment
    :param mdp: The mdp where the agent will work in.
    :param timeout: Maximum timeout period before shutting down
    :param kwargs: key word arguments for the initialization of the agent_class
    :return: An agent instance wrapped as a Process
    """
    return WrappedAgentBase(
        agent_class=agent_class,
        agent_index=agent_index,
        event_queue=event_queue,
        command_queue=command_queue,
        mdp=mdp,
        timeout=timeout,
        **kwargs
    )
