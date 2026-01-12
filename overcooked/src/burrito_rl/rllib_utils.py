from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
#from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from burrito_rl.env_wrapper.burrito_env import BurritoRLLibWrapper

##################
# Training Utils #
##################

timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

from burrito.visualization.new_state_visualizer import StateVisualizer 

class AgentEvaluatorMultiAgent(AgentEvaluator):
    """
    Class used to get rollouts and evaluate performance of various types of agents.

    TODO: This class currently only fully supports fixed mdps, or variable mdps that can be created with the LayoutGenerator class,
    but might break with other types of variable mdps. Some methods currently assume that the AgentEvaluator can be reconstructed
    from loaded params (which must be pickleable). However, some custom start_state_fns or mdp_generating_fns will not be easily
    pickleable. We should think about possible improvements/what makes most sense to do here.
    """

    def __init__(self, config, force_compute=False):
        """
        env_params (dict): params for creation of an OvercookedEnv
        mdp_fn (callable function): a function that can be used to create mdp
        force_compute (bool): whether should re-compute MediumLevelActionManager although matching file is found
        mlam_params (dict): the parameters for mlam, the MediumLevelActionManager
        debug (bool): whether to display debugging information on init
        """
        self.env_wrapper = BurritoRLLibWrapper(config)
        self.env = self.env_wrapper.env # BurritoEnv
        #self.visualizer = StateVisualizer(player_colors = ['red', 'green', "blue"])
        self.visualizer = StateVisualizer(config.get("layout"))

from burrito_rl.agents.burrito_agent import RLlibAgent
def evaluate(
    config,
    policies,
    num_episodes,
    display,
    ifsave=False,
    save=None,
    verbose=True,
    save_vids=True
):
    if verbose:
        print("eval mdp params", config)
    evaluator = AgentEvaluatorMultiAgent(
        config=config
    )
    agents = []
    for agent in evaluator.env_wrapper.agents: # TODO: make this RLlib agnostic (any agent should work here)
        print("Adding agent", agent, policies[agent])
        agents.append(RLlibAgent(agent, policy=policies[agent], featurize_fn=evaluator.env_wrapper.get_obs))
    
    results = evaluator.evaluate_agent_pair(
        AgentGroup(*agents),
        num_games=num_episodes,
        display=display,
        dir= (None if not ifsave else save),
        display_phi=False,
        info=verbose,
    )
    import os
    import time
    import pickle
    if ifsave:
        os.makedirs(save, exist_ok=True)
        fname = save + "/roll_out_" + str(time.time()) + ".pkl"
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {fname}")


    if save_vids:
        for idx in range(num_episodes):
            img_array = evaluator.visualizer.process_burrito_episode(results["ep_states"][idx], results["ep_infos"][idx],results["ep_lengths"][idx], results["ep_actions"][idx])
            evaluator.visualizer.make_video(img_array) 
    return results
