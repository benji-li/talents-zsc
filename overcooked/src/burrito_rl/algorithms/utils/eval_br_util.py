import os
import random
import numpy as np
import pickle
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from collections import defaultdict
from ray.rllib.models import ModelCatalog
from burrito_rl.algorithms.callbacks import LoggingCallbacks

# this is not actually called during eval runs -> called during training evals
def eval_br_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == 1:  # BR agent
        print('mapped to br1')
        return 'polBR'
    elif agent_id == 2:  # BR agent for 3-agent environments
        print('mapped to br')
        return 'polBR1'
    else:
        print('mapped to ppo')
        return 'polPPO'

def eval_br_config(trainer, Config):
    """
    Configuration for evaluating BR agents
    """
    # Load Gaussian clusters from file
    gaussians_path = Config.get("gaussians_path")
    if not gaussians_path or not os.path.exists(gaussians_path):
        raise ValueError(f"Gaussians path {gaussians_path} not found or not specified")

    with open(gaussians_path, "rb") as f:
        gaussians = pickle.load(f)
    print("=====Gaussians Found=====")
    print(gaussians.keys())
    print(len(gaussians))

    # Store gaussians in config
    #Config["gaussians"] = gaussians
    #Config["n_clusters"] = len(gaussians)

    # Set up the callback
    Config["callbacks"] = LoggingCallbacks

    from burrito_rl.algorithms.agent_noreg_policy import NoRegretPolicy
    from agent_characterization.gen_agents import VAEClusterModel
    from burrito_rl.model.cluster_conditioned_actor_critic import ClusterConditionedActorCritic
    from burrito_rl.model.cluster_conditioned_actor_critic import ActionBiasedClusterConditionedActorCritic

    ModelCatalog.register_custom_model("VAEClusterModel", VAEClusterModel)
    ModelCatalog.register_custom_model("ClusterConditionedActorCritic", ClusterConditionedActorCritic)
    ModelCatalog.register_custom_model("ActionBiasedClusterConditionedActorCritic", ActionBiasedClusterConditionedActorCritic)

    # Set up multiagent configuration
    Config["multiagent"] = {
        "policies": {
            # BR policy
            "polPPO": (
                None,  # default ppo policy
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polPPO",
                    "framework": "torch",
                    "model": Config["model"],
                    "pretrained_model_path": Config["pretrained_model_path_eval"],
                },
            ),

            "polBR": (
                NoRegretPolicy,
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polBR",
                    "framework": "torch",
                    "model": {
                        "custom_model": Config["br_model_name"],
                        "custom_model_config": {
                            "num_clusters": len(gaussians),
                            "model_type": "custom_conv",
                            "input_conv_channels": Config["obs_shape"][-1],
                            "actor_output_size": Config["action_dim"],
                            "critic_share_layers": False,
                            "conv_filters": Config["model"]["custom_model_config"]["conv_filters"],
                            "actor_layer_sizes": Config["br_actor_layer_sizes"],
                            "critic_layer_sizes": Config["br_critic_layer_sizes"],
                            "action_masking": True
                        }
                    },

                    # noregretpolicy params:
                    "vae_path": Config["vae_path"],
                    "obs_shape": Config["obs_shape"],
                    "action_dim": Config["action_dim"],
                    "latent_dim": Config["latent_dim"],
                    "window_length": Config["window_length"],
                    "vae_horizon": Config["vae_horizon"],
                    "cluster_params": gaussians,
                    "pretrained_model_path": Config["pretrained_model_path_BR"],

                    # fixed share params
                    "learning_rate": Config["learning_rate"],
                    "alpha": Config["alpha"],
                    "decay_factor": Config["decay_factor"]

                },
            ),

            "polBR1": (
                NoRegretPolicy,
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polBR1",
                    "framework": "torch",
                    "model": {
                        "custom_model": Config["br_model_name"],
                        "custom_model_config": {
                            "num_clusters": len(gaussians),
                            "model_type": "custom_conv",
                            "input_conv_channels": Config["obs_shape"][-1],
                            "actor_output_size": Config["action_dim"],
                            "critic_share_layers": False,
                            "conv_filters": Config["model"]["custom_model_config"]["conv_filters"],
                            "actor_layer_sizes": Config["br_actor_layer_sizes"],
                            "critic_layer_sizes": Config["br_critic_layer_sizes"],
                            "action_masking": True
                        }
                    },

                    # noregretpolicy params:
                    "vae_path": Config["vae_path"],
                    "obs_shape": Config["obs_shape"],
                    "action_dim": Config["action_dim"],
                    "latent_dim": Config["latent_dim"],
                    "window_length": Config["window_length"],
                    "vae_horizon": Config["vae_horizon"],
                    "cluster_params": gaussians,
                    "pretrained_model_path": Config["pretrained_model_path_BR"],

                    # fixed share params
                    "learning_rate": Config["learning_rate"],
                    "alpha": Config["alpha"],
                    "decay_factor": Config["decay_factor"]

                },
            )

        },
        "policy_mapping_fn": eval_br_mapping_fn,
        "policies_to_train": [],
    }
    print("======configs initialized!======")
    
    return Config
