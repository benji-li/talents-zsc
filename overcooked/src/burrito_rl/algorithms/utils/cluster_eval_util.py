import os
import pickle
import numpy as np
from collections import defaultdict
from ray.rllib.models import ModelCatalog
from burrito_rl.algorithms.callbacks import LoggingCallbacks

class ClusterEvalCallback(LoggingCallbacks):

    def __init__(self):
        super().__init__()
        self.episode_rewards = defaultdict(list)
        self.policy_scores = {}

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

def cluster_eval_mapping_fn(agent_id, **kwargs):
    return "polCluster" if agent_id == 0 else "polPartner"

def cluster_eval_config(trainer, Config):
    with open(Config["gaussians_path"], "rb") as f:
        gaussians = pickle.load(f)
    cluster_id = int(Config["eval_cluster_id"])

    from agent_characterization.gen_agents import VAEClusterModel
    ModelCatalog.register_custom_model("VAEClusterModel", VAEClusterModel)

    Config["callbacks"] = ClusterEvalCallback

    Config["multiagent"] = {
        "policies": {
            "polCluster": (
                None,
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "framework": "torch",
                    "model": {
                        "custom_model": "VAEClusterModel",
                        "custom_model_config": {
                            "vae_path": Config["vae_path"],
                            "obs_shape": Config["obs_shape"],
                            "action_dim": Config["action_dim"],
                            "latent_dim": Config["latent_dim"],
                            "window_length": Config["window_length"],
                            "vae_horizon": Config["vae_horizon"],
                            "cluster_params": gaussians[cluster_id]
                        }
                    }
                }
            ),
            "polPartner": (
                None,
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polPartner",
                    "framework": "torch",
                    "model": Config["model"],
                    "pretrained_model_path": Config["pretrained_model_path_eval"],
                },
            )
        },
        "policy_mapping_fn": cluster_eval_mapping_fn,
        "policies_to_train": [],
    }
    print("========cluster eval config initialized!========")

    return Config
