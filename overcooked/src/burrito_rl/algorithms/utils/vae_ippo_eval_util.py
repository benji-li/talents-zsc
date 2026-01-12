import os
import pickle
from ray.rllib.models import ModelCatalog
from burrito_rl.algorithms.callbacks import LoggingCallbacks

class VAEIPPOEvalCallback(LoggingCallbacks):
    """Simple callback for VAE vs IPPO evaluation"""
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Log rewards for each agent
        vae_reward = episode.agent_rewards.get((0, 'polCluster'), 0)
        ippo_reward = episode.agent_rewards.get((1, 'polIPPO'), 0)
        
        # Track metrics
        episode.custom_metrics.update({
            'vae_reward': vae_reward,
            'ippo_reward': ippo_reward,
            'total_reward': vae_reward + ippo_reward
        })
        
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, 
                              episode=episode, **kwargs)

def vae_ippo_mapping_fn(agent_id, **kwargs):
    """VAE is agent 0, IPPO is agent 1"""
    return "polCluster" if agent_id == 0 else "polIPPO"

def vae_ippo_eval_config(Config):
    """Configuration for VAE Cluster vs IPPO evaluation"""
    # Load cluster parameters
    with open(Config["gaussians_path"], "rb") as f:
        gaussians = pickle.load(f)
    cluster_id = Config["eval_cluster_id"]
    
    # Register VAE model
    from agent_characterization.gen_agents import VAEClusterModel
    ModelCatalog.register_custom_model("VAEClusterModel", VAEClusterModel)
    
    # Set callback
    Config["callbacks"] = VAEIPPOEvalCallback
    
    # Create multiagent configuration
    Config["multiagent"] = {
        "policies": {
            # VAE cluster policy
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
                },
            ),
            # IPPO policy
            "polIPPO": (
                None,
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "framework": "torch",
                    "model": Config["model"],
                },
            )
        },
        "policy_mapping_fn": vae_ippo_mapping_fn,
        "policies_to_train": [],  # No training in evaluation
    }
    
    # Set cluster ID in environment on episode start
    def set_cluster_env(worker, base_env, policies, episode, **kwargs):
        env_id = episode.env_id
        sub_envs = worker.foreach_env(lambda e: e)
        if hasattr(sub_envs[env_id], "set_cluster_id"):
            sub_envs[env_id].set_cluster_id(cluster_id)
    
    # Create combined callback
    orig_callback = Config["callbacks"]
    
    class CombinedCallback(orig_callback):
        def on_episode_start(self, **kwargs):
            set_cluster_env(**kwargs)
            super().on_episode_start(**kwargs)

    Config["callbacks"] = CombinedCallback
    
    return Config
