import os
import random
import numpy as np
import pickle
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from collections import defaultdict
from ray.rllib.models import ModelCatalog
from burrito_rl.algorithms.callbacks import LoggingCallbacks

class VAEBRCallback(LoggingCallbacks):
    """
    Callback for VAE Cluster Best Response training
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = defaultdict(list)  # Track rewards by cluster
        self.policy_scores = {}  # Average rewards for each cluster
        self.policy_vars = {}    # Reward variances for each cluster
        self.cluster_counts = defaultdict(int)
        
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        # When assigning policy in policy_mapping_fn, store cluster_id
#        print("EPISODE AGENT",episode._agent_to_policy)
#        partner_policy = episode._agent_to_policy[0]
#        if "Cluster" in partner_policy:
#            cluster_id = int(partner_policy.replace("polCluster", ""))
#            # Store for this episode
#            episode.cluster_id = cluster_id
#
#            env_id = episode.env_id
#
#            envs = base_env.get_sub_environments()
#            envs[env_id].set_cluster_id(cluster_id)
#            print("==============================")
#            print("env id:", env_id, "cluster_id", cluster_id)
#
            super().on_episode_start(worker=worker,base_env=base_env, policies=policies, episode=episode, **kwargs)

    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies, episode: EpisodeV2, **kwargs):
        # Get the partner policy/cluster that worked with BR
        partner_policy = episode._agent_to_policy[0]

        if "Eval" in partner_policy:
            super().on_episode_end(worker=worker, base_env=base_env, policies=policies, 
                              episode=episode, **kwargs)
            return

        cluster_id = int(partner_policy.replace("polCluster", ""))
        

        self.cluster_counts[cluster_id] += 1
        for cid in self.cluster_counts:
            episode.custom_metrics[f'with_cluster{cid}_count'] = self.cluster_counts[cid]
            
        # Calculate total reward for this episode
        br_reward = episode.agent_rewards[(1, 'polBR')]
        episode.custom_metrics[f'with_cluster{cluster_id}_reward'] = br_reward

        # Record the reward
        self.episode_rewards[cluster_id].append(br_reward)
        
        # Keep only recent history
        max_history = worker.config["num_history"]
        if len(self.episode_rewards[cluster_id]) > max_history:
            self.episode_rewards[cluster_id] = self.episode_rewards[cluster_id][-max_history:]
            
        # Update average scores
        # for cid, rewards in self.episode_rewards.items():
        #     self.policy_scores[cid] = np.mean(rewards)
        #     if len(rewards) > 1:
        #         self.policy_vars[cid] = np.var(rewards)
        #     else:
        #         print("Variance not computed for cluster", cid, "insufficient data")
            
        # # Share scores with worker 
        # if not hasattr(worker, "global_vars"):
        #     worker.global_vars = {}
        # worker.global_vars["cluster_scores"] = self.policy_scores
        # worker.global_vars["cluster_vars"] = self.policy_vars

        # Call parent method
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, 
                              episode=episode, **kwargs)


def vae_cluster_br_mapping_fn(agent_id, episode, worker, **kwargs):
    """
    Policy mapping function for VAE Cluster Best Response training
    """
    if agent_id == 1:  # BR agent
        #print("BR agent chosen")
        return 'polBR'
    elif agent_id == 2: # adding for 3 agent support 
        #print("BR1 agent chosen")
        return 'polBR1'
    else:

        #print("partner agent chosen")
        # Only set partner policy once per episode
        if hasattr(episode, "partner_policy"):
            return episode.partner_policy
        
        # Get the list of cluster policies
        cluster_policies = [pid for pid in worker.policy_map.keys() 
                           if 'Cluster' in pid]
        #print("====================")
        #print(cluster_policies, "cluster policies to train with")
        
        # Choose a partner policy
        if worker.config.get('partner_sampling_method') == 'random':
            # Random sampling
            episode.partner_policy = random.choice(cluster_policies)

        elif worker.config.get('partner_sampling_method') == 'priority':
            print("priority sampling")
            # Priority-based sampling
            print("Worker global vars:", worker.global_vars)
            if hasattr(worker, "global_vars") and "cluster_scores" in worker.global_vars:
                cluster_scores = worker.global_vars["cluster_scores"]
                
                print("Cluster scores:", cluster_scores)
                # If not all clusters have been evaluated yet, use random sampling
                if len(cluster_scores) < len(cluster_policies):
                    episode.partner_policy = random.choice(cluster_policies)
                    print("Not all clusters evaluated yet, choosing random policy:", episode.partner_policy)
                    return episode.partner_policy
                
                # Convert scores to priorities (prioritize low-performing clusters)
                inv_rewards = {}
                for pid in cluster_policies:
                    cluster_id = int(pid.replace("polCluster", ""))
                    score = cluster_scores.get(cluster_id, 0)
                    
                    if score <= 0:
                        inv_rewards[pid] = float('inf')  # Prioritize non-positive rewards
                    else:
                        inv_rewards[pid] = 1.0 / score
                
                # Sort by inverse rewards
                sorted_policies = sorted(inv_rewards.items(), key=lambda x: x[1])
                
                # Assign ranks (1 = best performing, n = worst performing)
                ranks = {}
                for i, (pid, _) in enumerate(sorted_policies):
                    ranks[pid] = i + 1
                
                # Calculate selection probabilities
                total_rank = sum(ranks.values())
                probs = [ranks[pid]/total_rank for pid in cluster_policies]
                
                # Select based on rank probabilities
                episode.partner_policy = random.choices(cluster_policies, weights=probs, k=1)[0]
            else:
                print("exception: choosing random sampling")
                episode.partner_policy = random.choice(cluster_policies)

        # Store cluster ID on the episode for observation augmentation
        cluster_id = int(episode.partner_policy.replace("polCluster", ""))
        episode.cluster_id = cluster_id

        env_id = episode.env_id

        sub_envs = worker.foreach_env(lambda e: e)
        sub_envs[env_id].set_cluster_id(cluster_id)
        #print("==============================")
        #print("env id:", env_id, "cluster_id", sub_envs[env_id].cluster_id)

        return episode.partner_policy


def vae_cluster_br_config(trainer, Config):
    """
    Configuration for VAE Cluster Best Response training
    """
    # Load Gaussian clusters from file
    gaussians_path = Config.get("gaussians_path")
    if not gaussians_path or not os.path.exists(gaussians_path):
        raise ValueError(f"Gaussians path {gaussians_path} not found or not specified")
    
    with open(gaussians_path, "rb") as f:
        gaussians = pickle.load(f)
    gaussians = {int(k): v for k, v in gaussians.items()}  # Ensure keys are ints and not int32
    print("=====Gaussians Found=====")
    print(gaussians.keys())
    print(len(gaussians))
    
    # Store gaussians in config
    #Config["gaussians"] = gaussians
    #Config["n_clusters"] = len(gaussians)
    
    # Set up the callback
    Config["callbacks"] = VAEBRCallback
    
    # Create policy IDs for each cluster + BR policy
    cluster_policy_ids = [f"polCluster{i}" for i in range(len(gaussians))]

    if Config.get("custom_eval", False):
        eval_policy_ids = [f"polEval{i}" for i in Config["pretrained_model_paths"]]
        eval_model_paths = [os.path.join(Config["pretrained_model_dir"],"pol"+str(Config["pretrained_model_paths"][i]),"policy_state.pkl") for i in range(len(Config["pretrained_model_paths"]))]
        print("Evaluating with models:", eval_model_paths)

    from agent_characterization.gen_agents import VAEClusterModel
    from burrito_rl.model.cluster_conditioned_actor_critic import ClusterConditionedActorCritic
    from burrito_rl.model.cluster_conditioned_actor_critic import ActionBiasedClusterConditionedActorCritic
    from burrito_rl.algorithms.agent_noreg_policy import NoRegretPolicy

    ModelCatalog.register_custom_model("VAEClusterModel", VAEClusterModel)
    ModelCatalog.register_custom_model("ClusterConditionedActorCritic", ClusterConditionedActorCritic)
    ModelCatalog.register_custom_model("ActionBiasedClusterConditionedActorCritic", ActionBiasedClusterConditionedActorCritic)

    # Set up multiagent configuration
    Config["multiagent"] = {
        "policies": {
            # VAE cluster policies
            **{
                pid: (
                    None,  # policy spec
                    Config["env_config"]["observation_space"],  # observation spec
                    Config["env_config"]["action_space"],       # action spec
                    {
                        "name": pid,
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
                                "cluster_params": gaussians[int(pid.replace("polCluster", ""))]
                            }
                        }
                    },
                ) 
                for pid in cluster_policy_ids
            },
            # BR policy
            "polBR": (
                None,  # policy spec
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polBR",
                    "framework": "torch",
                    "model": Config["model"] # this is actionbiasedclusterconditionedactorcritic
                },
            ),
            "polBR1": (
                None,  # policy spec
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polBR1",
                    "framework": "torch",
                    "model": Config["model"] # this is actionbiasedclusterconditionedactorcritic
                },
            ),
            # BR policy
            "polBREval": (
                NoRegretPolicy,  # policy spec
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "name": "polBREval",
                    "framework": "torch",
                    "model": Config["model"], # this is actionbiasedclusterconditionedactorcritic

                    "vae_path": Config["vae_path"],
                    "obs_shape": Config["obs_shape"],
                    "action_dim": Config["action_dim"],
                    "latent_dim": Config["latent_dim"],
                    "window_length": Config["window_length"],
                    "vae_horizon": Config["vae_horizon"],
                    "cluster_params": gaussians,
                    # NOTE: this won't be incl
                    #"pretrained_model_path": Config["pretrained_model_path_BR"],

                    # fixed share params
                    "learning_rate": Config["learning_rate"],
                    "alpha": Config["alpha"],
                    "decay_factor": Config["decay_factor"]

                },

            ),
 
            # Evaluation policies
            **({
                pid: (
                    None,  # policy spec
                    Config["env_config"]["observation_space"],  # observation spec
                    Config["env_config"]["action_space"],       # action spec
                    {
                        "name": pid,
                        "framework": "torch",
                        "model": {
                            "custom_model": "IPPO_model",
                            "custom_model_config": {
                                "model_type": "custom_conv",
                                "input_conv_channels": Config["obs_shape"][-1],
                                "actor_output_size": Config["action_dim"],
                                "critic_share_layers": False,
                                "conv_filters": Config["model"]["custom_model_config"]["conv_filters"],
                                "actor_layer_sizes": Config["model"]["custom_model_config"]["actor_layer_sizes"],
                                "critic_layer_sizes": Config["model"]["custom_model_config"]["critic_layer_sizes"],
                                "action_masking": True
                            }
                        },
                       "pretrained_model_path": eval_model_paths[i],
                    },
                ) 
                for i,pid in enumerate(eval_policy_ids)
            } if Config.get("custom_eval", False) else {}),
        },
        "policy_mapping_fn": vae_cluster_br_mapping_fn,
        "policies_to_train": ['polBR','polBR1'],
    }

    if Config.get("custom_eval", False):
        from burrito_rl.algorithms.custom_eval import vae_cluster_br_eval
        Config["custom_eval_function"] = vae_cluster_br_eval
        Config["evaluation_interval"] = Config.get("evaluation_interval", 20)
        # WARN: uncommenting this line causes the program to freeze
        #Config["evaluation_num_workers"] = Config.get("evaluation_num_workers", 1)

        print("EVAL INTERVAL", Config["evaluation_interval"])
        #print("EVAL WORKERS", Config["evaluation_num_workers"])

    print("======configs initialized!======")
    
    return Config