from burrito_rl.algorithms.callbacks import LoggingCallbacks
import random
import numpy as np
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from collections import defaultdict
import os


class BRCallback(LoggingCallbacks):
    """
    Logging BR+partner performance and mechanism of priority-based partner sampling
    """
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = defaultdict(list)  # Track rewards for each episode by policy pair
        self.policy_scores = {}    # Average rewards for each policy when paired with BR
        self.partner_policy_counts = defaultdict(int)
    
            
    def on_episode_end(self, *, worker:RolloutWorker, base_env, policies, episode:EpisodeV2, **kwargs):
        # Get the partner policy that worked with BR
        partner_policy = episode._agent_to_policy[0]
        self.partner_policy_counts[partner_policy] += 1
        for partner_policy in self.partner_policy_counts:
            episode.custom_metrics[f'with_{partner_policy}_count_per_worker'] = self.partner_policy_counts[partner_policy]
            
        # Calculate total reward for this episode
        br_reward = episode.agent_rewards[(1,'polBR')]
        episode.custom_metrics[f'with_{partner_policy}_reward'] = br_reward
            
        # Record the reward
        self.episode_rewards[partner_policy].append(br_reward)
        
        # Keep only recent history (last 5 episodes)
        max_history = worker.config["num_history"]
        if len(self.episode_rewards[partner_policy]) > max_history:
            self.episode_rewards[partner_policy] = self.episode_rewards[partner_policy][-max_history:]
            
        # Update average scores
        for pid, rewards in self.episode_rewards.items():
            self.policy_scores[pid] = np.mean(rewards)
            
        # Share scores with worker
        if not hasattr(worker, "global_vars"):
            worker.global_vars = {}
        worker.global_vars["policy_scores"] = self.policy_scores

        # Call parent method
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, 
                              episode=episode, **kwargs)
        



def best_response_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == 1: # TODO: currently assume only two agents
        return 'polBR'
    else:
        # Only set partner policy once per episode
        if hasattr(episode, "partner_policy"):
            return episode.partner_policy
        
        policy_ids = list(worker.policy_map.keys())
        policy_ids.remove('polBR')
        policy_ids = [pid for pid in policy_ids 
                      if pid != 'polBR' 
                      and pid != 'polBREval' 
                      and not pid.startswith('polEval')]

        print("========================")
        print(policy_ids, "policy ids to select from")
        episode.partner_policy = random.choice(policy_ids)
        if worker.config['partner_sampling_method'] != 'priority':
            return episode.partner_policy
        print("priority sampling")
        if hasattr(worker, "global_vars") and "policy_scores" in worker.global_vars:
            
            policy_scores = worker.global_vars["policy_scores"]

            if len(policy_scores) < len(list(worker.policy_map.keys()))-1:
                return episode.partner_policy
            
            inv_rewards = {}
            for pid, score in policy_scores.items():
                if score <= 0:
                    inv_rewards[pid] = float('inf')  # Treat non-positive rewards as infinitely bad
                else:
                    inv_rewards[pid] = 1.0 / score
            
            # Sort policies by inverse rewards (ascending)
            sorted_policies = sorted(inv_rewards.items(), key=lambda x: x[1])
            
            # Assign ranks from 1 to n (1 for lowest inverse reward, n for highest)
            # This means rank 1 is best-performing policy, rank n is worst-performing
            ranks = {}
            for i, (pid, _) in enumerate(sorted_policies):
                ranks[pid] = i + 1  # Ranks start from 1
            
            # Calculate selection probabilities using the formula
            # p(pid) = rank(pid) / sum(all_ranks)
            total_rank = sum(ranks.values())
            probs = [ranks[pid]/total_rank for pid in policy_ids]
            print("-------------------------")
            print("priority_sampling probs", probs)
            # Select based on calculated rank-based probabilities
            episode.partner_policy = random.choices(policy_ids, weights=probs, k=1)[0]

        return episode.partner_policy


def best_response_rllib_config(trainer, Config:dict) -> dict:

    Config["callbacks"] = BRCallback
    policy_ids = [f"pol{i+1}" for i in range(Config["n_population"])]
    if Config.get("specific_partners"):
        policy_ids = [f"pol{i}" for i in Config["specific_partners"]]
    
    policy_ids += ['polBR']

    if Config.get("custom_eval", False):
        eval_policy_ids = [f"polEval{i}" for i in Config["pretrained_model_paths"]]
        eval_model_paths = [os.path.join(Config["pretrained_model_dir"],"pol"+str(Config["pretrained_model_paths"][i]),"policy_state.pkl") for i in range(len(Config["pretrained_model_paths"]))]
        print("Evaluating with models:", eval_model_paths)
        policy_ids += ['polBREval']

    Config["multiagent"] = {
        "policies": {
            **{
                pid: (
                    None,  # policy spec
                    Config["env_config"]["observation_space"],  # observation spec
                    Config["env_config"]["action_space"],       # action spec
                    {
                        "name": pid,
                        "framework": "torch",
                        "model": Config["model"],
                        "pretrained_model_path": f"{Config['population_path']}/{pid}/policy_state.pkl" if (pid != 'polBR' and pid != 'polBREval') else 0,
                    },
                )
                for i, pid in enumerate(policy_ids)
            },
            # Evaluation policies
            **({
                pid: (
                    None,  # policy spec
                    Config["env_config"]["observation_space"],  # observation spec
                    Config["env_config"]["action_space"],       # action spec
                    {
                        "name": pid,
                        "framework": "torch",
                        "model": Config["model"],
                        "pretrained_model_path": eval_model_paths[i],
                    },
                ) 
                for i,pid in enumerate(eval_policy_ids)
            } if Config.get("custom_eval", False) else {}),
        },
        # Use the top-level function instead of a lambda
        "policy_mapping_fn": best_response_policy_mapping_fn,
        "policies_to_train": ['polBR'],
    }

    if Config.get("custom_eval", False):
        from burrito_rl.algorithms.custom_eval import vae_cluster_br_eval
        Config["custom_eval_function"] = vae_cluster_br_eval
        Config["evaluation_interval"] = Config.get("evaluation_interval", 20)
        # WARN: uncommenting this line causes the program to freeze
        #Config["evaluation_num_workers"] = Config.get("evaluation_num_workers", 1)

        print("EVAL INTERVAL", Config["evaluation_interval"])
        #print("EVAL WORKERS", Config["evaluation_num_workers"])

    return Config
