import os
import numpy as np
import pickle
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from burrito_rl.algorithms.callbacks import LoggingCallbacks
from collections import defaultdict


class NoRegretAdapter:

    def __init__(self, num_clusters, cluster_policies, learning_rate=0.1, exploration_rate=0.1):
        """
        Args:
            num_clusters: Number of clusters to evaluate
            cluster_policies: Dictionary mapping cluster IDs to policies
            learning_rate: How quickly to update weights based on accuracy
            exploration_rate: Probability of random exploration
        """
        self.num_clusters = num_clusters
        self.cluster_policies = cluster_policies
        self.weights = np.ones(num_clusters) / num_clusters
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.current_cluster = 0
        self.prediction_accuracies = defaultdict(list)  # Track accuracy by cluster
        self.avg_accuracies = np.zeros(num_clusters)
        self.cluster_visits = np.zeros(num_clusters)
        
    def select_cluster(self):
        """Choose cluster based on current weights with exploration"""
        if np.random.random() < self.exploration_rate:
            self.current_cluster = np.random.randint(0, self.num_clusters)
        else:
            self.current_cluster = np.argmax(self.weights)
        
        self.cluster_visits[self.current_cluster] += 1
        return self.current_cluster
    
    def evaluate_clusters(self, partner_obs, partner_action, env_num_clusters):
        """
        Evaluate each cluster's prediction accuracy for the partner's action
        
        Args:
            partner_obs: The observation for the partner agent
            partner_action: The actual action taken by the partner
            env_num_clusters: Number of clusters in the environment
            
        Returns:
            dict: Mapping from cluster_id to prediction accuracy
        """
        cluster_accuracies = {}
        
        for cluster_id, policy in self.cluster_policies.items():

            # Get action prediction from this cluster's policy
            with torch.no_grad():
                predicted_action_logits = predict_partner_action(vae, z, processed_obs)
                
                # Convert logits to action probabilities
                predicted_action_probs = torch.softmax(predicted_action_logits, dim=0)
                
                actual_action_tensor = torch.tensor(actual_partner_action, device=device)

                loss = -1*torch.log(predicted_action_probs[actual_partner_action]+1e-10)
                loss = min(5.0, loss.item()) # bound per-step regret

                action_dist_inputs, _ = policy.model.forward({
                    "obs": partner_obs,
                }, [], None)

                predicted_action_probs = torch.softmax(action_dist_inputs, dim=0)
                loss = -1*torch.log(predicted_action_probs[partner_action]+1e-10)
                loss = min(5.0, loss.item()) # bound per-step regret

                # Get most likely action
                predicted_action = torch.argmax(action_dist_inputs).item()

                # Calculate accuracy (1 if correct, 0 if incorrect)
                accuracy = 1.0 if predicted_action == partner_action else 0.0
                cluster_accuracies[cluster_id] = accuracy

                # Store accuracy for tracking
                self.prediction_accuracies[cluster_id].append(accuracy)

                # Keep limited history
                if len(self.prediction_accuracies[cluster_id]) > 100:
                    self.prediction_accuracies[cluster_id].pop(0)

                # Update average accuracy
                self.avg_accuracies[cluster_id] = np.mean(self.prediction_accuracies[cluster_id])
        
        return cluster_accuracies
    
    def update(self, partner_obs, partner_action, env_num_clusters):
        """
        Update weights based on prediction accuracies and select next cluster
        
        Args:
            partner_obs: The observation for the partner agent
            partner_action: The actual action taken by the partner 
            env_num_clusters: Number of clusters in the environment
            
        Returns:
            int: Next cluster to use
        """
        # Get accuracies for all clusters
        cluster_accuracies = self.evaluate_clusters(partner_obs, partner_action, env_num_clusters)
        
        # Update weights based on prediction accuracy
        for cluster_id, accuracy in cluster_accuracies.items():
            # Multiplicative weight update
            self.weights[cluster_id] *= (1 + self.learning_rate * accuracy)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Select next cluster
        return self.select_cluster()
    
    def get_state(self):
        """Get current state of the adapter"""
        return {
            "weights": self.weights.copy(),
            "avg_accuracies": self.avg_accuracies.copy(),
            "current_cluster": self.current_cluster,
            "cluster_visits": self.cluster_visits.copy()
        }


class NoRegretEvalCallback(LoggingCallbacks):
    """Callback for no-regret cluster adaptation based on prediction accuracy"""
    
    def __init__(self):
        super().__init__()
        self.adapter = None
        self.update_frequency = 5  # Update every N steps
        self.current_cluster = 0
        self.partner_agent_id = 0  # Partner is agent 0
        self.adaptive_agent_id = 1  # Adaptive agent is 1
        
    def on_episode_start(self, *, worker: RolloutWorker, base_env, policies, episode: EpisodeV2, **kwargs):
        # Initialize adapter if needed
        if self.adapter is None:
            # Get number of clusters
            env = worker.foreach_env(lambda e: e)[0]
            num_clusters = env.num_clusters
            
            # Get policy objects for each cluster
            cluster_policies = {
                i: policies[f"polCluster{i}"] 
                for i in range(num_clusters)
                if f"polCluster{i}" in policies
            }
            
            self.adapter = NoRegretAdapter(
                num_clusters=num_clusters,
                cluster_policies=cluster_policies,
                learning_rate=worker.config.get("no_regret_lr", 0.1),
                exploration_rate=worker.config.get("no_regret_exploration", 0.1)
            )
        
        # Select initial cluster
        self.current_cluster = self.adapter.select_cluster()
        env = worker.foreach_env(lambda e: e)[episode.env_id]
        env.set_cluster_id(self.current_cluster)
        
        # Initialize episode data
        episode.user_data["current_cluster"] = self.current_cluster
        episode.user_data["cluster_switches"] = 0
        
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
    
    def on_episode_step(self, *, worker: RolloutWorker, base_env, episode: EpisodeV2, **kwargs):
        # Only update at specified frequency
        if episode.length % self.update_frequency != 0 or episode.length == 0:
            return
        
        # Get partner's actual action
        partner_policy_id = episode._agent_to_policy[self.partner_agent_id]
        partner_action = episode.last_action_for((self.partner_agent_id, partner_policy_id))
        
        # Skip if we don't have the partner's action yet
        if partner_action is None:
            return
        
        # Get latest observation for the partner
        partner_obs = episode.last_observation_for((self.partner_agent_id, partner_policy_id))
        if partner_obs is None:
            return
        
        # Get environment
        env = worker.foreach_env(lambda e: e)[episode.env_id]
            
        # Update adapter with partner observation and action
        next_cluster = self.adapter.update(partner_obs, partner_action, env.num_clusters)
        
        # Apply cluster change if different
        if next_cluster != episode.user_data["current_cluster"]:
            env.set_cluster_id(next_cluster)
            episode.user_data["current_cluster"] = next_cluster
            episode.user_data["cluster_switches"] = episode.user_data.get("cluster_switches", 0) + 1
            
            # Log the switch for this timestep
            episode.custom_metrics[f'timestep_{episode.length}_switch_to_{next_cluster}'] = 1.0
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies, episode: EpisodeV2, **kwargs):
        # Log metrics from adapter
        adapter_state = self.adapter.get_state()
        
        for c in range(self.adapter.num_clusters):
            # Log weights
            episode.custom_metrics[f'cluster_{c}_weight'] = adapter_state["weights"][c]
            
            # Log average prediction accuracy
            episode.custom_metrics[f'cluster_{c}_accuracy'] = adapter_state["avg_accuracies"][c]
            
            # Log visit counts
            episode.custom_metrics[f'cluster_{c}_visits'] = adapter_state["cluster_visits"][c]
        
        # Log overall metrics
        episode.custom_metrics['final_cluster'] = episode.user_data["current_cluster"]
        episode.custom_metrics['total_cluster_switches'] = episode.user_data.get("cluster_switches", 0)
        episode.custom_metrics['exploration_rate'] = self.adapter.exploration_rate
        
        # Optionally decay exploration rate
        if worker.config.get("decay_exploration", True):
            self.adapter.exploration_rate = max(0.01, self.adapter.exploration_rate * 0.95)
        
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)


def noregret_policy_mapping_fn(agent_id, **kwargs):
    """
    Policy mapping function for no-regret evaluation
    Partner is agent 0, BR plays agent 1
    """
    return 'polAdaptive' if agent_id == 1 else 'polPartner'

def noregret_eval_config(trainer, Config):
    """Configuration for No-Regret Cluster adaptation evaluation"""
    # Load Gaussian clusters from file
    gaussians_path = Config.get("gaussians_path")
    if not gaussians_path or not os.path.exists(gaussians_path):
        raise ValueError(f"Gaussians path {gaussians_path} not found or not specified")
    
    with open(gaussians_path, "rb") as f:
        gaussians = pickle.load(f)
    print("=====Gaussians Found=====")
    print(gaussians.keys())
    print(len(gaussians))
    
    # Set number of clusters
    num_clusters = len(gaussians)
    Config["num_clusters"] = num_clusters
    
    # Create policy IDs for each cluster + main policies
    cluster_policy_ids = [f"polCluster{i}" for i in range(num_clusters)]
    
    # Register needed models
    from agent_characterization.gen_agents import VAEClusterModel
    from burrito_rl.model.cluster_conditioned_actor_critic import ClusterConditionedActorCritic
    
    ModelCatalog.register_custom_model("VAEClusterModel", VAEClusterModel)
    ModelCatalog.register_custom_model("ClusterConditionedActorCritic", ClusterConditionedActorCritic)
    
    # Set up the callback
    Config["callbacks"] = NoRegretEvalCallback
    
    # Set up multiagent configuration
    Config["multiagent"] = {
        "policies": {
            # VAE cluster policies for prediction
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
            # Partner agent (agent 0, the one we're trying to model)
            "polPartner": (
                None, 
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "framework": "torch",
                    "model": Config.get("partner_model", Config["model"]),
                },
            ),
            # Adaptive agent (agent 1, uses cluster-conditioned model)
            "polAdaptive": (
                None, 
                Config["env_config"]["observation_space"],
                Config["env_config"]["action_space"],
                {
                    "framework": "torch",
                    "model": {
                        "custom_model": "ClusterConditionedActorCritic",
                        "custom_model_config": Config.get("adaptive_model_config", {})
                    },
                },
            )
        },
        "policy_mapping_fn": noregret_policy_mapping_fn,
        "policies_to_train": [],  # No training during evaluation
    }
    
    # Ensure environment is configured for cluster conditioning
    if not Config["env_config"].get("cluster_br", False):
        print("WARNING: Setting env_config.cluster_br=True for no-regret evaluation")
        Config["env_config"]["cluster_br"] = True
    
    if not Config["env_config"].get("num_clusters"):
        print(f"WARNING: Setting env_config.num_clusters={num_clusters}")
        Config["env_config"]["num_clusters"] = num_clusters
    
    return Config
