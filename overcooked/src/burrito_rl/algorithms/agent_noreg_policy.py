from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.qmix.qmix_policy import QMixTorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.torch_mixins import ValueNetworkMixin
from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2, _unpack_obs
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import TensorType

from burrito_rl.algorithms.utils.ppo_util import compute_async_gae_for_sample_batch, centralized_postprocess
from ray import cloudpickle
from pathlib import Path
import torch
from ray.rllib.evaluation.postprocessing import Postprocessing

from typing import Dict, List, Optional, Tuple, Type, Union
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import os
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
import numpy as np
from ray.rllib.utils.annotations import override

from agent_characterization.analysis.analyze_vae import load_vae
from torch.nn.functional import cross_entropy

class NoRegretPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.opponent_count = config['env_config'].get('n_agents') - 1
        self.pretrain_path = config.get("pretrained_model_path", None)
        self.model_type = config["model"].get("custom_model", "ClusterConditionedActorCritic")
        self.action_level = config["env_config"].get("action_level", 'high')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = config["obs_shape"]
        self.action_dim = config["action_dim"]
        #self.name = config.get("name")
        
        self.vae = load_vae(
            model_path= os.path.join(config["vae_path"], "encoder.pt"),
            obs_shape = config["obs_shape"],
            action_dim = config["action_dim"],
            latent_dim = config["latent_dim"],
            window_length =config["window_length"],
            horizon = config["vae_horizon"],
        )

        # these need to be the full list of cluster_params so we can sample from every cluster mean
        self.cluster_params = config["cluster_params"]
        self.mvns = self._init_cluster(self.cluster_params)
    
        self.expert_weights = None

        # Initialize cumulative regret
        self.cumulative_regret = None # gets init at compute_actions_from_input_dict
        
        # Track which expert is selected at each timestep
        self.episode_expert_selections = []

        self.prev_decodes = {}

        # fixed share params
        self.lr = config["learning_rate"]
        self.alpha = config["alpha"]
        self.decay_factor = config["decay_factor"]

        self.inference_time = 0.0
        self.num_samples = 0

        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        
        # this should be a cluster conditioned critic
        self._load_pretrained_model(self.pretrain_path)
        

    def _init_cluster(self, cluster_params):
        # Create Gaussian distributions for each expert cluster
        mvns = {}
        for cluster_id, params in cluster_params.items():
            mean = torch.tensor(params['mean'], dtype=torch.float32, device=self.device)
            cov = torch.tensor(params['cov'], dtype=torch.float32, device=self.device)
            #eye = torch.eye(cov.shape[0], dtype=torch.float32, device=self.device)
            #cov = cov + eye * 1e-4  # Regularization
            mvns[cluster_id] = torch.distributions.MultivariateNormal(mean, cov)
        return mvns
    

    def _load_pretrained_model(self, import_path, model = None):
        """
        load pretrained model for the learning agent
        """
        if not import_path or import_path == 0:
            print("No pretrained model path provided")
            return
        checkpoint_path = Path(import_path).expanduser()
        print("==========LOADING FROM CHECKPOINT==========")
        print(checkpoint_path,"checkpoint path")
        print("NO REGRET LOADING")
        assert os.path.exists(checkpoint_path)
        with checkpoint_path.open('rb') as f:
            checkpoint = cloudpickle.load(f)
            state_dict = checkpoint['weights']
            filtered_state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(convert_to_torch_tensor(filtered_state_dict,device=self.device),
                                            strict=False)
            print("NO REGRET POLICY LOADED SUCCESSFULLY")

    @override(TorchPolicyV2)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        if self.model_type != 'MAPPO_model':
            # compute value with torch_mixin
            return super().extra_action_out(input_dict, state_batches, model, action_dist)
        elif self.model_type == "MAPPO_model":
            # agent do not have global info access in compute actions. Therefore not computing value here
            return {}


    def _unpack_obs(self, packed_obs, tensor_lib=None):
        original_space = getattr(self.model.obs_space, "original_space", self.model.obs_space)

        if tensor_lib is not None and isinstance(packed_obs, np.ndarray):
            if tensor_lib == torch:
                packed_obs = convert_to_torch_tensor(packed_obs, self.device)
        obs = _unpack_obs(packed_obs, original_space, tensorlib = tensor_lib)
        return obs


    def postprocess_trajectory(self, sample_batch:SampleBatch, other_agent_batches=None, episode=None):
        '''
        This function is called after one trajectory (segment) is collected from one environment \n
        This function processes the collected sample (s,a,s',r) trajectories, and computes advantages \n
        The processed trajectory is ready to be used for policy update \n
        For on-policy algorithms, the computed logits are also included in the sample \n
        Logic can be added here to customize sample
        '''
        with torch.no_grad():
            if self.model_type != "MAPPO_model":
                if self.action_level == 'high':
                    return compute_async_gae_for_sample_batch(
                        self, sample_batch, other_agent_batches, episode
                    )
                else:
                    return super().postprocess_trajectory(sample_batch, other_agent_batches=None, episode=None)
            elif self.model_type == "MAPPO_model":
                batch = centralized_postprocess(
                    self, sample_batch, other_agent_batches, episode
                )
                return batch

    def _swap_ego_obs(self, obs):
        image = obs["image"]
        batch_size = image.shape[0]
        
        processed_images = image.copy()
        
        ego_layers = image[:, :, :, 0]  # Shape: (batch_size, height, width)
        player_pos_layers = image[:, :, :, 9]  # Shape: (batch_size, height, width)
        
        ego_positions = ego_layers > 0  # Shape: (batch_size, height, width)
        
        # Get player IDs at ego positions for all batch items
        original_ego_indices = []
        original_partner_indices = []
        
        for b in range(batch_size):
            ego_mask = ego_positions[b]
            if np.any(ego_mask):
                # Get the first ego position (assuming only one ego per batch)
                ego_coords = np.where(ego_mask)
                if len(ego_coords[0]) > 0:
                    original_ego_index = int(player_pos_layers[b, ego_coords[0][0], ego_coords[1][0]])
                    original_ego_indices.append(original_ego_index)
                    original_partner_indices.append(1 if original_ego_index == 2 else 0)
                else:
                    original_ego_indices.append(None)
                    original_partner_indices.append(None)
            else:
                original_ego_indices.append(None)
                original_partner_indices.append(None)
        
        all_player_positions = player_pos_layers > 0  # Shape: (batch_size, height, width)
        
        # Find other player positions (not current ego) for entire batch
        other_player_positions = all_player_positions & ~ego_positions
        
        # Set new ego layers for entire batch at once
        processed_images[:, :, :, 0] = other_player_positions.astype(image.dtype)
        
        return processed_images, original_ego_indices, original_partner_indices

    def _swap_ego_obs1(self, obs):
        image = obs["image"]
        batch_size = image.shape[0]
        processed_images = image.copy()
        original_ego_indices = []
        original_partner_indices = []
        
        for b in range(batch_size):
            # ego layer 0, player pos layer 9 (9 since added 1 to 8 when adding ego)
            ego_layer = image[b, :, :, 0]
            player_pos_layer = image[b, :, :, 9]
            
            # Find where the current ego agent is located
            current_ego_y, current_ego_x = np.where(ego_layer > 0)
            
            # Get the player ID at the ego position (should be 1 or 2)
            original_ego_index = None
            if len(current_ego_y) > 0:
                original_ego_index = int(player_pos_layer[current_ego_y[0], current_ego_x[0]])
                original_ego_indices.append(original_ego_index)
            else:
                original_ego_indices.append(None)
            
            # Find positions of all players
            player_positions = (player_pos_layer > 0)
            
            # Find position of current ego agent
            current_ego_positions = (ego_layer > 0)
            
            # Find position of the other player
            other_player_positions = player_positions & ~current_ego_positions
            
            # Create new ego layer with the other player position set to 1
            new_ego_layer = np.zeros_like(ego_layer)
            new_ego_layer[other_player_positions] = 1
            
            # Replace the current ego layer for this batch item
            processed_images[b, :, :, 0] = new_ego_layer

            if original_ego_index == 1:
                original_partner_indices.append(1)
            elif original_ego_index == 2:
                original_partner_indices.append(0)
 
        return processed_images, original_ego_indices, original_partner_indices

    def predict_partner_action(self, vae, z, obs):
        """
        Use VAE to predict partner action given latent z and observation.
        """
        # Expand z to match batch size
        batch_size = obs.shape[0]
        batch_z = z.unsqueeze(0).expand(batch_size, -1)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        #obs_tensor = obs_tensor.permute(0,3,1,2).contiguous()
        obs_tensor = obs_tensor.unsqueeze(1)

        # Use VAE decoder to predict action
        with torch.no_grad():
            logits = vae.decode_sequence(batch_z, obs_tensor, True)
            action_logits = logits[:, 0, :self.action_dim]
            #print("ACTION LOGITS", action_logits, "ACTION LOGTS LEN", len(action_logits))

        return action_logits

    def save_expert_weights(self, weights_data, file_path="expert_weights.jsonl"):
        import json
        import os
        """
        Append expert weights to a JSONL file.
        Each line contains a JSON object with timestep, timestamp, and weights.
        """
        # Convert any non-serializable types (like numpy arrays) to standard Python types
        serializable_weights = {}
        for cluster_id, weight in weights_data.items():
            # Convert cluster_id to string if it's not already
            key = str(cluster_id)
            # Convert weight to float if it's a numpy type
            value = float(weight) if hasattr(weight, "item") else weight
            serializable_weights[key] = value
        
        # Create a record with timestamp and weights
        record = {
            "weights": serializable_weights
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        # Append the record as a new line in the file
        with open(file_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def fixed_share_expforget(self, obs, partner_idx):
        learning_rate = self.lr
        alpha = self.alpha
        decay_factor = self.decay_factor

        losses = {}
        prev_gd_truth_action = obs["prev_action"]
        if np.all(prev_gd_truth_action == 0):
            #self.save_expert_weights(self.expert_weights, save_file)
            return
        prev_action_tensor = torch.tensor(prev_gd_truth_action, dtype=torch.float32, device=self.device)
        
        # Calculate losses for each expert
        for cluster_id, mvn in self.mvns.items():
            with torch.no_grad():
                action_logits = self.prev_decodes[cluster_id]
                prev_action_tensor = prev_action_tensor.to(action_logits.device)
                targets = prev_action_tensor.argmax(dim=-1)
                loss = cross_entropy(action_logits, targets,reduction='none')
                losses[cluster_id] = loss #.item()
        
        if self.decay_factor < 1.0:
            for cluster_id in self.cumulative_regret:
                self.cumulative_regret[cluster_id] *= decay_factor
        
        all_losses = torch.stack(list(losses.values()))
        min_losses = all_losses.min(dim=0)[0]

        valid_mask = torch.tensor(
            [not np.all(env == 0) for env in prev_gd_truth_action],
            device=min_losses.device
        )
                
        # Update regret with fresh observations
        #min_loss = min(losses.values())
        for cluster_id, loss in losses.items():
            per_step_regret = loss - min_losses
            per_step_regret *= valid_mask
            clipped_regret = torch.clamp(per_step_regret,max=1.0)
            self.cumulative_regret[cluster_id] = self.cumulative_regret[cluster_id].to(min_losses.device)
            self.cumulative_regret[cluster_id] += clipped_regret
        
        # Compute master weights using exponential weights algorithm
        master_weights = {}
        total_master_weight = torch.zeros(loss.shape[0], dtype=torch.float32, device=min_losses.device)
        for cluster_id in self.mvns.keys():
            master_weights[cluster_id] = torch.exp(-learning_rate * self.cumulative_regret[cluster_id])
            total_master_weight += master_weights[cluster_id]
        
        # Normalize master weights
        for cluster_id in self.mvns.keys():
            master_weights[cluster_id] /= total_master_weight
        
        num_experts = len(self.mvns)
        if num_experts == 1:
            return
        pool = sum(alpha * master_weights[cluster_id] for cluster_id in self.mvns.keys())
        
        for cluster_id in self.mvns.keys():
            # Each expert keeps (1-alpha) of its weight
            # And gets 1/(n-1) of the pool, excluding its own contribution
            self.expert_weights[cluster_id] = (1 - alpha) * master_weights[cluster_id] + \
                                            (1 / (num_experts - 1)) * (pool - alpha * master_weights[cluster_id])
        
        
        # Track expert selections for analytics
        self.episode_expert_selections.append(
            {cluster_id: weight for cluster_id, weight in self.expert_weights.items()}
        )

    def no_reg(self, obs, partner_idx):
        learning_rate = 0.1  # Adjust this value as needed
        losses = {}

        prev_gd_truth_action = obs["prev_action"]
        prev_action_tensor = torch.tensor(prev_gd_truth_action, dtype=torch.float32, device=self.device)
        for cluster_id, mvn in self.mvns.items():
            with torch.no_grad():
                action_logits = self.prev_decodes[cluster_id]
                loss = cross_entropy(action_logits.unsqueeze(0), prev_action_tensor.argmax().unsqueeze(0))

                loss = min(5.0, loss.item()) # bound per step regret
                losses[cluster_id] = loss
        
        # Update regret
        min_loss = min(losses.values())
        for cluster_id, loss in losses.items():
            # Update cumulative regret (difference between expert's loss and best expert's loss)
            self.cumulative_regret[cluster_id] += loss - min_loss
 

        
        # Update weights using exponential weights algorithm
        total_weight = 0.0
        for cluster_id in self.mvns.keys():
            # Compute weight using exponential of negative regret
            self.expert_weights[cluster_id] = np.exp(-learning_rate * self.cumulative_regret[cluster_id])
            total_weight += self.expert_weights[cluster_id]
        
        # Normalize weights
        for cluster_id in self.mvns.keys():
            self.expert_weights[cluster_id] /= total_weight
        
        # Track expert selections for analytics
        self.episode_expert_selections.append(
            {cluster_id: weight for cluster_id, weight in self.expert_weights.items()}
        )

    def sample_vae(self, partner_obs):
         # Make predictions for next step by sampling from each expert's distribution
        for cluster_id, mvn in self.mvns.items():
            z = mvn.sample()
            
            # Use VAE to predict partner action
            with torch.no_grad():
                predicted_action_logits = self.predict_partner_action(self.vae, z, partner_obs)
                self.prev_decodes[cluster_id] = predicted_action_logits
 
    def compute_actions_from_input_dict(self,
                                        input_dict: Dict[str, TensorType],
                                        explore: bool = None,
                                        timestep: int = None,
                                        episodes: Optional[List["Episode"]] = None,
                                        **kwargs) \
                                        -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        '''
        `input_dict`: {'obs':a batch of observation}, batch size depending on the environments running in parallel
        '''
        obs = convert_to_numpy(self._unpack_obs(input_dict['obs'], tensor_lib = torch))
        batch_size = obs["image"].shape[0] # num of parallel envs
        #print(obs["image"].shape, "=======OBS SHAPE IN COMPUTE ACTIONS")
        # initialize these here because of parallel envs
        if self.cumulative_regret is None:
            self.cumulative_regret = {cluster_id: torch.zeros(batch_size,dtype=torch.float32).to(device=self.device) for cluster_id in self.cluster_params.keys()}
        if self.expert_weights is None:
            self.expert_weights = {cluster_id: torch.ones(batch_size,dtype=torch.float32).to(device=self.device) * 1.0 / len(self.cluster_params) for cluster_id in self.cluster_params.keys()}

        # Swap ego perspective to get partner observation
        partner_img, ego_idx, partner_idx = self._swap_ego_obs(obs)

        processed_img = partner_img
        # may need this line
        #processed_img = torch.tensor(partner_img, dtype=torch.float32, device=device).unsqueeze(0)
        # Skip first timestep when we don't have previous predictions
        if self.prev_decodes:
            #self.no_reg(obs,partner_idx)
            self.fixed_share_expforget(obs, partner_idx)
        self.sample_vae(processed_img)

        cluster_ids = list(self.expert_weights.keys())
        weights = torch.stack([self.expert_weights[cid] for cid in cluster_ids]).to(device=self.device)
        selected_cluster_idx = torch.argmax(weights,dim=0)

        cluster_one_hot = torch.zeros((batch_size,len(self.cluster_params)), dtype=torch.float32, device=self.device)
        cluster_one_hot[torch.arange(batch_size),selected_cluster_idx] = 1.0

        obs_dict = obs.copy()  # Create a copy to avoid modifying the original
        
        # Add the cluster_id to the observation dictionary
        obs_dict['cluster_id'] = cluster_one_hot

        input_dict['obs'] = obs_dict
        
        # Log which cluster was selected for debugging/analysis
        #print(f"Selected cluster: {selected_cluster} with weight {weights[selected_cluster_idx]}")

        sampled_action, state_out, extra_fetches = super().compute_actions_from_input_dict(
            input_dict, explore, timestep, episodes=episodes, **kwargs)

        # Debug output shapes
#        print(f"sampled_action: type={type(sampled_action)}, shape={getattr(sampled_action, 'shape', 'no shape')}")
#        print(f"state_out: type={type(state_out)}, len={len(state_out)}")
#        print(f"extra_fetches keys: {list(extra_fetches.keys())}")
#
        
        # Add expert weights to extra_fetches for logging
        #extra_fetches["expert_weights"] = {k: float(v) for k, v in self.expert_weights.items()}
        #extra_fetches["selected_cluster"] = selected_cluster
        
        # Process high-level actions if needed
        if self.action_level == 'low':
            return (sampled_action, state_out, extra_fetches)
        else:
            obs = convert_to_numpy(self._unpack_obs(input_dict['obs'], tensor_lib = torch))
            action_stats = obs['action_stats']
            sample_mask = action_stats[np.arange(len(sampled_action)), sampled_action] == 1
            unfinished_action = np.argmax(action_stats, axis=1)
            processed_action = np.where(sample_mask, sampled_action, unfinished_action)
            input_dict[SampleBatch.ACTIONS] = processed_action
            extra_fetches["prev_action_finished"] = sample_mask


            return (processed_action, state_out, extra_fetches)

    def stats_fn(self, train_batch):
        '''
        Update the stats of local worker policies using train batch \n
        The function is called after each time loss function is called \n
        e.g. stats['var1'] = 1 \n
        The updated stats will be collected into the results with key 'learner_stats/policy_name/var1'
        '''
        stats = super().stats_fn(train_batch)
        return convert_to_numpy(stats)


    # doesn't matter so much for eval
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
        Returns:
            The PPO loss tensor given the input batch.
        """
        # print(train_batch['opponent_dist'])
        if self.model_type == "MAPPO_model":
            train_batch[SampleBatch.CUR_OBS] = self._unpack_obs(train_batch[SampleBatch.CUR_OBS], tensor_lib=torch)

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        advantages = train_batch[Postprocessing.ADVANTAGES]

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            if self.model_type == "MAPPO_model":
                value_fn_out = model.value_function(train_batch)
            else:
                value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            

            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

class HumanNoRegretPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.opponent_count = config['env_config'].get('n_agents') - 1
        self.pretrain_path = config.get("pretrained_model_path", None)
        self.model_type = config["model"].get("custom_model", "ClusterConditionedActorCritic")
        self.action_level = config["env_config"].get("action_level", 'high')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = config["obs_shape"]
        self.action_dim = config["action_dim"]
        #self.name = config.get("name")
        
        self.vae = load_vae(
            model_path= os.path.join(config["vae_path"], "encoder.pt"),
            obs_shape = config["obs_shape"],
            action_dim = config["action_dim"],
            latent_dim = config["latent_dim"],
            window_length =config["window_length"],
            horizon = config["vae_horizon"],
        )

        # these need to be the full list of cluster_params so we can sample from every cluster mean
        self.cluster_params = config["cluster_params"]
        #print("CLUSTER PARAMS", self.cluster_params)

        self.mvns = self._init_cluster(self.cluster_params)
    
        self.expert_weights = {cluster_id: 1.0 / len(self.cluster_params) for cluster_id in self.cluster_params.keys()}
        #print("EXPERT WEIGHTS", self.expert_weights)

        # Initialize cumulative regret
        self.cumulative_regret = {cluster_id: 0.0 for cluster_id in self.cluster_params.keys()}
        
        # Track which expert is selected at each timestep
        self.episode_expert_selections = []

        self.prev_decodes = {}


        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        
        # this should be a cluster conditioned critic
        self._load_pretrained_model(self.pretrain_path)
        

    def _init_cluster(self, cluster_params):
        # Create Gaussian distributions for each expert cluster
        mvns = {}
        for cluster_id, params in cluster_params.items():
            mean = torch.tensor(params['mean'], dtype=torch.float32, device=self.device)
            cov = torch.tensor(params['cov'], dtype=torch.float32, device=self.device)
            #eye = torch.eye(cov.shape[0], dtype=torch.float32, device=self.device)
            #cov = cov + eye * 1e-4  # Regularization
            mvns[cluster_id] = torch.distributions.MultivariateNormal(mean, cov)
        return mvns
    

    def _load_pretrained_model(self, import_path, model = None):
        """
        load pretrained model for the learning agent
        """
        if not import_path or import_path == 0:
            return
        checkpoint_path = Path(import_path).expanduser()
        print("==========LOADING FROM CHECKPOINT==========")
        print(checkpoint_path,"checkpoint path")
        print("NO REGRET LOADING")
        assert os.path.exists(checkpoint_path)
        with checkpoint_path.open('rb') as f:
            checkpoint = cloudpickle.load(f)
            state_dict = checkpoint['weights']
            filtered_state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(convert_to_torch_tensor(filtered_state_dict,device=self.device),
                                            strict=False)
            print("NO REGRET POLICY LOADED SUCCESSFULLY")

    @override(TorchPolicyV2)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        if self.model_type != 'MAPPO_model':
            # compute value with torch_mixin
            return super().extra_action_out(input_dict, state_batches, model, action_dist)
        elif self.model_type == "MAPPO_model":
            # agent do not have global info access in compute actions. Therefore not computing value here
            return {}


    def _unpack_obs(self, packed_obs, tensor_lib=None):
        original_space = getattr(self.model.obs_space, "original_space", self.model.obs_space)
        obs = _unpack_obs(packed_obs, original_space, tensorlib = tensor_lib)
        return obs


    def postprocess_trajectory(self, sample_batch:SampleBatch, other_agent_batches=None, episode=None):
        '''
        This function is called after one trajectory (segment) is collected from one environment \n
        This function processes the collected sample (s,a,s',r) trajectories, and computes advantages \n
        The processed trajectory is ready to be used for policy update \n
        For on-policy algorithms, the computed logits are also included in the sample \n
        Logic can be added here to customize sample
        '''
        with torch.no_grad():
            if self.model_type != "MAPPO_model":
                if self.action_level == 'high':
                    return compute_async_gae_for_sample_batch(
                        self, sample_batch, other_agent_batches, episode
                    )
                else:
                    return super().postprocess_trajectory(sample_batch, other_agent_batches=None, episode=None)
            elif self.model_type == "MAPPO_model":
                batch = centralized_postprocess(
                    self, sample_batch, other_agent_batches, episode
                )
                return batch

    def _swap_ego_obs(self, obs):
        image = obs["image"]
        batch_size = image.shape[0]
        processed_images = image.copy()
        original_ego_indices = []
        original_partner_indices = []
        
        # Process each batch item separately
        for b in range(batch_size):
            # Get layers for this batch item
            ego_layer = image[b, :, :, 0]
            player_pos_layer = image[b, :, :, 9]
            
            # Find where the current ego agent is located
            current_ego_y, current_ego_x = np.where(ego_layer > 0)
            
            # Get the player ID at the ego position (should be 1 or 2)
            original_ego_index = None
            if len(current_ego_y) > 0:
                original_ego_index = int(player_pos_layer[current_ego_y[0], current_ego_x[0]])
                original_ego_indices.append(original_ego_index)
            else:
                original_ego_indices.append(None)
            
            # Find positions of all players
            player_positions = (player_pos_layer > 0)
            
            # Find position of current ego agent
            current_ego_positions = (ego_layer > 0)
            
            # Find position of the other player
            other_player_positions = player_positions & ~current_ego_positions
            
            # Create new ego layer with the other player position set to 1
            new_ego_layer = np.zeros_like(ego_layer)
            new_ego_layer[other_player_positions] = 1
            
            # Replace the current ego layer for this batch item
            processed_images[b, :, :, 0] = new_ego_layer

            if original_ego_index == 1:
                original_partner_indices.append(1)
            elif original_ego_index == 2:
                original_partner_indices.append(0)
 
        return image, original_ego_indices, original_partner_indices

    def predict_partner_action(self, vae, z, obs):
        """
        Use VAE to predict partner action given latent z and observation.
        """
        # Expand z to match batch size
        batch_size = obs.shape[0]
        batch_z = z.unsqueeze(0).expand(batch_size, -1)

        #print(obs.shape,"OBS SHAPE")
        #print(batch_z, "z SHAPE")

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        #obs_tensor = obs_tensor.permute(0,3,1,2).contiguous()
        obs_tensor = obs_tensor.unsqueeze(1)

        # Use VAE decoder to predict action
        with torch.no_grad():
            logits = vae.decode_sequence(batch_z, obs_tensor, True)
            # Extract action logits (assuming 6 possible actions)
            action_logits = logits[0, 0, :self.action_dim]
            #print("ACTION LOGITS", action_logits, "ACTION LOGTS LEN", len(action_logits))

        return action_logits
    
    def fixed_share(self, obs, partner_idx):
        learning_rate = 0.1  # Adjust this value as needed
        alpha = 0.1  # Sharing parameter - adjust based on how quickly environment changes
        losses = {}
        prev_gd_truth_action = obs["prev_action"]
        #print(prev_gd_truth_action, "GD ACTION")
        prev_action_tensor = torch.tensor(prev_gd_truth_action, dtype=torch.float32, device=self.device)
        
        # Calculate losses for each expert
        for cluster_id, mvn in self.mvns.items():
            with torch.no_grad():
                action_logits = self.prev_decodes[cluster_id]
                loss = cross_entropy(action_logits.unsqueeze(0), prev_action_tensor.argmax().unsqueeze(0))
                losses[cluster_id] = loss.item()
        
        # Update regret
        min_loss = min(losses.values())
        for cluster_id, loss in losses.items():
            per_step_regret = loss - min_loss
            clipped_regret = min(1.0, per_step_regret)
            self.cumulative_regret[cluster_id] += clipped_regret

        #print(self.cumulative_regret, "cumulative regret")
        
        # Compute master weights using exponential weights algorithm
        master_weights = {}
        total_master_weight = 0.0
        for cluster_id in self.mvns.keys():
            master_weights[cluster_id] = np.exp(-learning_rate * self.cumulative_regret[cluster_id])
            total_master_weight += master_weights[cluster_id]
        
        # Normalize master weights
        for cluster_id in self.mvns.keys():
            master_weights[cluster_id] /= total_master_weight
        
        # Step 2: Apply Fixed Share - create pool of weights to be redistributed
        num_experts = len(self.mvns)
        pool = sum(alpha * master_weights[cluster_id] for cluster_id in self.mvns.keys())
        
        # Step 3: Distribute weights according to Fixed Share formula
        for cluster_id in self.mvns.keys():
            # Each expert keeps (1-alpha) of its weight
            # And gets 1/(n-1) of the pool, excluding its own contribution
            self.expert_weights[cluster_id] = (1 - alpha) * master_weights[cluster_id] + \
                                            (1 / (num_experts - 1)) * (pool - alpha * master_weights[cluster_id])

        #print("Expert weights", self.expert_weights)
        
        # Track expert selections for analytics
        self.episode_expert_selections.append(
            {cluster_id: weight for cluster_id, weight in self.expert_weights.items()}
        )

    def no_reg(self, obs, partner_idx):
        learning_rate = 0.1  # Adjust this value as needed
        losses = {}

        prev_gd_truth_action = obs["prev_action"]
        prev_action_tensor = torch.tensor(prev_gd_truth_action, dtype=torch.float32, device=self.device)
        #print(prev_gd_truth_action,"gd")
        for cluster_id, mvn in self.mvns.items():
            with torch.no_grad():
                action_logits = self.prev_decodes[cluster_id]
                loss = cross_entropy(action_logits.unsqueeze(0), prev_action_tensor.argmax().unsqueeze(0))

#                action_prob = self.prev_decodes[cluster_id][prev_gd_truth_action]
                #print(cluster_id,"clusterid")
                #print(prev_gd_truth_action,"gd")
#                loss = -1 * torch.log(action_prob + 1e-10)
                #print("probs",torch.softmax(action_logits, dim=0))
                #print(action_logits, "ACTION LOGITS")
                #print(np.argmax(action_logits))
                #print(loss, "LOSS")
                loss = min(5.0, loss.item()) # bound per step regret
                losses[cluster_id] = loss
        
        # Update regret
        min_loss = min(losses.values())
        for cluster_id, loss in losses.items():
            # Update cumulative regret (difference between expert's loss and best expert's loss)
            self.cumulative_regret[cluster_id] += loss - min_loss
        print(self.cumulative_regret, "cumulative regret")
        
        # Update weights using exponential weights algorithm
        total_weight = 0.0
        for cluster_id in self.mvns.keys():
            # Compute weight using exponential of negative regret
            self.expert_weights[cluster_id] = np.exp(-learning_rate * self.cumulative_regret[cluster_id])
            total_weight += self.expert_weights[cluster_id]
        
        # Normalize weights
        for cluster_id in self.mvns.keys():
            self.expert_weights[cluster_id] /= total_weight
        print("Expert weights", self.expert_weights)
        
        # Track expert selections for analytics
        self.episode_expert_selections.append(
            {cluster_id: weight for cluster_id, weight in self.expert_weights.items()}
        )

    def sample_vae(self, partner_obs):
         # Make predictions for next step by sampling from each expert's distribution
        for cluster_id, mvn in self.mvns.items():
            z = mvn.sample()
            
            # Use VAE to predict partner action
            with torch.no_grad():
                predicted_action_logits = self.predict_partner_action(self.vae, z, partner_obs)
                #print(predicted_action_logits, "predicted action logits")
                # Convert logits to action probabilities
                #predicted_action_probs = torch.softmax(predicted_action_logits, dim=0)
                #print(predicted_action_probs,"action probs")
                self.prev_decodes[cluster_id] = predicted_action_logits
 
    def compute_actions_from_input_dict(self,
                                        input_dict: Dict[str, TensorType],
                                        explore: bool = None,
                                        timestep: int = None,
                                        episodes: Optional[List["Episode"]] = None,
                                        **kwargs) \
                                        -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        '''
        `input_dict`: {'obs':a batch of observation}, batch size depending on the environments running in parallel
        '''
        obs = convert_to_numpy(self._unpack_obs(input_dict['obs'], tensor_lib = torch))
        
        # Swap ego perspective to get partner observation
        partner_img, ego_idx, partner_idx = self._swap_ego_obs(obs)

        processed_img = partner_img
        # may need this line
        #processed_img = torch.tensor(partner_img, dtype=torch.float32, device=device).unsqueeze(0)
        # Skip first timestep when we don't have previous predictions
        if self.prev_decodes:
            #print(self.prev_decodes, "prev decodes")
            #self.no_reg(obs,partner_idx)
            self.fixed_share(obs, partner_idx)
        self.sample_vae(processed_img)

        cluster_ids = list(self.expert_weights.keys())
        weights = [self.expert_weights[cid] for cid in cluster_ids]
        selected_cluster_idx = np.argmax(weights)
        selected_cluster = cluster_ids[selected_cluster_idx]

        #selected_cluster_idx = 0

        cluster_one_hot = np.zeros(len(self.cluster_params))
        cluster_one_hot[selected_cluster_idx] = 1.0

        # Now let's add this to the observation dictionary
        obs_dict = obs.copy()  # Create a copy to avoid modifying the original
        
        # Add the cluster_id to the observation dictionary
        obs_dict['cluster_id'] = cluster_one_hot
        
        # Now we need to flatten this modified observation
        #from ray.rllib.utils.spaces.space_utils import flatten_space, flatten_to_single_ndarray
        
        #flattened_obs = flatten_to_single_ndarray(obs_dict)
        
        # Replace the observation in the input_dict
        #input_dict['obs'] = flattened_obs
        input_dict['obs'] = obs_dict
        #print(input_dict['obs'],"INPUT DICT OBS")
        
        # Log which cluster was selected for debugging/analysis
        #print(f"Selected cluster: {selected_cluster} with weight {weights[selected_cluster_idx]}")
        

        sampled_action, state_out, extra_fetches = super().compute_actions_from_input_dict(
            input_dict, explore, timestep, episodes=episodes, **kwargs)

        # Debug output shapes
#        print(f"sampled_action: type={type(sampled_action)}, shape={getattr(sampled_action, 'shape', 'no shape')}")
#        print(f"state_out: type={type(state_out)}, len={len(state_out)}")
#        print(f"extra_fetches keys: {list(extra_fetches.keys())}")
#
        
        # Add expert weights to extra_fetches for logging
        #extra_fetches["expert_weights"] = {k: float(v) for k, v in self.expert_weights.items()}
        #extra_fetches["selected_cluster"] = selected_cluster
        
        # Process high-level actions if needed
        if self.action_level == 'low':
            return (sampled_action, state_out, extra_fetches)
        else:
            obs = convert_to_numpy(self._unpack_obs(input_dict['obs'], tensor_lib = torch))
            action_stats = obs['action_stats']
            sample_mask = action_stats[np.arange(len(sampled_action)), sampled_action] == 1
            unfinished_action = np.argmax(action_stats, axis=1)
            processed_action = np.where(sample_mask, sampled_action, unfinished_action)
            input_dict[SampleBatch.ACTIONS] = processed_action
            extra_fetches["prev_action_finished"] = sample_mask
            return (processed_action, state_out, extra_fetches)

    def stats_fn(self, train_batch):
        '''
        Update the stats of local worker policies using train batch \n
        The function is called after each time loss function is called \n
        e.g. stats['var1'] = 1 \n
        The updated stats will be collected into the results with key 'learner_stats/policy_name/var1'
        '''
        stats = super().stats_fn(train_batch)
        return convert_to_numpy(stats)


    # doesn't matter so much for eval
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
        Returns:
            The PPO loss tensor given the input batch.
        """
        # print(train_batch['opponent_dist'])
        if self.model_type == "MAPPO_model":
            train_batch[SampleBatch.CUR_OBS] = self._unpack_obs(train_batch[SampleBatch.CUR_OBS], tensor_lib=torch)

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        advantages = train_batch[Postprocessing.ADVANTAGES]

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            if self.model_type == "MAPPO_model":
                value_fn_out = model.value_function(train_batch)
            else:
                value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            

            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss