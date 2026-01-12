
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


class PPOPolicy(PPOTorchPolicy):
    '''
    PPO policy built based on the PPOTorchPolicy class \n
    reserved necessary interfaces for customization
    '''
    def __init__(self, observation_space, action_space, config):
        self.opponent_count = config['env_config'].get('n_agents') - 1
        self.pretrain_path = config.get("pretrained_model_path", None)
        self.model_type = config["model"].get("custom_model", "IPPO_model")
        self.action_level = config["env_config"].get("action_level", 'high')
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        print("init ppo policy")
        
        print(self.pretrain_path,"pretrain path")
        self._load_pretrained_model(self.pretrain_path)
        


    def _load_pretrained_model(self, import_path, model = None):
        """
        load pretrained model for the learning agent
        """
        if not import_path or import_path == 0:
            return
        checkpoint_path = Path(import_path).expanduser()
        print("==========LOADING FROM CHECKPOINT==========")
        print(checkpoint_path,"checkpoint path")
        assert os.path.exists(checkpoint_path)
        with checkpoint_path.open('rb') as f:
            checkpoint = cloudpickle.load(f)
            state_dict = checkpoint['weights']
            filtered_state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(convert_to_torch_tensor(filtered_state_dict,device=self.device),
                                            strict=False)

        print("LOADED PPO POLICY")

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
                if self.action_level == 'high' and not getattr(self, '_eval_mode', False):
                    return compute_async_gae_for_sample_batch(
                        self, sample_batch, other_agent_batches, episode
                    )
                else:
                    print("calling super postprocess")
                    return super().postprocess_trajectory(sample_batch, other_agent_batches=None, episode=None)
            elif self.model_type == "MAPPO_model":
                batch = centralized_postprocess(
                    self, sample_batch, other_agent_batches, episode
                )
                return batch
    

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
        # Print policy info for debugging
        #print(f"\n===== ACTION DEBUG =====")
        #print(f"Policy: {self.config.get('name', 'unknown')}")
        #print(f"Model type: {self.model_type}")
        
        sampled_action, state_out, extra_fetches = super().compute_actions_from_input_dict(input_dict, explore, timestep, episodes=episodes, **kwargs)
        # Debug output shapes
#        print(f"sampled_action: type={type(sampled_action)}, shape={getattr(sampled_action, 'shape', 'no shape')}")
#        print(f"state_out: type={type(state_out)}, len={len(state_out)}")
#        print(f"extra_fetches keys: {list(extra_fetches.keys())}")
#
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

#            def reduce_mean_valid(t):
#                print(t.shape, mask.shape, num_valid, "===============================")
#                return torch.sum(t[mask]) / num_valid

            def reduce_mean_valid(t):
                if mask.shape != t.shape:
                    mask_reshaped = mask.view(-1)
                    t = t.view(-1)
                else:
                    mask_reshaped = mask

                if mask_reshaped.sum() == 0:
                    return torch.tensor(0.0, device=t.device)
                return t[mask_reshaped].mean()

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
