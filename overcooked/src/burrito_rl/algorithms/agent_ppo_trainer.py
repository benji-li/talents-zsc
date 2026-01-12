import logging
import functools

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.logger import Logger
from ray.util.debug import log_once

from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

from ray.rllib.evaluation.metrics import (
    collect_episodes,
)
from ray.rllib.evaluation.worker_set import WorkerSet


logger = logging.getLogger(__name__)


from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from burrito_rl.algorithms.agent_ppo_policy import PPOPolicy
from burrito_rl.algorithms.callbacks import LoggingCallbacks



def ppo_policy_mapping_fn(aid, **kwargs):
    return "pol" + str(aid + 1)

def ppo_rllib_config(Config:dict) -> dict:
    Config["callbacks"] = LoggingCallbacks
    Config["multiagent"] = {
        "policies": {
            "pol" + str(i+1): (
                None,  # policy spec
                Config["env_config"]["observation_space"],  # observation spec
                Config["env_config"]["action_space"],       # action spec
                {
                    "name": "pol" + str(i+1),
                    "framework": "torch",
                    "model": Config["model"],
                    "pretrained_model_path": Config["pretrained_model_path"][i],
                },
            )
            for i in range(Config["env_config"]["n_agents"])
        },
        # Use the top-level function instead of a lambda
        "policy_mapping_fn": ppo_policy_mapping_fn,
        "policies_to_train": [
            f'pol{i+1}' for i in range(Config["env_config"]["n_agents"])
        ],
        # "policies_to_train": ['pol1']
    }
    return Config



class PPOPolicyTrainer(PPO):

    def get_config(self, config:dict=None):
        base = ppo_rllib_config(config)
        return base


    def __init__(self,
        config:dict = None,
        env=None,  # deprecated arg
        logger_creator = None,
        **kwargs,
    ):
        base = self.get_config(config)
        ppo_conf = PPOConfig.from_dict(base)
        super().__init__(
            config = ppo_conf.copy(),
            env=env,
            logger_creator=logger_creator,
            **kwargs
        )

    def get_default_policy_class(self, config):
        return PPOPolicy

    
    def process_train_batch(self, train_batch):
        return train_batch


    def train_with_batch(self,train_batch):
        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        # print(train_batch)
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        
        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        for pol_id in train_batch.policy_batches.keys():
            self._counters[f'{pol_id}_num_steps_trained'] += train_batch[pol_id].count

        train_batch = self.process_train_batch(train_batch)

        return self.train_with_batch(train_batch)

    def _sync_global_cluster_stats(self):
        """Collect and broadcast global cluster statistics across all workers"""
        from collections import defaultdict
        import numpy as np
        from burrito_rl.algorithms.utils.vaebr_util import VAEBRCallback
        
        print("Syncing global cluster stats at iteration", self._iteration)
        # 1. Collect episode rewards from all workers
        all_rewards = defaultdict(list)
        
        def collect_rewards(worker):
            if isinstance(worker.callbacks, VAEBRCallback):
                return dict(worker.callbacks.episode_rewards)
            return {}
        
        all_worker_rewards = self.workers.foreach_worker(collect_rewards)
        
        # Aggregate
        for worker_rewards in all_worker_rewards:
            for cluster_id, rewards in worker_rewards.items():
                all_rewards[cluster_id].extend(rewards)
        
        # 2. Calculate global statistics
        max_history = self.config.get("num_history", 100)
        global_cluster_vars = {}
        global_cluster_scores = {}
        
        for cluster_id, rewards in all_rewards.items():
            recent_rewards = rewards[-max_history:]
            print(recent_rewards, f"recent rewards for cluster {cluster_id}")
            
            if len(recent_rewards) > 1:
                global_cluster_scores[cluster_id] = float(np.mean(recent_rewards))
                global_cluster_vars[cluster_id] = float(np.var(recent_rewards))
            elif len(recent_rewards) == 1:
                global_cluster_scores[cluster_id] = float(recent_rewards[0])
                global_cluster_vars[cluster_id] = 0.0
        
        # 3. Directly update worker.global_vars (avoids policy updates)
        if global_cluster_vars:
            def update_cluster_stats(worker):
                # Direct update to avoid triggering policy.on_global_var_update()
                worker.global_vars["cluster_vars"] = global_cluster_vars
                worker.global_vars["cluster_scores"] = global_cluster_scores
            
            self.workers.foreach_worker(update_cluster_stats)
            
            if self.config.get("log_global_stats", False):
                logger.info(f"Synced cluster stats: {list(global_cluster_scores.keys())}")
            print(f"Synced cluster stats: {list(global_cluster_scores.keys())}")
            print(global_cluster_vars, "global cluster vars")
       
    def _compile_iteration_results(self, *, episodes_this_iter, step_ctx, iteration_results=None):
        result = super()._compile_iteration_results(episodes_this_iter=episodes_this_iter, step_ctx=step_ctx, iteration_results=iteration_results)
        for c in self._counters.keys():
            if c in [
                NUM_AGENT_STEPS_SAMPLED,
                NUM_AGENT_STEPS_TRAINED,
                NUM_ENV_STEPS_SAMPLED,
                NUM_ENV_STEPS_TRAINED,
            ]:
                continue
            else:
                # some keys are not automatiaclly included in PPOTrainer's global _counter, include them here
                result[c] = self._counters[c]
        return result

