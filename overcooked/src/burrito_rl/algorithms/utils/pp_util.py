from typing import Dict, Tuple
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from burrito_rl.algorithms.callbacks import LoggingCallbacks
import torch
import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing


class MEPCallback(LoggingCallbacks):

    def on_learn_on_batch(
            self, 
            *, 
            policy: Policy, 
            train_batch: SampleBatch, 
            result: Dict, **kwargs) -> None:
        # 1) Call super() first (optional if you have other logic there)
        super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)

        # 2) If your postprocess_fn put a "population_entropy" key on the batch, grab & average it
        if "population_entropy" in train_batch:
            # train_batch["population_entropy"] is a numpy array of shape [B]
            avg_pe = float(train_batch["population_entropy"].mean())

            # 3) Inject it into result["population_entropy"]
            #    RLlib will pick up anything you put there and surface it in TensorBoard/logs :contentReference[oaicite:0]{index=0}
            result["population_entropy"] = avg_pe



def population_entropy(
    worker: RolloutWorker,
    train_batch: MultiAgentBatch
) -> MultiAgentBatch:
    # Grab the dict of policy_id → SampleBatch
    policy_batches = train_batch.policy_batches
    policy_ids = list(policy_batches.keys())

    for pid, batch in policy_batches.items():
        # 1) Get this batch’s observations
        obs = batch[SampleBatch.CUR_OBS]  # shape [B, ...]; change to OBS_EMBEDS if needed
        device = worker.policy_map[pid].device
        obs_t = torch.from_numpy(obs).to(device)

        # 2) Compute each policy’s action‐probabilities on these obs
        probs_list = []
        with torch.no_grad():
            for pid in policy_ids:
                model = worker.policy_map[pid].model
                out = model({"obs": obs_t})
                # Depending on your model, out may be a dict or (logits, state)
                logits = out["logits"] if isinstance(out, dict) else out[0]
                probs = torch.softmax(logits, dim=-1)  # [B, A]
                probs_list.append(probs)

        # 3) Form the population‐average distribution
        population_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # [B, A]

        # 4) Compute Shannon entropy per time‐step: H = -sum p*log p
        eps = 1e-8
        population_entropy = - (population_probs * torch.log(population_probs + eps)).sum(dim=-1)  # [B]
        pe = population_entropy.cpu().numpy()

        # 5) Convert to numpy bonus and scale
        if worker.policy_config['alg'] == 'MEP':
            # How strongly to weight the entropy bonus
            coeff = worker.policy_config["mep_coeff"]
            pe_bonus = pe * coeff

            # 6) Shape rewards (+ advs & value‐targets if they exist)
            batch[SampleBatch.REWARDS] = batch[SampleBatch.REWARDS] + pe_bonus
            if Postprocessing.ADVANTAGES in batch:
                batch[Postprocessing.ADVANTAGES]     = batch[Postprocessing.ADVANTAGES] + pe_bonus
                batch[Postprocessing.VALUE_TARGETS]  = batch[Postprocessing.VALUE_TARGETS] + pe_bonus

        batch["population_entropy"] = pe

    return train_batch



import random

def population_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if not hasattr(episode, "policy_to_use"):
        policy_ids = list(worker.policy_map.keys())
        episode.policy_to_use = random.choice(policy_ids)
    return episode.policy_to_use

def population_rllib_config(trainer, Config:dict) -> dict:

    Config["callbacks"] = MEPCallback #if Config['alg'] == 'MEP' else LoggingCallbacks 

    policy_ids = [f"pol{i+1}" for i in range(Config["n_population"])]
    Config["multiagent"] = {
        "policies": {
            pid: (
                None,  # policy spec
                Config["env_config"]["observation_space"],  # observation spec
                Config["env_config"]["action_space"],       # action spec
                {
                    "name": pid,
                    "framework": "torch",
                    "model": Config["model"],
                    "pretrained_model_path": Config["pretrained_model_path"][i],
                },
            )
            for i, pid in enumerate(policy_ids)
        },
        # Use the top-level function instead of a lambda
        "policy_mapping_fn": population_policy_mapping_fn,
        "policies_to_train": policy_ids,
    }
    return Config


def process_MEP_train_batch(trainer, train_batch):
    return population_entropy(trainer.workers.local_worker(), train_batch)
