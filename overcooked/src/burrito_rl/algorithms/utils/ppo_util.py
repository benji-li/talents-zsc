import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy
from typing import Dict, Optional
from ray.rllib.utils.typing import AgentID
from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

def concat_samples(policy, opponent_keys, opponents_batch):
    """
    Concatenate sample batches from multiple opponent agents.
    
    - For each *sub-key* in `obs`, produce shape: [T, num_opponents * subkey_dim].
    - For the rest of the keys (e.g. "actions"), produce shape: [T, num_opponents, ...].

    Args:
        policy: Object providing `_unpack_obs(...)`, which returns a dict of NumPy arrays
                if the obs is itself a dictionary. Each sub-array has shape (subkey_dim,).
        opponent_keys (list[str]): Keys to process, e.g. ['obs', 'actions', ...].
        opponents_batch (list[dict]): Each element is an opponent's data dictionary,
                                      with entries like:
                                         opponents_batch[i]['obs'] = [obs_t0, obs_t1, ..., obs_t{T-1}]
                                         opponents_batch[i]['actions'] = [act_t0, act_t1, ..., act_t{T-1}]
                                      etc.

    Returns:
        final_batch (dict):
            - For each sub-key in 'obs': final_batch['obs'][sub_key] has shape [T, num_opponents * subkey_dim].
            - For each non-obs key: final_batch[key] has shape [T, num_opponents, ...].
    """
    num_opponents = len(opponents_batch)
    # We assume all opponents have the same number of timesteps for each key
    num_timesteps = len(opponents_batch[0][opponent_keys[0]])

    # We'll store subkeys of 'obs' in a separate dict of lists:
    #   obs_accumulators[sub_key] -> list of 1D arrays (one per timestep).
    obs_accumulators = {}  # will map sub_key -> list of shape (num_opponents*subkey_dim,)
    # We'll store the rest of the keys (e.g. 'actions') normally in rest_data
    rest_data = {k: [] for k in opponent_keys if k != "obs"}

    # 1) Iterate over timesteps
    for t in range(num_timesteps):
        # 1a) Handle the 'obs' key
        if "obs" in opponent_keys:
            # For each opponent, unpack the obs at time t
            # -> returns a dict of sub-keys (e.g. {"obs_key1": arr1, "obs_key2": arr2, ...})
            obs_per_opponent = [
                policy._unpack_obs(opponents_batch[i]["obs"][t].reshape(1,-1), tensor_lib=np)
                for i in range(num_opponents)
            ]

            # If there's a chance that different opponents have different sub-keys,
            # you'll need a consistent set of sub-keys. For simplicity, we assume
            # they match what's in obs_per_opponent[0].
            subkeys = obs_per_opponent[0].keys()

            for subk in subkeys:
                # Gather subk data from each opponent => shape (num_opponents, subkey_dim)
                stacked_subk = np.array([obs[subk].squeeze(0) for obs in obs_per_opponent])
                # General reshaping: from (num_opponents, ..., last_dim) to (..., num_opponents * last_dim)
                num_opponents = stacked_subk.shape[0]
                last_dim = stacked_subk.shape[-1]
                middle_dims = stacked_subk.shape[1:-1]

                # If there are only 2 dimensions like (1, 27), then middle_dims will be empty ()
                if stacked_subk.ndim == 2:
                    # Special case: shape (num_opponents, last_dim) => (num_opponents * last_dim,)
                    flattened_subk = stacked_subk.reshape(num_opponents * last_dim)
                else:
                    # Transpose to (..., num_opponents, last_dim)
                    reshaped_subk = stacked_subk.transpose(*range(1, stacked_subk.ndim - 1), 0, stacked_subk.ndim - 1)
                    # Then reshape to (..., num_opponents * last_dim)
                    flattened_subk = reshaped_subk.reshape(*middle_dims, num_opponents * last_dim)

                # Append to the accumulator list for that subk
                if subk not in obs_accumulators:
                    obs_accumulators[subk] = []
                obs_accumulators[subk].append(flattened_subk)

        # 1b) Handle the rest of the keys
        for key in rest_data:
            # shape => (num_opponents, ...)
            stacked_value = np.array([
                opponents_batch[i][key][t] for i in range(num_opponents)
            ])
            rest_data[key].append(stacked_value)

    # 2) Convert the obs_accumulators lists into final arrays => shape [T, num_opponents * subkey_dim]
    final_batch = {key:{} for key in opponent_keys}
    for subk, data_list in obs_accumulators.items():
        final_batch['obs'][subk] = np.array(data_list)
        # final_batch[subk].shape is (T, num_opponents * subkey_dim)

    # 3) Convert rest_data to shape => [T, num_opponents, ...]
    for key, data_list in rest_data.items():
        final_batch[key] = np.array(data_list)
        # final_batch[key].shape is (T, num_opponents, ...)

    return final_batch


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


def centralized_postprocess(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    
    if other_agent_batches:
        opponent_batch = [item[1] for item in list(other_agent_batches.values())]
    else:
        opponent_batch = [sample_batch.copy() for _ in range(policy.opponent_count)]
    
    opponents_batch = concat_samples(policy, opponent_keys=[SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.ACTION_DIST_INPUTS], opponents_batch=opponent_batch)
    sample_batch[OPPONENT_OBS] = opponents_batch[SampleBatch.CUR_OBS]
    sample_batch[OPPONENT_ACTION] = opponents_batch[SampleBatch.ACTIONS]

    input_batch = sample_batch.copy()
    obs = policy._unpack_obs(sample_batch[SampleBatch.CUR_OBS], tensor_lib = np)
    input_batch['obs'] = obs

    # manually compute vf_preds with centralized information here
    sample_batch[SampleBatch.VF_PREDS] = (
        policy.model.value_function(
            convert_to_torch_tensor(input_batch, policy.device)
        ).detach().cpu().numpy()
    )

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    if policy.action_level == 'high':
        return compute_async_advantages(
            sample_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True),
        )
    else:
        return compute_advantages(
            sample_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic = policy.config.get("use_critic", True),
        )


def compute_async_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds Async GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._value(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_async_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    # print("processed batch:", batch)
    return batch


def discounted_sum_of_rewards(rs, discount):
    """rs: 1D array of rewards; discount: gamma"""
    s = 0.0
    for r in reversed(rs):
        s = r + discount * s
    return s



def compute_async_advantages(
    sample_batch: SampleBatch,
    last_r: float,
    gamma: float,
    lam: float,
    use_gae: bool = True,
    use_critic: bool = True,
    debug = False
) -> SampleBatch:
    """Compute chunk-based (asynchronous) GAE advantages for a SampleBatch.

    This function identifies consecutive steps that took the same action
    (and were not done) and compresses them into a single "chunk" transition.
    Each chunk i has:
      - start index s_i
      - end index e_i (excluded)
      - chunk length k = e_i - s_i
      - compressed reward r_c = sum_{t=0}^{k-1} (gamma^t * r_{s_i+t})
      - delta_i = r_c + (gamma^k)*V_{e_i} - V_{s_i}
    Then, GAE is computed over these chunks in a reverse pass:
      adv_i = delta_i + (gamma^k * lam^k)*adv_{i+1}
    (similar to standard per-timestep GAE, but applying exponent k for each chunk).

    Args:
        sample_batch: The original SampleBatch (with per-step transitions).
        last_r: Value function estimate for the final next state if truncated,
            or 0.0 if the episode ended.
        gamma: Discount factor.
        lam: GAE(lambda) parameter.
        use_gae: Whether to use GAE or not. If False, advantages are just
            discounted returns minus value predictions.
        use_critic: Whether to use critic (value function) for advantage
            calculation. If False, advantages are just rewards.

    Returns:
        A new compressed SampleBatch containing one record per chunk. The
        fields "advantages" and "value_targets" will be added, containing
        the GAE advantages and discounted returns, respectively.
    """

    # -------------------------------------------------------
    # Step 1: Extract arrays from the sample_batch for easier handling.
    # ------------------------------------------------------
    rewards = sample_batch[SampleBatch.REWARDS]
    dones = sample_batch[SampleBatch.DONES]
    vpreds = (
        sample_batch[SampleBatch.VF_PREDS]
        if use_critic and SampleBatch.VF_PREDS in sample_batch
        else np.zeros_like(rewards, dtype=np.float32)
    )
    actions = sample_batch[SampleBatch.ACTIONS]
    prev_actions_done = sample_batch["prev_action_finished"]
    # For convenience, create a np.array of the same length for the next state's value
    # We'll fill it up in chunk-based logic. For final chunk, if truncated, we use last_r.
    length = len(rewards)
    if debug:
        print("in compute async advantage: ")
        print("reward: ", rewards, len(rewards))
        print("actions: ", actions, len(actions))
        print("vf preds: ", vpreds, len(vpreds))
        print("last r: ", last_r)

    # -------------------------------------------------------
    # Step 2: Identify chunk boundaries
    # -------------------------------------------------------
    # We'll break whenever:
    #   - The action changes from step t to t+1, OR
    #   - dones[t] == True (which means the episode ended after step t).
    chunk_starts = []
    chunk_ends = []
    start_idx = 0

    for i in range(1, length):
        # Break the chunk if we are done at i-1 OR the action changed
        if dones[i-1] or prev_actions_done[i]:
            chunk_starts.append(start_idx)
            chunk_ends.append(i)
            start_idx = i
    # Last chunk from 'start_idx' to 'length'
    chunk_starts.append(start_idx)
    chunk_ends.append(length)

    # -------------------------------------------------------
    # Step 3: Build a new compressed buffer for all chunk-level data
    # -------------------------------------------------------
    # We will store arrays for each chunk, then build a new SampleBatch at the end.
    chunk_observations = []
    chunk_nextobservations = []
    chunk_actions = []
    chunk_rewards = []
    chunk_dones = []
    chunk_vpreds = []
    chunk_nextvpreds = []  # value at the next state after chunk
    chunk_advantages = []
    chunk_value_targets = []

    # We'll also copy over any other fields from the "start" of each chunk,
    # such as "obs", "logp", etc., to keep them consistent. We'll store them
    # in a dictionary of lists first.
    # For example, if you have additional fields you want to preserve in the
    # compressed batch, do that here.
    other_columns = {}
    for key in sample_batch.keys():
        # We'll ignore the typical keys that we'll handle ourselves
        if key in [
            SampleBatch.OBS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
            SampleBatch.VF_PREDS,
            SampleBatch.NEXT_OBS,
        ]:
            continue
        # We'll store only the start-of-chunk item.
        other_columns[key] = []
        if type(sample_batch[key]) == dict:
            other_columns[key] = {}
            for sample_key in sample_batch[key]:
                other_columns[key][sample_key] = []


    # First pass: collect chunk data, ignoring advantage logic for now
    # We compute the compressed reward for each chunk, store relevant info,
    # but not advantage/returns yet.
    chunk_count = len(chunk_starts)
    for ci in range(chunk_count):
        s_i = chunk_starts[ci]
        e_i = chunk_ends[ci]
        k = e_i - s_i  # chunk length
        # -- Compressed reward:
        #    r_c = sum_{t=0 to k-1} gamma^t * r[s_i + t]
        rewards_slice = rewards[s_i:e_i]
        r_c = discounted_sum_of_rewards(rewards_slice, gamma)

        # -- Next value:
        #    If e_i < length => next state value = vpreds[e_i]
        #    If e_i == length => next state value = last_r (truncated or done).
        if e_i < length:
            next_val = vpreds[e_i]
        else:
            # we reached the end of the entire batch
            next_val = last_r

        # Prepare chunk arrays
        # Observations: just take the start-of-chunk
        chunk_observations.append(sample_batch[SampleBatch.OBS][s_i])
        chunk_actions.append(actions[s_i])
        chunk_rewards.append(r_c)
        chunk_nextobservations.append(sample_batch[SampleBatch.NEXT_OBS][e_i-1])

        # The done flag for the chunk is the final step's done
        # (it might be True or False).
        chunk_dones.append(dones[e_i - 1])  # e_i-1 is last step in chunk

        chunk_vpreds.append(vpreds[s_i])
        chunk_nextvpreds.append(next_val)

        # Copy over other fields from the start index of the chunk
        for key, buff in other_columns.items():
            if type(sample_batch[key]) == dict:
                for sample_key in sample_batch[key]:
                    buff[sample_key].append(sample_batch[key][sample_key][s_i])
            else:
                buff.append(sample_batch[key][s_i])
    # -------------------------------------------------------
    # Step 4: Compute GAE (or discounted returns) chunk-by-chunk
    #         in a backward pass.
    # -------------------------------------------------------
    chunk_advantages = np.zeros(chunk_count, dtype=np.float32)
    chunk_deltas = np.zeros(chunk_count, dtype=np.float32)
    # We'll compute the advantage from the end -> start
    last_adv = 0.0

    for ci in reversed(range(chunk_count)):
        s_i = chunk_starts[ci]
        e_i = chunk_ends[ci]
        k = e_i - s_i
        v_i = chunk_vpreds[ci]
        next_v = chunk_nextvpreds[ci]
        r_c = chunk_rewards[ci]  # the compressed reward


        # delta = r_c + gamma^k * next_v - v_i
        chunk_deltas[ci] = r_c + (gamma ** k) * next_v - v_i
        if debug:
            print("compute each delta")
            print("r_c: ", r_c)
            print("v': ", next_v)
            print("v: ", v_i)
            print("delta: ", chunk_deltas[ci])

        if use_gae:
            # advantage[ci] = delta + gamma^k * lam^k * advantage[ci+1]
            chunk_advantages[ci] = chunk_deltas[ci] + (gamma ** k) * (lam ** k) * last_adv
            if debug:
                print("compute each advantage")
                print("delta: ", chunk_deltas[ci])
                print("A': ", last_adv)
                print("A: ", chunk_advantages[ci])
            last_adv = chunk_advantages[ci]
        else:
            # If not using GAE, advantage is just delta; then we do not do
            # the partial bootstrapping trick with subsequent advantage.
            chunk_advantages[ci] = chunk_deltas[ci]
            last_adv = 0.0  # reset for next iteration

    # Now we have the advantage for each chunk. We can also compute the
    # "value_target" for each chunk: delta + current value
    chunk_value_targets = chunk_deltas + np.array(chunk_vpreds, dtype=np.float32)

    # -------------------------------------------------------
    # Step 5: Build the final compressed SampleBatch
    # -------------------------------------------------------
    # Note that for a standard PPO, we want:
    #   - SampleBatch.OBS
    #   - SampleBatch.ACTIONS
    #   - SampleBatch.REWARDS
    #   - SampleBatch.DONES
    #   - SampleBatch.VF_PREDS
    #   - "advantages"
    #   - "value_targets" (sometimes called "returns")
    # etc.
    compressed_data = {
        SampleBatch.OBS: np.array(chunk_observations, dtype=sample_batch[SampleBatch.OBS].dtype),
        SampleBatch.ACTIONS: np.array(chunk_actions, dtype=sample_batch[SampleBatch.ACTIONS].dtype),
        SampleBatch.REWARDS: np.array(chunk_rewards, dtype=sample_batch[SampleBatch.REWARDS].dtype),
        SampleBatch.DONES: np.array(chunk_dones, dtype=sample_batch[SampleBatch.DONES].dtype),
        SampleBatch.NEXT_OBS: np.array(chunk_nextobservations, dtype=sample_batch[SampleBatch.NEXT_OBS].dtype),
        SampleBatch.VF_PREDS: np.array(chunk_vpreds, dtype=sample_batch[SampleBatch.VF_PREDS].dtype),
        Postprocessing.ADVANTAGES: chunk_advantages,
        Postprocessing.VALUE_TARGETS: chunk_value_targets,
    }

    # Add back other columns
    for key, buff in other_columns.items():
        if type(buff) == dict:
            compressed_data[key] = {}
            for sample_key in buff:
                compressed_data[key][sample_key] = np.array(buff[sample_key], dtype=sample_batch[key][sample_key].dtype)
        else:
            compressed_data[key] = np.array(buff, dtype=sample_batch[key].dtype)

    # If needed/desired, you can also store NEXT_OBS for the chunk. Typically for
    # on-policy PPO, we only need OBS, but if you want to do off-policy
    # or debugging, you could store the last state's obs from the chunk:
    # e.g. compressed_data[SampleBatch.NEXT_OBS] = ...
    #   sample_batch[SampleBatch.OBS][chunk_ends[ci]] if chunk_ends[ci] < length else final_obs

    # Finally, create a new SampleBatch with the compressed data.
    compressed_batch = SampleBatch(compressed_data)

    if debug:
        print("compressed r: ", compressed_batch[SampleBatch.REWARDS])
        print("compressed a: ", compressed_batch[SampleBatch.ACTIONS])
        print("compressed v: ", compressed_batch[SampleBatch.VF_PREDS])
        print("compreesed d: ", chunk_deltas)
        print("compressed v' ", compressed_batch[Postprocessing.VALUE_TARGETS])
        print("compressed A: ", compressed_batch[Postprocessing.ADVANTAGES])

    return compressed_batch
