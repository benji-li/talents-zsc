from collections import defaultdict

def vae_cluster_br_eval(algorithm, eval_workers, debug=False):
    if eval_workers is None:
        workers = algorithm.workers
        print("Warning: No eval workers provided, using training workers for evaluation.")
    else:
        # old code: syncs all weights
        # eval_workers.sync_weights(
        #     from_worker=algorithm.workers.local_worker()
        # )
        # workers = eval_workers
        # print("Creating eval workers for evaluation.")
        training_worker = algorithm.workers.local_worker()
        eval_worker = eval_workers.local_worker()
        workers = eval_workers

        policy_sync_map = {
            'polBR': 'polBREval',
        }

    for train_id, eval_id in policy_sync_map.items():
        if train_id in training_worker.policy_map and eval_id in eval_worker.policy_map:
            weights = training_worker.policy_map[train_id].get_weights()
            eval_worker.policy_map[eval_id].set_weights(weights)

            print(f"Synchronized weights from {train_id} to {eval_id}.")

    # save the original mapping fn and policy map
    local_worker = workers.local_worker()
    original_mapping_fn = local_worker.policy_mapping_fn
    policy_map = local_worker.policy_map

    # so that we dont compute async advantages during eval
    for policy in policy_map.values():
        #print("setting eval mode to true")
        policy._eval_mode = True
        
    metrics = {}
    eval_metrics = {}

    def eval_mapping_fn(agent_id, episode, worker, **kwargs):
        env_id = episode.env_id
        # print(episode.episode_id, env_id, "episode and env id")

        if agent_id == 1:
            #print("agent 1 eval agent")
            return 'polBREval'
        elif agent_id == 2:
            #print("agent 2 br agent")
            return 'polBR1'
        else:
            # print("worker curr partner idx", worker.global_eval_vars["curr_partner_idx"])
            # print("partner", worker.global_eval_vars["eval_policies"][worker.global_eval_vars["curr_partner_idx"]])
            return worker.global_eval_vars["eval_policies"][worker.global_eval_vars["curr_partner_idx"]]

    eval_policies = [pid for pid in policy_map.keys()
                        if 'polEval' in pid]
    # TODO: make this a config option
    num_eval_eps = 1 #3
    episodes_by_policy = defaultdict(list)

    for sub_env in workers.local_worker().async_env.get_sub_environments():
        sub_env.reset()
    num_parallel = workers.local_worker().async_env.num_envs

    if debug:
        print("EVAL POLICIES:", eval_policies)

    local_worker.set_policy_mapping_fn(eval_mapping_fn)
    if not hasattr(local_worker, 'global_eval_vars'):
        local_worker.global_eval_vars = {}

    local_worker.global_eval_vars["curr_partner_idx"] = 0
    local_worker.global_eval_vars["env_inc"] = 0
    local_worker.global_eval_vars["eval_policies"] = eval_policies

    if "flush_all" in local_worker.global_eval_vars:
        for e in range(num_parallel):
            _ = local_worker.sample()
            #print("flushing")
    #print("finished flushing")

    for _ in range(len(eval_policies) * num_eval_eps):
        # this inner loop just gets all the eps in the parallel env
        for e in range(num_parallel):
            if debug:
                # Check the base environment wrapper
                print(f"Async env type: {type(local_worker.async_env)}")
                print(f"Async env num_envs: {local_worker.async_env.num_envs}")

                # Get actual sub-environments
                sub_envs = local_worker.async_env.get_sub_environments()
                print(f"Number of actual sub-environments: {len(sub_envs)}")
                # 1. Raw environment level
                env = local_worker.async_env
                print(f"1. BaseEnv num_envs: {env.num_envs}")

                # 2. Check what poll returns directly
                poll_result = env.poll()
 
            batch = local_worker.sample()

            if debug:
                print(f"Total batch size: {batch.count}")
                print(f"Policies in batch: {batch.policy_batches.keys()}")
                for pid, sb in batch.policy_batches.items():
                    print(f"  {pid}: {sb.count} steps")
            for policy_id, sb in batch.policy_batches.items():
                if "polEval" in policy_id:
                    episodes_by_policy[policy_id].append(sb)
                    if debug:
                        print(f"Collected episode with eval policy {policy_id}")

    eval_metrics = {}
    total_all_pol_rew = 0
    total_eps = 0

    for pid, episodes in episodes_by_policy.items():
        total_reward = 0
        # total_sparse_rew0 = 0
        # total_sparse_rew1 = 0
        total_shaped_rew0 = 0
        total_shaped_rew1 = 0
        total_dishes_delivered = 0
        for i, ep in enumerate(episodes):
            total_reward += ep["rewards"].sum()
            #total_sparse_rew0 += sum(info["sparse_r_by_agent"][0] for info in ep["infos"])
            #total_sparse_rew1 += sum(info["sparse_r_by_agent"][1] for info in ep["infos"])
            total_shaped_rew0 += sum(info["shaped_r_by_agent"][0] for info in ep["infos"])
            total_shaped_rew1 += sum(info["shaped_r_by_agent"][1] for info in ep["infos"])
            total_dishes_delivered += len([(ep["t"][i],rew) for i,rew in enumerate(ep["rewards"]) if rew != 0.0])

            if debug:
                print("Episode for policy", pid)
                print(len(ep["rewards"]), "steps")
                print(len(ep["infos"]), "infos")
                print([(ep["t"][i],rew) for i,rew in enumerate(ep["rewards"]) if rew != 0.0])
                print([(ep["t"][i], info["sparse_r_by_agent"]) 
                    for i, info in enumerate(ep["infos"])
                    if info["sparse_r_by_agent"] != [0.0, 0.0]])
                print(f"Total reward against {pid}: {total_reward}")
                #print(f" (sparse_r: {total_sparse_rew0}, {total_sparse_rew1})")
                #print(f" (shaped_r: {total_shaped_rew0}, {total_shaped_rew1})")
        if debug:
            print(f"----------------------------")
        mean_reward = total_reward*2 / len(episodes) # x2 because game rw is 2x agent rew
        total_all_pol_rew += total_reward*2
        total_eps += len(episodes)
        #mean_sparse_rew0 = total_sparse_rew0 / len(episodes)
        #mean_sparse_rew1 = total_sparse_rew1 / len(episodes)
        mean_shaped_rew0 = total_shaped_rew0 / len(episodes)
        mean_shaped_rew1 = total_shaped_rew1 / len(episodes)
        mean_dishes_delivered = total_dishes_delivered / len(episodes)
        eval_metrics[f"with_{pid}_mean_dishes_delivered"] = int(mean_dishes_delivered)
        eval_metrics[f"with_{pid}_mean_reward"] = float(mean_reward)
        eval_metrics[f"with_{pid}_mean_shaped_rew0"] = float(mean_shaped_rew0)
        eval_metrics[f"with_{pid}_mean_shaped_rew1"] = float(mean_shaped_rew1)

    for policy in policy_map.values():
        if debug:
            print("setting eval mode to false")
        policy._eval_mode = False
    workers.local_worker().set_policy_mapping_fn(original_mapping_fn)
    print("resetting mapping fn")

    metrics.update(eval_metrics)
    metrics["num_eval_episodes"] = int(total_eps)

    avg_eval_reward = total_all_pol_rew / total_eps 
    metrics["avg_eval_reward"] = float(avg_eval_reward)
    print("Average eval reward across all eval policies: ", avg_eval_reward)
    metrics["num_eval_policies"] = len(eval_policies)
    
    return metrics