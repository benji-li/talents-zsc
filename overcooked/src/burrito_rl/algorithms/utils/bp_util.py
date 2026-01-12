from burrito_rl.algorithms.callbacks import LoggingCallbacks


def bp_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == len(list(worker.policy_map.keys()))-1:
        return 'polBR'
    else:
        return "pol" + str(agent_id + 1)


def behavior_preference_config(trainer, Config:dict) -> dict:
    Config["callbacks"] = LoggingCallbacks
    # behavior preference reward shaping config:
    # Here by default, we set pol1 (agent 0) corresponds to the bp agent
    print(Config['pref_to_tune'])
    for key in Config['env_config']['rew_shaping_params']:
        if key in Config["pref_to_tune"]:
            print(key, Config[key])
            print(Config['env_config']['rew_shaping_params'][key])
            Config['env_config']['rew_shaping_params'][key][0] = Config[key]

    policy_ids = [f'pol{i+1}' for i in range(Config["env_config"]["n_agents"]-1)]
    policy_ids += ['polBR']
    Config["multiagent"] = {
        "policies": {
            pid: (
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
            for i, pid in enumerate(policy_ids)
        },
        # Use the top-level function instead of a lambda
        "policy_mapping_fn": bp_policy_mapping_fn,
        "policies_to_train": policy_ids,
        # "policies_to_train": ['pol1']
    }
    return Config
