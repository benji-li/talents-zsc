from burrito_rl.algorithms.callbacks import LoggingCallbacks

def sp_policy_mapping_fn(aid, **kwargs):
    return "default_policy"

def sp_rllib_config(trainer, Config:dict) -> dict:
    Config["callbacks"] = LoggingCallbacks
    Config["multiagent"] = {
        "policies": {
            "default_policy": (
                None,  # policy spec
                Config["env_config"]["observation_space"],  # observation spec
                Config["env_config"]["action_space"],       # action spec
                {
                    "name": "pol0",
                    "framework": "torch",
                    "model": Config["model"],
                    "pretrained_model_path": Config["pretrained_model_path"][0],
                },
            )
        },
        # Use the top-level function instead of a lambda
        "policy_mapping_fn": sp_policy_mapping_fn,

    }
    return Config
