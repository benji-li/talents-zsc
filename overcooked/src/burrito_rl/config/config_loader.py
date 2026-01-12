import collections.abc
import os
import pathlib
import yaml

import torch
import gym.spaces
import numpy as np


class ConfigLoader():

    @staticmethod
    def load_config(name, use_hpo = False):
        if "yaml" in name:
            path = name #os.path.join(name + ".yaml")
        else:
            names = name.split("/")

            path = os.path.join("config", names[0] + ".yaml")
        # path = os.path.join("src", "burrito_rl", "config", names[0] + ".yaml")
        # print(os.path.abspath(path)) 
        configs = yaml.safe_load(open(path))       
        configs = ConfigLoader._process_config(configs)
        configs = ConfigLoader._initialize_configs(configs)

        config = {"BASE_CONFIG":configs["BASE_CONFIG"]}

        #if len(names) > 1:
        #    config.update(configs[names[1]])

        if "HPO_CONFIG" in configs and use_hpo:
            # config.update(configs["HPO_CONFIG"])
            config = ConfigLoader._update_config(config, configs["HPO_CONFIG"])
        # print(config)
        return config

    @staticmethod
    def _process_config(config):
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = ConfigLoader._process_config(value)

            elif isinstance(value, str):
                if len(value) >= 5 and value[:5] == "tune.":
                    from ray import tune
                    config[key] = eval(value)

                elif len(value) >= 5 and value[:5] == "$SRC/":
                    # Note: could use string replace here instead of only checking prefix but am not sure it would ever be necessary
                    config[key] = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), value[5:])
            elif isinstance(value, list):
                for i, content in enumerate(value):
                    if isinstance(content, str):
                        if len(content) >= 5 and content[:5] == "$SRC/":
                            config[key][i] = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), content[5:])
        return config

    @staticmethod
    def _parse_gym_space(space_config: dict):
        space_type = space_config["type"]

        if space_type == "Box":
            return gym.spaces.Box(
                low = space_config["low"],
                high = space_config["high"],
                shape = tuple(space_config["shape"]),
                dtype = getattr(np, space_config["dtype"])
            )
        elif space_type == "Discrete": # action space
            return gym.spaces.Discrete(n=space_config["n"])
        elif space_type == "Dict": # recursively parse gym space
            return gym.spaces.Dict({
                key: ConfigLoader._parse_gym_space(subspace)
                for key, subspace in space_config["spaces"].items()
            })
        else:
            raise ValueError(f"Unsupported space type: {space_type}")

    @staticmethod
    def _initialize_configs(config):
        config["BASE_CONFIG"]["env_config"] = config["ENV_CONFIG"]
        config["BASE_CONFIG"]["model"] = config["MODEL_CONFIG"]
        config["BASE_CONFIG"]["num_gpus"] = config["BASE_CONFIG"]["num_gpus"] if torch.cuda.is_available() else 0
        env_config = config["ENV_CONFIG"]
        env_config["observation_space"] = ConfigLoader._parse_gym_space(env_config["observation_space"])
        env_config["action_space"] = ConfigLoader._parse_gym_space(env_config["action_space"])

        return config

    @staticmethod
    def _update_config(d, u):
        for k, v in u.items():
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = ConfigLoader._update_config(d.get(k, {}), v)
                    d[k] = r
                else:
                    d[k] = u[k]
            else:
                d = {k: u[k]}
        return d
