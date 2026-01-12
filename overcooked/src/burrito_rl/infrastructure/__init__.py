# register model
from ray.rllib.models import ModelCatalog
from burrito_rl.model.actor_critic import ActorCritic
from burrito_rl.model.centralized_critic import CentralizedActorCritic

ModelCatalog.register_custom_model(
    "IPPO_model",
    ActorCritic
)
ModelCatalog.register_custom_model(
    "MAPPO_model",
    CentralizedActorCritic
)

# regieter environment
from burrito_rl.env_wrapper.burrito_env import BurritoRLLibWrapper
from ray.tune import register_env

register_env("burrito", lambda env_config: BurritoRLLibWrapper(env_config))
