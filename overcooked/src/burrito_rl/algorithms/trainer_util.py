from burrito_rl.algorithms.agent_ppo_trainer import PPOPolicyTrainer
from sklearn import get_config

def create_ppo_trainer_class(get_config_fn=None, process_train_batch_fn=None, class_name="PPOPolicyTrainer"):

    class CustomTrainer(PPOPolicyTrainer):
        pass

    if get_config_fn is not None:
        CustomTrainer.get_config = get_config_fn
            
    # Override process_train_batch if provided
    if process_train_batch_fn is not None:
        CustomTrainer.process_train_batch = process_train_batch_fn
    

    CustomTrainer.__name__ = class_name

    return CustomTrainer



def get_trainer(alg='PPO'):
    if alg == 'PPO':
        return PPOPolicyTrainer
    
    elif alg == 'SP':
        from burrito_rl.algorithms.utils.sp_util import sp_rllib_config
        return create_ppo_trainer_class(
            get_config_fn = sp_rllib_config,
            class_name = "SelfPlayPPOTrainer"
        )
    
    elif alg == 'FCP' or alg == 'MEP':
        from burrito_rl.algorithms.utils.pp_util import population_rllib_config, process_MEP_train_batch
        return create_ppo_trainer_class(
            get_config_fn = population_rllib_config,
            process_train_batch_fn = process_MEP_train_batch, #if alg == 'MEP' else None,
            class_name = "PopulationPPOTrainer"
        )
    
    elif alg == 'BP':
        from burrito_rl.algorithms.utils.bp_util import behavior_preference_config
        return create_ppo_trainer_class(
            get_config_fn = behavior_preference_config,
            class_name = "BehaviorPreferenceTrainer"
        )
    
    elif alg == "BR":
        from burrito_rl.algorithms.utils.br_util import best_response_rllib_config
        return create_ppo_trainer_class(
            get_config_fn = best_response_rllib_config,
            class_name = "BestResponsePPOTrainer"
        )

    elif alg == "VAEBR":
        from burrito_rl.algorithms.utils.vaebr_util import vae_cluster_br_config
        return create_ppo_trainer_class(
            get_config_fn = vae_cluster_br_config,
            class_name = "VAEBestResponsePPOTrainer"
        )

    elif alg == "VAEIPPO_EVAL":
        from burrito_rl.algorithms.utils.vae_ippo_eval_util import vae_ippo_eval_config
        return create_ppo_trainer_class(
            get_config_fn = vae_ippo_eval_config,
            class_name = "VAEIPPOEvalTrainer"
        )

    elif alg == "CLUSTEREVAL":
        from burrito_rl.algorithms.utils.cluster_eval_util import cluster_eval_config
        return create_ppo_trainer_class(
            get_config_fn = cluster_eval_config,
            class_name = "ClusterEvalTrainer"
        )

    elif alg == "EVALBR":
        from burrito_rl.algorithms.utils.eval_br_util import eval_br_config
        return create_ppo_trainer_class(
            get_config_fn = eval_br_config,
            class_name = "EVALBRTrainer"
        )
    
    elif alg == "CLUSTERBR":
        from burrito_rl.algorithms.utils.cluster_br_eval_util import cluster_br_eval_config
        return create_ppo_trainer_class(
            get_config_fn = cluster_br_eval_config,
            class_name = "ClusterBREvalTrainer"
        )

    else:
        raise NotImplementedError
