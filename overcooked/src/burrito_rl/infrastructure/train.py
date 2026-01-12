import ray
from ray import tune
from ray.tune.logger import UnifiedLogger
from burrito_rl.infrastructure.logger_callbacks import CustomCheckpointCallback, CustomWandbLogger
# from ray.air.integrations.wandb import WandbLoggerCallback
from burrito_rl.algorithms.trainer_util import get_trainer
from burrito_rl.config.config_loader import ConfigLoader
from burrito_rl.rllib_utils import evaluate
from datetime import datetime
import tempfile
import os


def logger_creater(args):
    "create logger to a customized path"
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(args.name, timestr)
    dir = './exp_data'
    if args.mode == "train":
        dir = dir + "/train"
    if args.mode == "eval":
        dir = dir + "/eval"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=dir)
    def default_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)
    return default_logger_creator


def train(args):
    
    
    Config = ConfigLoader.load_config(args.config)

    if args.mode == "tune":
        assert args.model_path is not None # need to specify a path to store trained models
        restore = args.restore_path if args.load_model else None
        assert os.path.exists(args.restore_path) if restore else True
        
        ray.init(object_store_memory=50_000_000_000,_memory=100_000_000_000)

        callbacks = [CustomCheckpointCallback(args.model_path)]
        if args.logger == 'wandb':
            wandb_callback = CustomWandbLogger(
                checkpoint_dir = args.model_path,
                project=args.wandb_project,
                name_prefix = args.name,
                name_keys=['alg'], # 'mep_coeff'],
                api_key_file=None,
                log_config=True
            )
            callbacks = [wandb_callback]

        tune.run(
            get_trainer(alg=Config['BASE_CONFIG']['alg']),
            name = args.name,
            stop = {"timesteps_total":args.timesteps_total},#"training_iteration": args.stop_iters, },
            config = Config['BASE_CONFIG'],
            local_dir=f"./exp_data/{args.name}/",
            #storage_path = "file://" + os.path.abspath("./exp_data/br-2/"),
            verbose=3,                    # set to enable different extent of logging.
            checkpoint_freq=args.ckpt_freq,
            keep_checkpoints_num=5,
            checkpoint_at_end=True,
            restore = restore,
            callbacks = callbacks,
        )
        ray.shutdown()

    if args.mode == "eval":
        # Your ray.init() should look like this:
#        ray.init(
#            num_cpus=32,  # Much less than 255
#            num_gpus=1,   # Use GPU if available
#            object_store_memory=2000000000  # 2GB
#        )
        alg = Config['BASE_CONFIG']['alg']
        config = Config['BASE_CONFIG']
        trainer = get_trainer(alg=alg)(config=config, logger_creator=logger_creater(args))
        #trainer.restore(args.model_path)
        models = []
        policies = []
        for i,policy_id in enumerate(config["multiagent"]["policies"].keys()):
            print("policy",i)
            models.append(trainer.get_policy(policy_id).model)
            policies.append(trainer.get_policy(policy_id))
            models[i].eval()
        #if "burrito" in args.config:
        print("env config",config["env_config"])
        results = evaluate(config["env_config"], policies, num_episodes= args.eval_episodes, display=args.render, ifsave=args.save_render, save='./eval_data', save_vids=args.save_vid)
        return results
        return
        for i, (policy_id, pol) in enumerate(trainer.workers.local_worker().policy_map):
            models.append(trainer.get_policy(policy_id).model)
            policies.append(trainer.get_policy(policy_id))
            models[i].eval()
        if "burrito" in args.config:
            results = evaluate(config["env_config"], policies, num_episodes= args.eval_episodes, display=args.render, ifsave=args.save_render, save='./exp_data/', save_vids=args.save_vid)
            return results
        return
