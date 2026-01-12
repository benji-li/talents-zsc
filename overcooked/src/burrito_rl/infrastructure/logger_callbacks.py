import os
from typing import List
from ray.air._internal.checkpoint_manager import _TrackedCheckpoint
from ray.tune.callback import Callback
import shutil

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log
from multiprocessing import Process, Queue
from datetime import datetime
import wandb

from ray.tune.experiment import Trial

class CustomCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


    def on_trial_save(
        self,
        iteration: int,
        trials: List[Trial],
        trial: Trial,
        **info
    ):
        # 1) let W&B do its normal artifact upload
        super().on_trial_save(iteration, trials, trial, **info)

        # 2) then copy locally
        src = trial.checkpoint.dir_or_data
        if src:
            dst = os.path.join(self.checkpoint_dir, trial.trial_id[-4:], os.path.basename(src))
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[CustomLogger] copied checkpoint to {dst}")



class CustomWandbLogger(WandbLoggerCallback):
    def __init__(self, checkpoint_dir:str, project: str, name_prefix:str=None, name_keys = [], **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.name_keys = name_keys
        self.name_prefix = name_prefix
        super().__init__(project=project, **kwargs)
        

    def on_trial_save(
        self,
        iteration: int,
        trials: List[Trial],
        trial: Trial,
        **info
    ):
        # 1) let W&B do its normal artifact upload
        super().on_trial_save(iteration, trials, trial, **info)

        # 2) then copy locally
        src = trial.checkpoint.dir_or_data
        if src:
            dst = os.path.join(self.checkpoint_dir, os.path.basename(src))
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[CustomWandbLogger] copied checkpoint to {dst}")


    def on_trial_complete(self, iteration, trials, trial, **info):
        wandb.finish()
        super().on_trial_complete(iteration, trials, trial, **info)
        # close the run so it leaves the "running" state

    def on_trial_error(self, iteration, trials, trial, **info):
        wandb.finish()
        super().on_trial_error(iteration, trials, trial, **info)
        # also finish on errors/interrupts


    def _get_trial_name(self, trial: "Trial"):
        try:
            name = self.name_prefix if self.name_prefix else ''
            for key in self.name_keys:
                if key in trial.config:
                    val = trial.config[key]
                    name += f'{key}{val}_'
            timestr = datetime.now().strftime("%m-%d_%H-%M-%S")
            name += f"{trial.trial_id[-4:]}_"
            name += timestr
            return name
        except Exception:
            # Fallback to a safe name
            print('-------------------------------------------')
            print("failed to create trial name")
            print('-------------------------------------------')
            return f"trial_{trial.trial_id}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"



    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None
        trial_name = self._get_trial_name(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config = _clean_log(config)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=trial_name,
            resume=False,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=wandb_project,
            config=config,
        )
        wandb_init_kwargs.update(self.kwargs)

        self._trial_queues[trial] = Queue()
        self._trial_processes[trial] = self._logger_process_cls(
            logdir=trial.logdir,
            queue=self._trial_queues[trial],
            exclude=exclude_results,
            to_config=self._config_results,
            **wandb_init_kwargs,
        )
        self._trial_processes[trial].start()