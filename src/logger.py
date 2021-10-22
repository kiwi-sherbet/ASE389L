import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Callable, Optional
# from torch.utils.tensorboard import SummaryWriter
# from tensorboard.backend.event_processing import event_accumulator
import wandb

WRITE_TYPE = Union[int, Number, np.number, np.ndarray]


class WandbLogger():

    def __init__(
        self,
        project: str= "project",
        task: str= "task",
        path: str= "./log.dat",
        train_interval: int = 1,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
        actor: Any=None,
        critic: Any=None,
        reward_config = {},
        ppo_config = {},
        experiment_config = {},
    ) -> None:

        self.id = wandb.util.generate_id()
        self.writer = wandb.init(id=self.id, resume="allow",
                                project=project,
                                job_type=task,
                                config=ppo_config,
                                save_code=path,
                                dir=path,
                                sync_tensorboard=False
                                )
        self.writer.config.update(reward_config)
        self.writer.config.update(experiment_config)

        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1
        self.num_episode = 0

        self.writer .watch(actor, log_freq=100)
        self.writer .watch(critic, log_freq=100)

    def log_train_data(self, collect_result: dict, step: int) -> None:

        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                dicLogInfo = {}
                dicLogInfo["train/info/n_ep"] = collect_result["n/ep"]
                dicLogInfo["train/info/Reward"] = collect_result["rew"]
                dicLogInfo["train/info/Length"] = collect_result["len"]

                infos = collect_result["infos"]

                if "Reward" in infos.keys():
                    for key, value in infos["Reward"].items():
                        dicLogInfo["train/reward/{}".format(key)] = np.average(value)

                if "State" in infos.keys():
                    for key, value in infos["State"].items():
                        dicLogInfo["train/state/{}".format(key)] = np.average(value)

                self.writer.log(dicLogInfo, step=step, commit=False)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:

        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            dicLogInfo = {}
            dicLogInfo["test/info/Reward"] = rew
            dicLogInfo["test/info/Length"] = len_
            dicLogInfo["test/info/RewardStd"] = rew_std
            dicLogInfo["test/info/LengthStd"] = len_std


            infos = collect_result["infos"]

            if "Reward" in infos.keys():
                for key, value in infos["Reward"].items():
                    dicLogInfo["test/reward/{}".format(key)] = np.average(value)

            if "State" in infos.keys():
                for key, value in infos["State"].items():
                    dicLogInfo["test/state/{}".format(key)] = np.average(value)

            self.writer.log(dicLogInfo, step=step, commit=False)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            self.writer.log(update_result)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            dicLogInfo = {}
            dicLogInfo["save/Epoch"] = epoch
            dicLogInfo["save/EnvStep"] = env_step
            dicLogInfo["save/GradientStep"] = gradient_step
            self.writer.log(dicLogInfo)

    def restore_data(self) -> Tuple[int, int, int]:

        return None

