import torch
import numpy as np
from bbrl import instantiate_class, get_arguments, get_class

def log_reward_losses(logger, rewards, nb_steps):
        logger.add_log("reward/mean", rewards.mean(), nb_steps)
        logger.add_log("reward/max", rewards.max(), nb_steps)
        logger.add_log("reward/min", rewards.min(), nb_steps)
        logger.add_log("reward/median", rewards.median(), nb_steps)

def log_losses(logger, critic_loss, entropy_loss, actor_loss, steps):
        logger.add_log("critic_loss", critic_loss, steps)
        logger.add_log("entropy_loss", entropy_loss, steps)
        logger.add_log("actor_loss", actor_loss, steps)


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, steps):
        self.logger.add_scalar(log_string, loss.item(), steps)

    # A specific function for RL algorithms having a critic, an actor and an entropy losses
    def log_losses(self, critic_loss, entropy_loss, actor_loss, steps):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)


class RewardLogger:
    def __init__(self, steps_filename, rewards_filename):
        self.steps_filename = steps_filename
        self.rewards_filename = rewards_filename
        self.episode = 0
        self.all_rewards = []
        self.all_rewards.append([])
        self.all_steps = []

    def add(self, nb_steps, reward):
        if self.episode == 0:
            self.all_steps.append(nb_steps)
        self.all_rewards[self.episode].append(reward.item())

    def new_episode(self):
        self.episode += 1
        self.all_rewards.append([])

    def save(self):
        # print("reward loader save:", self.all_steps,  self.all_rewards)
        with open(self.steps_filename, "ab") as f:
            np.save(f, self.all_steps)
        with open(self.rewards_filename, "ab") as f:
            np.save(f, self.all_rewards)


class RewardLoader:
    def __init__(self, steps_filename, rewards_filename):
        self.steps_filename = steps_filename
        self.rewards_filename = rewards_filename

    def load(self):
        with open(self.steps_filename, "rb") as f:
            steps = np.load(f, allow_pickle=True)
        with open(self.rewards_filename, "rb") as f:
            rewards = np.load(f, allow_pickle=True)
        return steps, rewards
    

# %%
class MyLogger:
    def __init__(self, cfg):
        logger_cfg = cfg.logger
        logger_args = get_arguments(logger_cfg)
        self.logger = get_class(logger_cfg)(**logger_args)
        self.logger.save_hps(cfg)

    def add_log(self, log_string, log_item, epoch) -> None:
        if isinstance(log_item, torch.Tensor) and log_item.dim() == 0:
            log_item = log_item.item()
        self.logger.add_scalar(log_string, log_item, epoch)

    def close(self, exit_code=0) -> None:
        self.logger.close(exit_code)
