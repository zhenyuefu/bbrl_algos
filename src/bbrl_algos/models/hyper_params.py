import sys
import os
import torch

import hydra
import optuna
import yaml

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl_algos.models.loggers import Logger

from bbrl_algos.models.algos.ddpg import run_ddpg


# %%
def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
    suggest_type = cfg["suggest_type"]
    args = cfg.keys() - ["suggest_type"]
    args_str = ", ".join([f"{arg}={cfg[arg]}" for arg in args])
    return eval(f'trial.suggest_{suggest_type}("{variable_name}", {args_str})')


def get_trial_config(trial: optuna.Trial, cfg: DictConfig):
    for variable_name in cfg.keys():
        if type(cfg[variable_name]) != DictConfig:
            continue
        else:
            if "suggest_type" in cfg[variable_name].keys():
                cfg[variable_name] = get_trial_value(
                    trial, cfg[variable_name], variable_name
                )
            else:
                cfg[variable_name] = get_trial_config(trial, cfg[variable_name])
    return cfg


def objective(trial, run_func, cfg_raw):
    cfg_sampled = get_trial_config(trial, cfg_raw.copy())

    logger = Logger(cfg_sampled)
    try:
        trial_result: float = run_func(cfg_sampled, logger, trial)
        logger.close()
        return trial_result
    except optuna.exceptions.TrialPruned:
        logger.close(exit_code=1)
        return float("-inf")


def launch_optuna(cfg_raw):
    cfg_optuna = cfg_raw.optuna

    # study = optuna.create_study(**cfg_optuna.study)
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(), direction="maximize"
    )
    study.optimize(func=objective, **cfg_optuna.optimize)

    file = open("best_params.yaml", "w")
    yaml.dump(study.best_params, file)
    file.close()


# la partie ci-dessous sera dans le fichier de chaque algo ------------

# %%
@hydra.main(
    config_path="configs/",
    # config_name="ddpg_cartpole.yaml"
    # config_name="ddpg_pendulum.yaml",
    config_name="ddpg_pendulum_optuna.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)
    launch_optuna(cfg_raw)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw)
    else:
        logger = Logger(cfg_raw)
        run_ddpg(cfg_raw, logger)


if __name__ == "__main__":
    main()
