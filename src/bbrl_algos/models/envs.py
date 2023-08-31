import os
import gym
# necessary to see the bbrl_gymnasium environments
import bbrl_gymnasium

from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial


assets_path = os.getcwd() + '/../../assets/'


def get_env_agents(
    cfg, *, autoreset=True, include_last_state=True
) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`

    """
    if cfg.gym_env.xml_file is None:
        xml_file = None
    else:
        xml_file = assets_path + cfg.gym_env.xml_file
        # print ("loading:", xml_file)

    if cfg.gym_env.wrappers is not None:
        print ("using wrappers:", cfg.gym_env.wrappers)
    """

    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=autoreset), # , wrappers=cfg.gym_env.wrappers), # , xml_file=xml_file),
        cfg.algorithm.n_envs,
        include_last_state=include_last_state,
    ).seed(seed=cfg.algorithm.seed.train)

    # Test environment (implictly, autoreset=False, which is always the case for evaluation environments)
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name), # , wrappers=cfg.gym_env.wrappers), # , xml_file=xml_file),
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state,
    ).seed(cfg.algorithm.seed.eval)

    return train_env_agent, eval_env_agent
