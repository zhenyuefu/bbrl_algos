import sys
import os

import gym
import my_gym
import mujoco_py

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.utils.functionalb import gae

import hydra

import torch
import torch.nn as nn

from bbrl_algos.models.stochastic_actors import TunableVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import SquashedGaussianActor
from bbrl_algos.models.stochastic_actors import StateDependentVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import ConstantVarianceContinuousActor
from bbrl_algos.models.stochastic_actors import DiscreteActor, BernoulliActor
from bbrl_algos.models.critics import VAgent
from bbrl_algos.models.loggers import Logger
from bbrl.utils.chrono import Chrono

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic
from bbrl_algos.models.envs import get_env_agents

# HYDRA_FULL_ERROR = 1

import matplotlib

matplotlib.use("TkAgg")


# Create the A2C Agent
def create_a2c_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    action_agent = globals()[cfg.algorithm.actor_type](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    tr_agent = Agents(train_env_agent, action_agent)
    ev_agent = Agents(eval_env_agent, action_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return train_agent, eval_agent, critic_agent


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, action_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_advantages_loss(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference
    # target = reward[:-1] + cfg.algorithm.discount_factor * v_value[1:].detach() * must_bootstrap.int()
    # td = target - v_value[:-1]
    advantages = gae(
        v_value,
        reward,
        must_bootstrap,
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    # Compute critic loss
    td_error = advantages**2
    critic_loss = td_error.mean()
    return critic_loss, advantages


def compute_actor_loss(action_logp, td):
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def run_a2c(cfg):
    # 1)  Build the  logger
    chrono = Chrono()
    logger = Logger(cfg)
    best_reward = float("-inf")

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the A2C Agent
    a2c_agent, eval_agent, critic_agent = create_a2c_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, a2c_agent, critic_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            a2c_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps - 1,
                stochastic=True,
                predict_proba=False,
                compute_entropy=True,
            )
        else:
            a2c_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
                predict_proba=False,
                compute_entropy=True,
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()

        v_value, done, truncated, reward, action, action_logp = transition_workspace[
            "v_value",
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "action_logprobs",
        ]

        nb_steps += action[0].shape[0]
        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # Compute critic loss
        critic_loss, advantages = compute_advantages_loss(
            cfg, reward, must_bootstrap, v_value
        )
        a2c_loss = compute_actor_loss(action_logp, advantages)

        # Compute entropy loss
        entropy_loss = torch.mean(train_workspace["entropy"])

        # Store the losses for tensorboard display
        logger.log_losses(nb_steps, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.entropy_coef * entropy_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            a2c_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
                predict_proba=False,
                render=False,  # render=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./a2c_policies/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#a2c#A1_22#"
                    + str(mean.item())
                    + ".agt"
                )
                policy = eval_agent.agent.agents[1]
                policy.save_model(filename)
                critic = critic_agent.agent
                if cfg.plot_agents:
                    plot_policy(
                        policy,
                        eval_env_agent,
                        "./a2c_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        critic,
                        eval_env_agent,
                        "./a2c_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )
    chrono.stop()


@hydra.main(
    config_path="./configs/",
    config_name="a2c_swimmer.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    run_a2c(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
