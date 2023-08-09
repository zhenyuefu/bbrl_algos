# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]

# # Outlook
#
# In this notebook, using BBRL, we first apply the TD3 algorithm to the MountainCarContinuous environment.
# We show that it does not work well due to a deceptive gradient effect.
# Then we compare three approaches to mitigate the issue: reward shaping, giving instructions, and behavioral cloning.

# We assume you know enough about BBRL, if this is not the case, please start with
# [the notebook about DDPG and TD3](http://master-dac.isir.upmc.fr/rld/rl/04-ddpg.student.ipynb)


# %% tags=["teacher"]
import sys
import os
from pathlib import Path
import math
from moviepy.editor import ipython_display as video_display

print("Not displaying video (hidden since not in a notebook)", file=sys.stderr)
def video_display(*args, **kwargs):
    pass
def display(*args, **kwargs):
    print(*args, **kwargs) 
    
    
testing_mode = os.environ.get("TESTING_MODE", None) == "ON"

# %%

import time
from typing import Tuple
from functools import partial

from omegaconf import OmegaConf
import torch
import bbrl_gymnasium

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
# [[remove]]
testing_mode = os.environ.get("TESTING_MODE", None) == "ON"
# [[/remove]]

# %% [markdown]
### BBRL imports

# %%

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace, 
# or until a given condition is reached
from bbrl.agents import Agents, TemporalAgent

# ParallelGymAgent is an agent able to execute a batch of gymnasium environments
# with auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/terminated’,
# 'env/truncated', 'env/done', ’env/cumulated_reward’, ... 
# 
# When called at timestep t=0, the environments are automatically reset. At
# timestep t>0, these agents will read the ’action’ variable in the workspace at
# time t − 1
from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, make_env

# Replay buffers are useful to store past transitions when training
from bbrl.utils.replay_buffer import ReplayBuffer

# %%
from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

import matplotlib

matplotlib.use("TkAgg")

# %% [markdown]

# ## Definition of agents

# %% [markdown]

# The function below builds a multi-layer perceptron where the size of each layer is given in the `size` list.
# We also specify the activation function of neurons at each layer and optionally a different activation function for the final layer.

# %% 
def build_mlp(sizes, activation, output_activation=nn.Identity()):
    """Helper function to build a multi-layer perceptron (function from $\mathbb R^n$ to $\mathbb R^p$)
    
    Args:
        sizes (List[int]): the number of neurons at each layer
        activation (nn.Module): a PyTorch activation function (after each layer but the last)
        output_activation (nn.Module): a PyTorch activation function (last layer)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


# %%
class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, name="critic"):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )
        self.with_name(name)

    def with_name(self, name):
        self.name = f"{name}/q_value"
        return self

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set((self.name, t), q_value)

    def predict_value(self, obs, action):
        osb_act = torch.cat((obs, action), dim=0)
        q_value = self.model(osb_act)
        return q_value


# %% [markdown]

# The actor is also a neural network, it takes a state $s$ as input and outputs
# an action $a$.

# %%
class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        print("obs", obs)
        action = self.model(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.model(obs)


# %% [markdown]

# ### Creating an Exploration method

# In the continuous action domain, basic exploration differs from the methods
# used in the discrete action domain. Here we generally add some Gaussian noise
# to the output of the actor.

# %%
from torch.distributions import Normal

# %%
class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)


# %% [markdown]

# In [the original DDPG paper](https://arxiv.org/pdf/1509.02971.pdf), the
# authors rather used the more sophisticated Ornstein-Uhlenbeck noise where
# noise is correlated between one step and the next.

# %%
class AddOUNoise(Agent):
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)

# %% 
from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial

def get_env_agents(cfg) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env,  cfg.gym_env.env_name, autoreset=True),
        cfg.algorithm.n_envs
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent


from bbrl import instantiate_class

class Logger():

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


# %%

def create_td3_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    print(obs_size, act_size)
    actor = ContinuousDeterministicActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
    tr_agent = Agents(train_env_agent, actor, noise_agent)
    ev_agent = Agents(eval_env_agent, actor)

    critic_1 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size,
        name="critic-1"
    )
    target_critic_1 = copy.deepcopy(critic_1).with_name("target-critic-1")
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size,
        name="critic-2"
    )
    target_critic_2 = copy.deepcopy(critic_2).with_name("target-critic-2")

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )

# %% [markdown]

# ### Setup the optimizers

# We use two separate optimizers to tune the parameters of the actor and the critic separately. That makes it possible to use a different learning rate for the actor and the critic.

# %%
# Configure the optimizer
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer

# %% [markdown]

# To update the target critic, one uses the following equation:
# $\theta' \leftarrow \tau \theta + (1- \tau) \theta'$
# where $\theta$ is the vector of parameters of the critic, and $\theta'$ is the vector of parameters of the target critic.
# The `soft_update_params(...)` function is in charge of performing this soft update.


# %%
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# %% [markdown]
# ### Compute actor loss
# The actor loss is straightforward. We want the actor to maximize Q-values, thus we minimize the mean of negated Q-values.

# %%
def compute_actor_loss(q_values):
    return -q_values.mean()

# %% [markdown]
# ### Compute critic loss

# %%

def compute_critic_loss(cfg, reward, must_bootstrap, q_values_1, q_values_2, q_next):
    # Compute temporal difference
    target = (
        reward[1:][0] + cfg.algorithm.discount_factor * q_next * must_bootstrap.int()
    )
    td_1 = target - q_values_1.squeeze(-1)
    td_2 = target - q_values_2.squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    return critic_loss_1, critic_loss_2

# %% [markdown]
# ### Main loop

# %%

def run_td3(cfg, env_creation_function):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agents
    train_env_agent, eval_env_agent = env_creation_function(cfg)

    # 3) Create the TD3 Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent_1 = TemporalAgent(critic_1)
    q_agents = TemporalAgent(Agents(critic_1, critic_2))
    target_q_agents = TemporalAgent(Agents(target_critic_1, target_critic_2))
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        done, truncated, reward = rb_workspace[
            "env/done", "env/truncated", "env/reward"
        ]
        # print(f"done {done}, reward {reward}, action {action}")
        if nb_steps > cfg.algorithm.learning_starts:
            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])

            # Critic update
            # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
            q_agents(rb_workspace, t=0, n_steps=1)

            q_values_rb_1, q_values_rb_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]

            with torch.no_grad():
                # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                ag_actor(rb_workspace, t=1, n_steps=1)
                # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                target_q_agents(rb_workspace, t=1, n_steps=1)

                post_q_values_1, post_q_values_2 = rb_workspace["target-critic-1/q_value", "target-critic-2/q_value"]

            post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)
            # Compute critic loss
            critic_loss_1, critic_loss_2 = compute_critic_loss(
                cfg,
                reward,
                must_bootstrap,
                q_values_rb_1[0],
                q_values_rb_2[0],
                post_q_values[1],
            )
            logger.add_log("critic_loss/1", critic_loss_1, nb_steps)
            logger.add_log("critic_loss/2", critic_loss_2, nb_steps)
            critic_loss = critic_loss_1 + critic_loss_2

            # Actor update
            # Now we determine the actions the current policy would take in the states from the RB
            ag_actor(rb_workspace, t=0, n_steps=1)

            # We determine the Q values resulting from actions of the current policy
            # We arbitrarily chose to update the actor with respect to critic_1
            # and we back-propagate the corresponding loss to maximize the Q values
            q_agents(rb_workspace, t=0, n_steps=1)

            q_values_1, q_values_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]
            current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)
            actor_loss = compute_actor_loss(current_q_values)
            logger.add_log("actor_loss", actor_loss, nb_steps)

            # Actor update part
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()
            # Critic update part 
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic_1.parameters(), cfg.algorithm.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                critic_2.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()

            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            soft_update_params(critic_1, target_critic_1, tau)
            soft_update_params(critic_2, target_critic_2, tau)
            # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if mean > best_reward:
                best_reward = mean
                if cfg.save_best:
                    directory = "./td3_agent/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    filename = directory + "td3_" + str(mean.item()) + ".agt"
                    eval_agent.save_model(filename)
                if cfg.plot_agents:
                    plot_policy(
                        actor,
                        eval_env_agent,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent_1.agent,  # TODO: do we want to plot both critics?
                        eval_env_agent,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )

# %% [markdown]
# ### Parameters : playing with MountainCarContinuous-v0

# In the gym MountainCar classic control environment, the goal is to push a car on top of a hill with three actions: pushing left, right, or no push.
# A large positive reward (+100) is obtained by reaching a flag on the top of the hill, but the car is underactuated: pushing right at all times is not enough to reach the top of the hill, pushing left to get some momentum is necessary.
# MountainCarContinuous-v0 is a variant where the action is in [-1, 1] and represents an oriented pushing force.
# But the applied forces are substracted to the reward, so the agent should push as few as possible.
# This gym classic control environment is depicted here: https://www.gymlibrary.ml/environments/classic_control/mountain_car_continuous/

# State variables in MountainCarContinuous-v0 are the position in $x$ and the velocity $v$ of the cart.
# The position $x$ is in $[-1.2, 0.6]$, the bottom of the valley between both hills is at $x = -0.5$ and the flag on top of the right hand side hill is at $x = 0.45$.
# More details are given in the comments of [the github code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py)

# %%
params={
  "save_best": False,
  # Set to true to have an insight on the learned policy
  # (but slows down the evaluation a lot!)
  "plot_agents": True,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tblogs/td3/" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,    
    },

  "algorithm":{
    "seed": 1,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 1,
    "n_steps": 100,
    "eval_interval": 50 if testing_mode else 2000,
    "nb_evals": 10,
    "gae": 0.8,
    "max_epochs": 100 if testing_mode else 21000,
    "discount_factor": 0.98,
    "buffer_size": 2e5,
    "batch_size": 64,
    "tau_target": 0.05,
    "learning_starts": 10000,
    "action_noise": 0.1,
    "architecture":{
        "actor_hidden_size": [400, 300],
        "critic_hidden_size": [400, 300],
        },
# [[remove]]
    # Reduces learning when testing
    "eval_interval": 3 if testing_mode else 2000,
    "max_epochs": 5 if testing_mode else 21000,
    "learning_starts": 10,
# [[/remove]]
  },
  "gym_env":{
    "env_name": "MountainCarContinuous-v0",
  },
  "actor_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
  },
  "critic_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
  }
}

import gymnasium as gym

class TutorInstructionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) The gym environment that will be wrapped
    # This is a custom wrapper for enforcing instructions in the MountainCarContinuous environment
    # We need to store the current observation to determine the instruction to give (as a function of the observation)
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(TutorInstructionWrapper, self).__init__(env)
        self.current_obs, _ = self.env.reset()
        self.nb_total_instructions = 0

    def reset(self, seed):
        """
        Reset the environment
        """
        obs, _ = self.env.reset()
        print(obs)
        self.current_obs = obs
        return obs

    def tutor_instruction_signal(self, action):
        # [[student]] # if False: # find an adequate condition
        if -1.0 < self.current_obs[0] < 0.4 and self.current_obs[1] > 0.007:
        # [[/student]]
            self.nb_total_instructions += 1
            # print("push right")
            return [1.0]
        # [[student]] # elif False: # find an adequate condition
        elif -1.0 < self.current_obs[0] < -0.5 and self.current_obs[1] < -0.01:
        # [[/student]]
            self.nb_total_instructions += 1
            # print("push left")
            return [-1.0]
        else:
            return action
        
    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        action = self.tutor_instruction_signal(action)
        print("act", action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(obs, action)
        self.current_obs = obs
        return obs, reward, terminated, truncated, info

# %% [markdown]

# We need to wrap the environment into the TutorInstructionWrapper.

# %%

def make_tutor_wrapped_env(env_name):
    env = TutorInstructionWrapper(gym.make(env_name))
    return env

# %% [markdown]

# ### Building the training and evaluation environments

# %%
def create_tutor_wrapped_env_agents(cfg) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_tutor_wrapped_env, cfg.gym_env.env_name), 
        cfg.algorithm.n_envs
    ).seed(cfg.algorithm.seed)
    
    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_tutor_wrapped_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent

# %% [markdown]

# ### Running the experiment

# %%

config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
run_td3(config, create_tutor_wrapped_env_agents)


# %%
# TODO: add a plot_trajectory function

# %% [markdown]

# ### Approach: Reward shaping

# In the second approach, the tutor provides additional feedback to the learner along the trajectory.
# In the code below, you have to fill code in the ```tutor_reward_signal(self, obs, action, reward)``` function so that the agent
# finally succeeds in solving the task. Once this is done and if you find time, try to minimize the number of reward signals.

# %%

import random as rd
import gymnasium as gym

class TutorFeedbackWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    # This is a custom wrapper for playing with reward shaping in the MountainCarContinuous environment
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(TutorFeedbackWrapper, self).__init__(env)
        self.nb_total_feedback = 0
        self.best_right = -0.9
        self.current_obs = None


    # This is the reward shaping function that you need to write
    def tutor_reward_signal(self, obs, action, reward):
        # [[student]]
        if reward > 0 or self.stop_helping == True: # if the agent found reward by itself, no need for shaping
            self.stop_helping = True
            return reward
        
        if obs[0] > self.best_right:
            self.best_right = obs[0]
            print("best right : ", obs[0])
            self.nb_total_feedback += 1
            return 10
        if obs[0] < -0.8: # if I'm a lot on the left, I should push right
            r = action[0]*5
            # print ("extreme left !!! : ", obs[0], ":", obs[1], ":", action[0], "reward:", r)
            self.nb_total_feedback += 1
            return r
        if obs[0] < -0.6 and obs[1] < -0.01 and action[0] < -0.2:
            r = abs(action[0])*3
            # print ("good left !!! : ", obs[0], ":", obs[1], ":", action[0], "reward:", r)
            self.nb_total_feedback += 1
            return r
        elif obs[0] < -0.5 and obs[1] > 0.008 and action[0] > 0.1:
            r = abs(action[0])*5
            # print("good momentum : ", obs[0], ":", obs[1], ":", action[0], "reward:", r)
            self.nb_total_feedback += 1
            return r
        elif obs[0] > -0.3 and obs[1] > 0.009 and action[0] > 0.2:
            r = abs(action[0])*5
            # print("good push right : ", obs[0], ":", obs[1], ":", action[0], "reward:", r)
            self.nb_total_feedback += 1
            return r
        else:
          if rd.random() < 1:
            return -0.9
          # print("no feedback")
          # [[/student]]
          return reward

    def reset(self, seed):
        """
        Reset the environment
        """
        obs, _ = self.env.reset()
        self.current_obs = obs
        # print("reset")
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.tutor_reward_signal(obs, action, reward)
        self.current_obs = obs
        if terminated:
          print("terminated : r", reward)
        return obs, reward, terminated, truncated, info

# %% [markdown]

# We need to wrap the environment into the TutorFeedbackWrapper.

# %%
def make_reward_shaped_env(env_name):
    env = TutorFeedbackWrapper(gym.make(env_name))
    return env

# %% [markdown]

# ### Building the training and evaluation environments

# %%
def create_reward_shaped_env_agents(cfg) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_reward_shaped_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)
    
    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_reward_shaped_env, cfg.gym_env.env_name), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent

# %%

config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
run_td3(config, create_reward_shaped_env_agents)

# TODO: add a plot_trajectory function

# %% [markdown]

# ### Approach: Behavioral cloning

# In a third approach, an expert policy is provided so as to bootstrap the learning process.
# The agent policy is regressed from data generated by the expert policy using the code below.
# This consists in collecting rollouts from the expert policy and then performing behavioral cloning on the resulting data.

# Here is the code of the function that generates expert data to solve the continuous mountain car problem:
# it consists in first going left to accumulate potential energy and then using the resulting momentum to rush
# towards the right hand side. Adding noise is useful to perform regression from more varied data.

# %%
def continuous_mountain_car_expert_policy(time_step: int, add_noise: bool) -> torch.Tensor:
    """
    This function is used to generate expert trajectories on Mountain Car environments
    so as to circumvent a deceptive gradient effect
    that makes standard policy gradient very inefficient on these environments.

    :param time_step: current episode step
    :param add_noise: whether to add some noise to the output
    ;return: an action depending on the current time step, eventually adding some noise
    """
    if add_noise:
        noise = random.random() / 20
    else:
        noise = 0.0
    if time_step < 50:
        return torch.tensor([[-1.0 + noise]])
    else:
        return torch.tensor([[1.0 + noise]])

# %% [markdown]

# And here is the code of the function performing behavioral cloning. 

# %%

from torch.nn import functional as func

def regress_policy(rollout_data, actor) -> None:
      """
      Perform behavioral cloning
      :param rollout_data: expert data from which to train your policy
      :param actor: the policy you are training
      ;return: nothing (side effect on the trained policy)
      """
      obs = rollout_data.observations
      actions = rollout_data.actions
      action_loss = 1e20
      while action_loss > 0.1:
          self_actions = actor.forward(obs)
          action_loss = func.mse_loss(actions, self_actions)
          actor.optimizer.zero_grad()
          action_loss.sum().backward()
          actor.optimizer.step()

# %%

# To do: write the equivalent of collect expert rollout

# TODO: add a plot_trajectory function

# %%

config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
run_td3(config, create_reward_shaped_env_agents)
