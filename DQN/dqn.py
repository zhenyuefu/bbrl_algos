#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from typing import Callable, List

import hydra
import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper

# %%
from bbrl import TimeAgent, SerializableAgent
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents.seeding import SeedableAgent


# %%
def compute_critic_loss_transitional(
        discount_factor, reward, must_bootstrap, action, q_values, q_target=None
):
    """Compute critic loss
    Args:
        discount_factor (float): The discount factor
        reward (torch.Tensor): a (2 × T × B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (2 × T × B) tensor containing 0 if the episode is completed at time $t$
        action (torch.LongTensor): a (2 × T) long tensor containing the chosen action
        q_values (torch.Tensor): a (2 × T × B × A) tensor containing Q values
        q_target (torch.Tensor, optional): a (2 × T × B × A) tensor containing target Q values

    Returns:
        torch.Scalar: The loss
    """
    if q_target is None:
        q_target = q_values
    max_q = q_target[1].amax(dim=-1).detach()
    target = reward[1] + discount_factor * max_q * must_bootstrap[1]
    act = action[0].unsqueeze(dim=-1)
    qvals = q_values[0].gather(dim=1, index=act).squeeze(dim=1)
    return nn.MSELoss()(qvals, target)


# %%
def make_wrappers(
        autoreset: bool,
) -> List[Callable[[Env], Env]]:
    return [AutoResetWrapper] if autoreset else []


# %%
def make_env(
        identifier: str,
        autoreset: bool,
        **kwargs,
) -> Env:
    env: Env = gym.make(id=identifier, **kwargs)
    wrappers = make_wrappers(
        autoreset=autoreset,
    )
    for wrapper in wrappers:
        env = wrapper(env)
    return env


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


# %%
def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


# %%
class DiscreteQAgent(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(
            self,
            state_dim,
            hidden_layers,
            action_dim,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t)).float()
        q_values = self.model(obs)
        self.set(("q_values", t), q_values)
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)


# %%

ActionSelector = TimeAgent, SeedableAgent, SerializableAgent


class EGreedyActionSelector(*ActionSelector):
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        # TODO: make it deterministic if seeded
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)


# %%
def create_dqn_agent(cfg_algo, train_env_agent, eval_env_agent):
    obs_space = train_env_agent.get_observation_space()
    obs_shape = obs_space.shape if len(obs_space.shape) > 0 else obs_space.n

    act_space = train_env_agent.get_action_space()
    act_shape = act_space.shape if len(act_space.shape) > 0 else act_space.n

    critic = DiscreteQAgent(
        state_dim=obs_shape[0],
        hidden_layers=list(cfg_algo.architecture.hidden_sizes),
        action_dim=act_shape,
        seed=cfg_algo.seed.q,
    )
    critic_target = copy.deepcopy(critic)

    explorer = EGreedyActionSelector(
        name="action_selector",
        epsilon_start=cfg_algo.explorer.epsilon_start,
        epsilon_end=cfg_algo.explorer.epsilon_end,
        epsilon_decay=cfg_algo.explorer.decay,
        seed=cfg_algo.seed.explorer,
    )
    q_agent = TemporalAgent(critic)
    q_agent_target = TemporalAgent(critic_target)

    tr_agent = Agents(train_env_agent, critic, explorer)
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return train_agent, eval_agent, q_agent, q_agent_target


# %%
# Configure the optimizer over the q agent
def setup_optimizer(optimizer_cfg, q_agent):
    optimizer_args = get_arguments(optimizer_cfg)
    parameters = q_agent.parameters()
    optimizer = get_class(optimizer_cfg)(parameters, **optimizer_args)
    return optimizer


# %%
def run_dqn(trial, cfg, logger):
    # 1) Create the environment agent
    train_env_agent = ParallelGymAgent(
        make_env_fn=get_class(cfg.gym_env_train),
        num_envs=cfg.algorithm.n_envs_train,
        make_env_args=get_arguments(cfg.gym_env_train),
        seed=cfg.algorithm.seed.train,
    )
    eval_env_agent = ParallelGymAgent(
        make_env_fn=get_class(cfg.gym_env_eval),
        num_envs=cfg.algorithm.n_envs_eval,
        make_env_args=get_arguments(cfg.gym_env_eval),
        seed=cfg.algorithm.seed.eval,
    )

    # 2) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, q_agent_target = create_dqn_agent(
        cfg.algorithm, train_env_agent, eval_env_agent
    )

    # 3) Create the training workspace
    train_workspace = Workspace()  # Used for training

    # 4) Create the replay buffer
    replay_buffer = ReplayBuffer(max_size=cfg.algorithm.buffer.max_size)

    # 5) Configure the optimizer
    optimizer = setup_optimizer(cfg.optimizer, q_agent)

    # 6) Define the steps counters
    best_reward = -1e19
    nb_steps = 0
    tmp_steps_target_update = 0
    tmp_steps_eval = 0

    while nb_steps < cfg.algorithm.n_steps_mini:
        # Decay the explorer epsilon
        explorer = train_agent.agent.get_by_name("action_selector")
        assert len(explorer) == 1, "There should be only one explorer"
        explorer[0].decay()

        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train - 1,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
            )

        transition_workspace = train_workspace.get_transitions(
            filter_key="env/done"
        )

        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        # Store the transition in the replay buffer
        replay_buffer.put(transition_workspace)

        for _ in range(cfg.algorithm.optim_n_updates):
            # Sample a batch from the replay buffer
            sampled_trans_ws = replay_buffer.get_shuffled(
                cfg.algorithm.buffer.batch_size
            )

            # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
            q_agent(sampled_trans_ws, t=0, n_steps=2, choose_action=False)

            q_values, terminated, truncated, reward, action = sampled_trans_ws[
                "q_values",
                'env/terminated',
                'env/truncated',
                'env/reward',
                "action",
            ]

            with torch.no_grad():
                q_agent_target(sampled_trans_ws, t=0, n_steps=2)

            # Compute the target q values
            q_target = sampled_trans_ws["q_values"]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not terminated.
            must_bootstrap = torch.logical_or(~terminated, truncated)

            # Store the replay buffer size
            logger.add_log("replay_buffer_size", replay_buffer.size(), nb_steps)

            if replay_buffer.size() > cfg.algorithm.buffer.learning_starts:
                # Compute critic loss
                critic_loss = compute_critic_loss_transitional(
                    cfg.algorithm.discount_factor,
                    reward,
                    must_bootstrap,
                    action,
                    q_values,
                    q_target,
                )

                # Store the loss for tensorboard display
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.max_grad_norm
                )
                optimizer.step()

        # Update the target network
        if (
                nb_steps - tmp_steps_target_update
                > cfg.algorithm.target_critic_update_interval
        ):
            tmp_steps_target_update = nb_steps
            q_agent_target.agent = copy.deepcopy(q_agent.agent)

        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                choose_action=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean, std = rewards.mean(), rewards.std()
            if mean > best_reward:
                best_reward = mean
            logger.add_log("reward", mean, nb_steps)
            stats_string = (
                "nb_steps: {}, mean reward: {:.2f}, std: {:.2f}".format(
                    nb_steps, mean, std
                )
            )
            print(stats_string)

    return best_reward


# %%
def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
    # code suivant assez moche, certes, piste d’amélioration possible
    suggest_type = cfg['suggest_type']
    args = cfg.keys() - ['suggest_type']
    args_str = ', '.join([f'{arg}={cfg[arg]}' for arg in args])
    return eval(f'trial.suggest_{suggest_type}("{variable_name}", {args_str})')


def get_trial_config(trial: optuna.Trial, cfg: DictConfig):
    for variable_name in cfg.keys():
        if type(cfg[variable_name]) != DictConfig:
            continue
        else:
            if 'suggest_type' in cfg[variable_name].keys():
                cfg[variable_name] = get_trial_value(trial, cfg[variable_name], variable_name)
            else:
                cfg[variable_name] = get_trial_config(trial, cfg[variable_name])
    return cfg


# %%
@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    torch.random.manual_seed(seed=cfg.algorithm.seed.torch)

    def objective(trial):

        cfg_sampled = cfg.copy()
        cfg_sampled = get_trial_config(trial, cfg_sampled)

        logger = MyLogger(cfg_sampled)
        try:
            trial_result: float = run_dqn(trial, cfg_sampled, logger)
            logger.close()
            return trial_result
        except optuna.exceptions.TrialPruned as e:
            logger.close(exit_code=1)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    main()
