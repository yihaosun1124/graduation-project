from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def orthogonal_initializer(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)


def normc_initializer(layer, std=1.0, bias_const=0.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))
    initializer(layer.weight)
    nn.init.constant_(layer.bias, bias_const)


def mlp(sizes, initializer, activation, output_activation=nn.Identity, is_actor=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        linear = nn.Linear(sizes[j], sizes[j+1])
        initializer(linear, std=1.0)
        if j == len(sizes)-2 and is_actor:
            initializer(linear, std=0.01)
        layers += [linear, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, initializer, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], initializer, activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, initializer, activation):
        super().__init__()
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))
        # self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * act_dim))
        self.mean_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], initializer, activation, is_actor=True)

    def _distribution(self, obs):
        mean = self.mean_net(obs)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, initializer, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], initializer, activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), initializer=orthogonal_initializer, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, initializer, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, initializer, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, initializer, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def compute_v(self, obs):
        v = self.v(obs)
        return v

    def compute_q(self, obs, act, entropy_coef):
        value = self.v(obs)
        pi = self.pi._distribution(obs)
        logp_a = pi.log_prob(act).sum(axis=-1)
        return value + entropy_coef * logp_a


if __name__ == '__main__':
    import gym
    env = gym.make('HalfCheetah-v2')
    ac_kwargs = dict(hidden_sizes=[64]*2)
    ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)