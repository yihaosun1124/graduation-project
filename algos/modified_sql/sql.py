import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy


class SQL():
    def __init__(self, actor_critic, lr, batch_size, update_every, gamma, polyak, value_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.actor_critic_targ = deepcopy(self.actor_critic)
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.polyak = polyak
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

    def update(self, data):
        obs, obs2, act, rew, done = data['obs'], data['obs2'], data['act'], data['rew'], data['done']
        q_pred = self.actor_critic.compute_q(obs, act, self.entropy_coef)
        with torch.no_grad():
            next_value = self.actor_critic_targ.compute_v(obs2)
            q_targ = rew + self.gamma * (1 - done) * next_value
        loss = 0.5 * ((q_pred - q_targ) ** 2).mean()

        self.optimizer.zero_grad()

        for p in self.actor_critic.v.parameters():
            p.requires_grad = False
        policy_stream_loss = loss * (1 / self.entropy_coef)
        policy_stream_loss.backward(retain_graph=True)

        for p in self.actor_critic.v.parameters():
            p.requires_grad = True
        for p in self.actor_critic.pi.parameters():
            p.requires_grad = False

        value_stream_loss = loss * self.value_coef
        value_stream_loss.backward()

        for p in self.actor_critic.v.parameters():
            p.requires_grad = True
        for p in self.actor_critic.pi.parameters():
            p.requires_grad = True

        self.optimizer.step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss
