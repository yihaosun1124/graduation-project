import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from buffer import generate_batch_data


class PPO():
    def __init__(self, actor_critic, clip_ratio, value_clip_ratio, ppo_epoch, mini_batch_size, value_loss_coef,
                 entropy_coef, lr, eps, max_grad_norm, use_clipped_value_loss=True):
        self.actor_critic = actor_critic
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, data):
        policy_loss_epoch, value_loss_epoch, entropy_loss_epoch, kl_epoch = 0, 0, 0, 0
        batch_size = len(data['ret'])

        for i in range(self.ppo_epoch):
            batch_data = generate_batch_data(data, batch_size, self.mini_batch_size)
            for mini_batch in batch_data:
                obs, act, ret, value_pred, adv, logp_old = mini_batch

                # Policy loss
                pi, logp = self.actor_critic.pi(obs, act)
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()
                approx_kl = (logp_old - logp).mean()

                # Value loss
                value = self.actor_critic.v(obs)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_pred + (value - value_pred).clamp(-self.value_clip_ratio, self.value_clip_ratio)
                    value_loss = (value - ret) ** 2
                    value_loss_clipped = (value_pred_clipped - ret) ** 2
                    value_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((ret - value) ** 2).mean()

                # Entropy loss
                entropy_loss = pi.entropy().mean()

                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                kl_epoch += approx_kl.item()

        num_updates = self.ppo_epoch * (batch_size / self.mini_batch_size)

        policy_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        kl_epoch /= num_updates

        return policy_loss_epoch, value_loss_epoch, entropy_loss_epoch, kl_epoch
