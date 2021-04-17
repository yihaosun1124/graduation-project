import numpy as np
import torch
import gym
import time
from utils.logx import EpochLogger
from utils.zfilter import ZFilter

from model import MLPActorCritic
from buffer import PPOBuffer
from ppo import PPO


def train(env_fn, seed=0, ppo_epoch=10, steps_per_epoch=2048, mini_batch_size=64,
          num_epoch=1500, gamma=0.99, clip_ratio=0.2, value_clip_ratio=10, value_loss_coef=0.5,
          entropy_loss_coef=0, use_value_clipped_loss=True, lr=3e-4, eps=1e-5, lam=0.95,
          max_grad_norm=0.5, max_ep_len=1000, save_freq=10, device=torch.device('cpu'),
          ac_kwargs=dict(), logger_kwargs=dict()):

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    actor_critic = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    ppo = PPO(actor_critic, clip_ratio, value_clip_ratio, ppo_epoch, mini_batch_size,
              value_loss_coef, entropy_loss_coef, lr, eps, max_grad_norm, use_value_clipped_loss)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device)

    # Set up model saving
    logger.setup_pytorch_saver(ppo.actor_critic)

    # Prepare for interaction with environment
    start_time = time.time()
    running_state = ZFilter((obs_dim[0],), clip=10)
    # running_reward = ZFilter((1,), demean=False, clip=10)
    obs, ep_ret, ep_len = env.reset(), 0, 0
    obs = running_state(obs)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(num_epoch):
        for t in range(steps_per_epoch):
            action, value, logp = ppo.actor_critic.step(torch.as_tensor(obs, dtype=torch.float32).to(device))

            next_obs, rew, done, _ = env.step(action)
            next_obs = running_state(next_obs)
            # rew = running_reward([rew])[0]
            ep_ret += rew
            ep_len += 1

            # save and log
            buf.store(obs, action, rew, value, logp)

            # Update obs (critical!)
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _ = ppo.actor_critic.step(torch.as_tensor(obs, dtype=torch.float32).to(device))
                else:
                    value = 0
                buf.finish_path(value)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs, ep_ret, ep_len = env.reset(), 0, 0
                    obs = running_state(obs)

        # Save model
        if save_freq != 0 and ((epoch % save_freq == 0) or (epoch == num_epoch - 1)):
            logger.save_state({'env': env}, None)

        # perform update
        data = buf.get()
        policy_loss, value_loss, entropy, kl = ppo.update(data)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', policy_loss)
        logger.log_tabular('LossV', value_loss)
        logger.log_tabular('Entropy', entropy)
        logger.log_tabular('KL', kl)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--steps_per_epoch', type=int, default=2048)
    parser.add_argument('--num_epoch', type=int, default=1500)
    parser.add_argument('--ppo_epoch', type=int, default=10)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--value_clip_ratio', type=float, default=10)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--entropy_loss_coef', type=float, default=0)
    parser.add_argument('--use_clipped_value_loss', type=bool, default=True)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.env, args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train(lambda : gym.make(args.env), seed=args.seed, ppo_epoch=args.ppo_epoch,
          steps_per_epoch=args.steps_per_epoch, mini_batch_size=args.mini_batch_size,
          num_epoch=args.num_epoch, gamma=args.gamma, clip_ratio=args.clip_ratio,
          value_clip_ratio=args.value_clip_ratio, value_loss_coef=args.value_loss_coef,
          entropy_loss_coef=args.entropy_loss_coef, use_value_clipped_loss=args.use_clipped_value_loss,
          lr=args.lr, eps=args.eps, lam=args.lam, max_grad_norm=args.max_grad_norm,
          max_ep_len=1000, save_freq=args.save_freq, device=device,
          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), logger_kwargs=logger_kwargs)


