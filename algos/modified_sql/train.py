import numpy as np
import torch
import gym
import time
from utils.logx import EpochLogger
from collections import deque

from buffer import ReplayBuffer
from sql import SQL
from model import MLPActorCritic


def train(env_fn, env_name, ac_kwargs=dict(), seed=0, steps_per_epoch=1000,
         epochs=3000, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=3e-4, batch_size=64, start_steps=10000,
         update_after=10000, update_every=1, num_test_episodes=10, value_coef=0.5, entropy_coef=0.02,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=10, device=torch.device('cpu')):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    env.seed(seed)
    test_env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    actor_critic = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    sql = SQL(actor_critic, lr, batch_size, update_every, gamma, polyak, value_coef, entropy_coef)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)

    rewards_log = []
    episode_rewards = deque(maxlen=10)

    # Set up model saving
    logger.setup_pytorch_saver(sql.actor_critic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                action = sql.actor_critic.act(torch.as_tensor(o, dtype=torch.float32).to(device))
                o, r, d, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            episode_rewards.append(ep_ret)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = sql.actor_critic.act(torch.as_tensor(o, dtype=torch.float32).to(device))
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss = sql.update(data=batch)
                logger.store(Loss=loss)
        else:
            logger.store(Loss=0.)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if save_freq != 0 and ((epoch % save_freq == 0) or (epoch == epochs)):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            rewards_log.append(np.mean(episode_rewards))

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('Loss', average_only=True)
            logger.dump_tabular()

    rewards_log = np.array(rewards_log)
    save_path = '../../log/modified_sql/' + env_name + '/' + str(seed) + '.npy'
    np.save(save_path, rewards_log)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--update_after', type=int, default=10000)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--exp_name', type=str, default='modified_sql')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.env, args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train(lambda: gym.make(args.env), args.env, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
          gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
          replay_size=int(args.replay_size), polyak=args.polyak, batch_size=args.batch_size,
          start_steps=args.start_steps, update_after=args.update_after, update_every=args.update_every,
          value_coef=args.value_coef, entropy_coef=args.entropy_coef, logger_kwargs=logger_kwargs, device=device)