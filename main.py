import os
import numpy as np
import torch
import argparse
import gym
from learner import Reinforce
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description=None)

parser.add_argument('--env_id', default='CartPole-v1', help='Select the environment to run.')
parser.add_argument('--max_episodes', default=5000, help='Maximum number of episodes')
parser.add_argument('--net_size', default=100, help='Size of the neural network')
parser.add_argument('--buffer_capacity', default=int(1e5), help='Capacity of the experience replay buffer')
parser.add_argument('--batch_size', default=32, help='Capacity of the experience replay buffer')
parser.add_argument('--gamma', default=0.99, help='Discount rate for primary network updates')
parser.add_argument('--tau', default=1e-2, help='For soft target network updates')
parser.add_argument('--learn_rate', default=5e-4, help='For target network updates')
parser.add_argument('--test_mode', default=False, help='Training or testing')
parser.add_argument('--device', default=device, help='Training or testing')
parser.add_argument('--print_rate', default=20, help='Info print rate')
parser.add_argument('--env_seed', default=np.random.randint(low=0, high=int(2e9)), help='Seed for environment')
parser.add_argument('--torch_seed', default=np.random.randint(low=0, high=int(2e9)), help='Seed for environment')

args = parser.parse_args()

torch.random.manual_seed(args.torch_seed)

env = gym.make(args.env_id)
args.env = env

outdir = '.\\tmp\\agent-results'
if not os.path.exists(outdir):
    os.makedirs(outdir)

agent = Reinforce(args, env.observation_space.shape[0], env.action_space.n)

do_render = True
print_rate = args.print_rate
render_interval = print_rate

total_rewards_list = []
total_reward = 0
reward = 0

tic = time.perf_counter()
running_reward = 10


def select_action(obs):
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    probs = agent.forward(obs)
    dist = torch.distributions.Categorical(probs)
    act = dist.sample()
    agent.log_probs.append(dist.log_prob(act))
    return act.item()


def finish():
    tot_rew = 0
    policy_loss = []
    returns = []
    for r in agent.rewards[::-1]:
        tot_rew = r + args.gamma * tot_rew
        returns.insert(0, tot_rew)
    returns = torch.tensor(returns)
    returns = (returns-returns.mean())/(returns.std())
    for log_prob, R in zip(agent.log_probs, returns):
        policy_loss.append(-log_prob * R)

    agent.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    agent.optimizer.step()
    del agent.log_probs[:]
    del agent.rewards[:]


for ep in range(args.max_episodes):
    obs = env.reset()
    total_reward = 0
    done = False

    if ep != 0 and (ep % print_rate == 0):
        print(f'idx: {ep} | avg_rew: {np.mean(total_rewards_list)}')

    while not done:
        action = select_action(obs)
        obs_next, reward, done, info = env.step(action)
        agent.rewards.append(reward)
        total_reward += reward

        running_reward = 0.05 * total_reward + .95 * running_reward

        finish()
        obs = obs_next
        if running_reward > 200:
            print("Solved! Running reward is now {}".format(running_reward))
            break
        if do_render and ep != 0 and (ep % render_interval == 0):
            env.render()
    total_rewards_list.append(total_reward)

    toc = time.perf_counter()
