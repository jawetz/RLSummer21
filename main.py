import os
import numpy as np
import torch
import argparse
import gym
import time
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional



class Reinforce(torch.nn.Module):
    def __init__(self):
        self.args=args
        self.device=args.device
        super(Reinforce, self).__init__()
        self.fc1=torch.nn.Linear(env.observation_space.shape[0], args.net_size)
        self.fc2=torch.nn.Linear(args.net_size, args.net_size)
        self.fcout=torch.nn.Linear(args.net_size, env.action_space.n)
        self.drop=torch.nn.Dropout(p=0.5)
        self.act=torch.relu
        self.gamma=args.gamma

        self.action_space=args.env.action_space
        self.log_probs=[]
        self.rewards=[]

    def forward(self, obser):
        out=self.act(self.fc1(obser))
        out=self.drop(out)
        out=self.act(self.fc2(out))
        out=self.drop(out)
        out=self.fcout(out)
        softmax=torch.nn.functional.softmax(out, dim=1)
        return softmax

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

agent = Reinforce()
optimizer=optim.Adam(agent.parameters(), lr=1e-2)


do_render = True
print_rate = args.print_rate
render_interval = print_rate

total_rewards_list = []
total_reward = 0
reward = 0
epso = np.finfo(np.float32).eps.item()

tic = time.perf_counter()



def select_action(obs):
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    probs = agent.forward(obs)
    dist = Categorical(probs)
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
    returns = (returns - returns.mean()) / (returns.std() + epso)

    for log_prob, R in zip(agent.log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del agent.log_probs[:]
    del agent.rewards[:]

running_reward = 10
for ep in range(args.max_episodes):
   obs, total_reward = env.reset(), 0
   done= False
   if ep != 0 and (ep % print_rate == 0):
       print(f'idx: {ep} | avg_rew: {np.mean(total_rewards_list)}')

   while not done:
       action = select_action(obs)
       obs, reward, done, _ = env.step(action)
       if do_render:
           env.render()
       agent.rewards.append(reward)
       total_reward += reward
       if done:
           break

   running_reward = 0.05 * total_reward + (1 - 0.05) * running_reward
   finish()

   if running_reward > 500:
       print("Solved! Running reward is now {}".format(running_reward))
       break
   if do_render and ep != 0 and (ep % render_interval == 0):
       env.render()
   total_rewards_list.append(total_reward)

   toc = time.perf_counter()

