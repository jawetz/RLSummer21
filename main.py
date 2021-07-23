import gym
import torch
import random
import torch
import numpy as np
from experiencereplay import ExperienceReplay
env=gym.make('LunarLander-v2')
env.reset()
class DQN():
    def __init__(self, n_state, n_action, net_size=100):
        self.lose=torch.nn.MSELoss()
        super(DQN, self).__init__()
        self.model=torch.nn.Sequential(
            torch.nn.Linear(n_state, net_size), 
            torch.nn.ReLU(),
            torch.nn.Linear(net_size, net_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(net_size, n_action)
        )
        self.target=torch.nn.Sequential(
            torch.nn.Linear(n_state, net_size), 
            torch.nn.ReLU(),
            torch.nn.Linear(net_size, net_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(net_size, n_action)
        )
        self.rewards=[]
        self.gamma=0.99
        self.replay_buffer=ExperienceReplay(100000)
        self.optim=torch.optim.Adam(self.model.parameters(),lr=0.001)

    def train(self):
        self.optim.zero_grad()
        batch=self.sample(batch_size)
        obs, act, rew, obs_n, done=zip(*[(trans['obs'], trans['act'], trans['rew'],trans['obs_n'], trans['done']) for trans in batch])
        obs=torch.from_numpy(np.stack(obs, axis=0))
        prim_act=self.model(obs)
        indices=torch.tensor(act).unsqueeze(-1)
        choice=torch.gather(prim_act, dim=1, index=indices)

        with torch.no_grad():

            obs_n = torch.from_numpy(np.stack(obs_n, axis=0))
            target_act=self.target(obs_n)
            target_max=torch.max(target_act, dim=1)[0]
            expected=torch.tensor(rew, dtype=torch.float32) + self.gamma * target_max
        loss = self.lose(choice, expected.unsqueeze(-1))
        loss.backward()
        self.optim.step()
    def update_target(self):
        for target_parameters, model_parameters in zip(self.target.parameters(),self.model.parameters()):
            target_parameters.data.copy_(.05*model_parameters.data+.95*target_parameters.data)

    def act(self, state):
        if random.random() < eps:
            return env.action_space.sample()
        else:
            q_action=self.model(torch.from_numpy(state).unsqueeze(0))
            return torch.argmax(q_action).item()
    def add(self, obs, act, rew, obs_n, done):
        self.replay_buffer.experience(obs, act, rew, obs_n, done)
    def sample(self, size):
        return self.replay_buffer.sample(size)
    def buf_size(self):
        return self.replay_buffer.buffer_size()


agent=DQN(env.observation_space.shape[0], env.action_space.n)
total_reward_episode=[]
do_render=True
print_rate=20
eps=0.99
eps_min=0.05
batch_size=32

def q_learning(n_ep=1000):
    global eps, eps_min
    for episode in range(n_ep):
        state=env.reset()
        tot_rew=0
        done=False
        if episode % print_rate ==0 and episode !=0:
            print(f'episode: {episode} | avg_rew: {np.mean(total_reward_episode[-20:])} | eps: {eps}')
        while not done:
            action=agent.act(state)
            state_n, rew, done, _=env.step(action)
            tot_rew+=rew
            agent.add(state, action, rew, state_n, done)
            if agent.buf_size() > batch_size:
                agent.train()
                agent.update_target()

            state=state_n.copy()
            if do_render and episode !=0 and episode % print_rate==0:
                env.render()

        total_reward_episode.append(tot_rew)
        if eps > eps_min:
            eps*=.995

        else:
            eps=eps_min

q_learning()



