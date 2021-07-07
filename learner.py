import numpy as np
import torch
from experience_replay import ExperienceReplay
import torch.optim as optim
import torch.distributions
import torch.nn.functional

class Reinforce(torch.nn.Module):
    def __init__(self, args, agent_obs_shape, agent_act_shape):
        self.args=args
        self.device=args.device
        super(Reinforce, self).__init__()
        self.fc1=torch.nn.Linear(agent_obs_shape, args.net_size)
        self.fc2=torch.nn.Linear(args.net_size, args.net_size)
        self.fcout=torch.nn.Linear(args.net_size, agent_act_shape)
        self.act=torch.relu
        self.epsilon_final=0.05
        self.tau=args.tau
        self.gamma=args.gamma
        self.optimizer=optim.Adam(self.parameters())
        self.action_space=args.env.action_space
        self.log_probs=[]
        self.rewards=[]

    def forward(self, obs):
        out=self.act(self.fc1(obs))
        out=self.act(self.fc2(out))
        out=self.fcout(out)
        softmax=torch.nn.functional.softmax(out, dim=1)
        return softmax









