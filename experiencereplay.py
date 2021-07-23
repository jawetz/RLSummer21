from collections import deque
import random

class ExperienceReplay(object):
    def __init__(self, max_size):
        self.size= max_size
        self.buffer=deque()

    def experience(self, obs, act, rew, obs_n, done):
        if len(self.buffer)>self.size:
            self.buffer.popleft()
        self.buffer.append({'obs':obs, 'act':act, 'rew':rew, 'obs_n':obs_n, 'done':done})

    def sample(self,batch_size):
        return random.sample(list(self.buffer), batch_size)
    def buffer_size(self):
        return len(self.buffer)