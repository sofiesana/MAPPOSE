import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.agent import Agent
from network import ActorNetwork, Critic_Network

class MAPPOSE(Agent):
    def __init__(self, memory_size, state_dimensions, n_actions, batch_size, lr, alpha_lr, discount_factor, action_scale_factor=1, tau=0.005):
        super().__init__(memory_size, state_dimensions, n_actions, lr, discount_factor)

   

    
    def choose_action(self, states):
        pass
    
    def choose_random_action(self):
        pass

    
    def learn(self):
        pass
