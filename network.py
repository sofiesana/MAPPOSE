import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic_Network(nn.Module): # shared value network
    """
    Inputs: sequences of concatenated agent observations.
    Outputs: scalar value per timestep (centralized training)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU for temporal dependencies (handling partial observability)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Value head: scalar output per timestep
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, joint_obs_seq, h0=None):
        """
        joint_obs_seq: (batch_size, sequence_length, input_size) where input_size = sum(obs_sizes of all agents)
        h0: initial hidden state for GRU (num_layers, batch_size, hidden_size)
        Returns:
            - values: (batch_size, sequence_length, 1)
            - h_n: GRU hidden state
        """
        out, h_n = self.gru(joint_obs_seq, h0)
        values = self.value_head(out)
        return values, h_n
    

class ActorNetwork(nn.Module):  # policy network 
    """
    Inputs: sequences of flattened observations.
    Outputs: logits for discrete actions per timestep.
    """
    def __init__(self, input_size, n_actions, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU for temporal dependencies (handling partial observability)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Policy head: outputs logits for discrete actions
        self.policy_head = nn.Linear(hidden_size, n_actions)

    def forward(self, obs_seq, h0=None):
        """
        obs_seq: (batch_size, sequence_length, input_size)
        h0: initial hidden state for GRU (num_layers, batch_size, hidden_size)
        Returns:
            - logits: (batch_size, sequence_length, n_actions)
            - h_n: GRU hidden state
        """
        out, h_n = self.gru(obs_seq, h0)
        logits = self.policy_head(out)
        return logits, h_n