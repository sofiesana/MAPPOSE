import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class Critic_Network(nn.Module): # shared value network
#     """
#     Inputs: flattened representation of the full map (global state) over time
#     Outputs: values per timestep in input sequence
#     """
#     def __init__(self, map_size, hidden_size=128, num_layers=1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # GRU for temporal dependencies (handling partial observability)
#         self.gru = nn.GRU(input_size=map_size,
#                           hidden_size=hidden_size,
#                           num_layers=num_layers,
#                           batch_first=True)

#         # Value head
#         self.value_head = nn.Linear(hidden_size, 1)

#     def forward(self, full_map_seq, h0=None):
#         """
#         full_map_seq: (batch_size, sequence_length, map_size)
#         h0: optional GRU initial hidden state (num_layers, batch_size, hidden_size)
#         Returns:
#             - values: (batch_size, sequence_length, 1)
#             - h_n: GRU hidden state
#         """
#         out, h_n = self.gru(full_map_seq, h0)
#         values = self.value_head(out)
#         return values, h_n

class Critic_Network(nn.Module):
    """
    Feedforward version of the Critic network.
    Inputs: flattened representation of the full map (global state)
    Outputs: scalar state value
    """
    def __init__(self, map_size, hidden_size=128, num_layers=2):
        super().__init__()
        layers = []

        input_dim = map_size
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        # Final value head
        layers.append(nn.Linear(hidden_size, 1))

        # use orthogonal initialization for all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.model = nn.Sequential(*layers)

    def forward(self, full_map):
        """
        full_map: (batch_size, map_size)
        Returns:
            - values: (batch_size, 1)
        """
        values = self.model(full_map)
        return values
    

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

        # use orthogonal initialization for all layers
        nn.init.orthogonal_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

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