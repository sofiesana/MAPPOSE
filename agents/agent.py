"""Base class for a RL agent"""

from abc import ABC, abstractmethod
import numpy as np

import random
from typing import Tuple
import torch

class Agent(ABC):
    def __init__(
        self,
        memory_size: int,
        state_dimensions: Tuple[int, int, int],
        n_actions: int,
        learning_rate = 0.0001,
        discount_factor = 0.99,
        # Add any other arguments you need here
        # e.g. learning rate, discount factor, etc.
    ) -> None:
        """!
        Initializes the agent.
        Agent is an abstract class that should be inherited by any agent that
        wants to interact with the environment. The agent should be able to
        store transitions, choose actions based on observations, and learn from the
        transitions.

        @param memory_size (int): Size of the memory buffer
        @param state_dimensions (int): Number of dimensions of the state space
        @param n_actions (int): Number of actions the agent can take
        """

        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.state_buffer = np.zeros((self.memory_size, state_dimensions), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, state_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros((self.memory_size, n_actions), dtype=np.float32)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.memory_size, dtype=bool)
        self.transition_number = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # For mac

    def store_transition(
        self,
        state: np.ndarray,
        action: int, # Is this always an int?
        reward: float,
        new_state: np.ndarray,
        done: bool
    ) -> None:
        """!
        Stores the state transition for later memory replay.
        Make sure that the memory buffer does not exceed its maximum size.

        Hint: after reaching the limit of the memory buffer, maybe you should start overwriting
        the oldest transitions?

        @param state        (list): Vector describing current state
        @param action       (int): Action taken
        @param reward       (float): Received reward
        @param new_state    (list): Newly observed state.
        """

        transition_idx = self.transition_number % self.memory_size # Restart idx if at max size

        self.state_buffer[transition_idx] = state
        self.new_state_buffer[transition_idx] = new_state
        self.action_buffer[transition_idx] = action
        self.reward_buffer[transition_idx] = reward
        self.terminal_buffer[transition_idx] = done
        self.transition_number += 1

    def sample_buffer(
        self,
        batch_size: int
    ):
        rand_indices = np.random.randint(0, self.get_current_buffer_size(), size=batch_size)

        sampled_states = self.state_buffer[rand_indices]
        sampled_next_states = self.new_state_buffer[rand_indices]
        sampled_actions = self.action_buffer[rand_indices]
        sampeld_rewards = self.reward_buffer[rand_indices]
        sampled_dones = self.terminal_buffer[rand_indices]

        return sampled_states, sampled_next_states, sampled_actions, sampeld_rewards, sampled_dones

    def get_current_buffer_size(
        self
    ) -> int: 

        return min(self.transition_number, self.memory_size)
    
    def state_numpy_to_tensor(self, states):
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        return states
    
    def action_numpy_to_tensor(self, action):
        action = np.array(action)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)

        return action
        

    @abstractmethod
    def choose_action(
        self,
        observation: np.ndarray
    ) -> int: # Is this always an int?
        """!
        Abstract method that should be implemented by the child class, e.g. DQN or DDQN agents.
        This method should contain the full logic needed to choose an action based on the current state.
        Maybe you can store the neural network in the agent class and use it here to decide which action to take?

        @param observation (np.ndarray): Vector describing current state

        @return (int): Action to take
        """

        return 0

    @abstractmethod
    def learn(self) -> None:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        """

        pass


# class Agent(ABC):
#     def __init__(
#         self, # ...
#     ) -> None:
#         pass
    
#     @abstractmethod
#     def choose_action(
#         self, observation: np.ndarray, # ...
#     ) -> int: # Is this always an int?
#         pass

#     @abstractmethod
#     def learn(self) -> None:
#         pass
