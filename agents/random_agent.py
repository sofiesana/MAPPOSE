import numpy as np

from agents.agent import Agent

class Random(Agent):
    def __init__(self, memory_size, state_dimensions, n_actions, action_scale):
        super().__init__(memory_size, state_dimensions, n_actions)

        self.num_actions = n_actions
        self.action_high = action_scale
        self.action_low = action_scale * -1

    def choose_action(self, observation):
        actions = np.random.uniform(self.action_low, self.action_high, size=self.num_actions)

        return actions, None
    
    def learn(self) -> None:
        pass