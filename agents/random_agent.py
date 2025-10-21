import numpy as np

from agents.agent import Agent

class Random(Agent):
    def __init__(self, memory_size, state_dimensions, n_actions, num_agents):
        super().__init__(memory_size, state_dimensions, n_actions)

        self.num_actions = n_actions
        self.num_agents = num_agents

    def choose_action(self, observation):
        action_list = np.random.randint(low=0, high=self.n_actions, size=self.num_agents)

        return action_list
    
    def learn(self) -> None:
        pass