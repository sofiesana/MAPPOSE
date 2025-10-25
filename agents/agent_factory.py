from agents.agent import Agent
from agents.random_agent import Random
from agents.mappose_agent import MAPPOSE

from util import get_full_state

class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env, batch_size: int, lr=0.0005, discount_factor=0.99) -> Agent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        state_dim = get_full_state(env, flatten=True).shape[0]
        obs_dim = env.observation_space[0].shape[0]
        num_agents = len(env.observation_space)
        num_actions = env.action_space[0].n

        ## For MPE env for debugging purposes
        # obs, _ = env.reset()
        # # obs is a dict like {'agent_0': obs_0, 'agent_1': obs_1, ...}
        # agent_ids = list(obs.keys())
        # num_agents = len(agent_ids)
        # # Use any agent ID to query observation/action spaces
        # sample_agent = agent_ids[0]
        # obs_dim = env.observation_space(sample_agent).shape[0]
        # num_actions = env.action_space(sample_agent).n
        # state_dim = num_agents * obs_dim
        
        print(f"Creating agent of type {agent_type} with observation dim: {obs_dim}, state dimensions {state_dim} and number of actions {num_actions}")

        # Best hyperparameters found using grid search for each model set below
        if agent_type == "MAPPOSE":
            return MAPPOSE(state_dim=state_dim, obs_dim=obs_dim, n_actions=num_actions, num_agents=num_agents, batch_size=batch_size, lr=lr, discount_factor=discount_factor)
        # elif agent_type == "Random":
        #     return Random(memory_size=buffer_size, state_dimensions=state_dims, n_actions=num_actions, action_scale = action_scale) # Given these variables but not used

        raise ValueError("Invalid agent type: ", agent_type)