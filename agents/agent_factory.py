from agents.agent import Agent
from agents.soft_actor_critic_agent import SAC
from agents.random_agent import Random

class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str, env, batch_size: int, buffer_size: int, lr=0.0001, alpha_lr=0.001, discount_factor=0.99, action_scale=1) -> Agent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        obs_space = env.observation_space
        state_dims = obs_space.shape[0]

        action_space = env.action_space
        print(f"Observation space: {obs_space}, Action space: {action_space}")
        num_actions = action_space.shape[0]
        print(f"Creating agent of type {agent_type} with state dimensions {state_dims} and number of actions {num_actions}")

        # Best hyperparameters found using grid search for each model set below
        if agent_type == "SAC":
            return SAC(memory_size=buffer_size, state_dimensions=state_dims, n_actions=num_actions, batch_size=batch_size, lr=lr, alpha_lr=alpha_lr, discount_factor=discount_factor, action_scale_factor=action_scale)
        elif agent_type == "Random":
            return Random(memory_size=buffer_size, state_dimensions=state_dims, n_actions=num_actions, action_scale = action_scale) # Given these variables but not used

        raise ValueError("Invalid agent type: ", agent_type)