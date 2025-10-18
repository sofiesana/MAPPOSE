import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy

from agents.agent import Agent
from network import ActorNetwork, Critic_Network

class MAPPOSE(Agent):
    def __init__(self, memory_size, state_dim, obs_dim, n_actions, num_agents, batch_size, lr, discount_factor, beta = 1, epsilon = 0.1, tau=0.005):
        super().__init__(memory_size, state_dim, n_actions, lr, discount_factor)
        """Initialize the MAPPOSE agents
        Args:
            - memory_size: size of the replay buffer
            - state_dim: dimension of the global state
            - obs_dim: dimension of the local observation
            - n_actions: number of discrete actions
            - num_agents: number of agents in the environment
            - batch_size: size of the training batch
            - lr: learning rate
            - discount_factor: discount factor for future rewards
            - tau: target network update parameter
        """

        self.batch_size = batch_size
        self.tau = tau
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.beta = beta

        # Initialize the state-value networks and their target networks
        self.critic_model = Critic_Network(num_input_neurons=state_dim)

        self.critic_model_target = Critic_Network(num_input_neurons=state_dim)
        self.critic_model_target.load_state_dict(self.critic_model.state_dict())

        # Initialize policy network for each agent
        self.actor_models_list = [ActorNetwork(num_input_neurons=obs_dim, n_actions=n_actions) for _ in range(num_agents)]
        self.actor_prev_models_list = [copy.deepcopy(actor_model) for actor_model in self.actor_models_list]

        # Set to device
        self.critic_model.to(self.device)
        self.critic_model_target.to(self.device)
        [actor_model.to(self.device) for actor_model in self.actor_models_list]

        self.optimizer_critic = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.optimizers_actor_list = [optim.Adam(actor_model.parameters(), lr=self.learning_rate) for actor_model in self.actor_models_list]

    def update_target_model(self): 
        """Update target network with weights of main network
        """

        target_state_dict = self.critic_model_target.state_dict()
            
        critic_model_state_dict = self.critic_model.state_dict()

        for key in critic_model_state_dict:
            target_state_dict[key] = critic_model_state_dict[key] * self.tau + target_state_dict[key] * (1-self.tau)

        self.critic_model_target.load_state_dict(target_state_dict)

    def update_prev_actor_model(self, agent_idx):
        """Change the previous actor model to the new updated model
        Args:
            - agent_idx: index of agent to update networks
        """

        self.actor_prev_models_list[agent_idx].load_state_dict(self.actor_models_list[agent_idx].state_dict())

        # for prev_actor, new_actor in zip(self.actor_prev_models_list, self.actor_models_list):
        #     prev_actor.load_state_dict(new_actor.state_dict())
   
    def choose_action(self, obs_list):
        """Choose actions for all agents based on their observations
        Args:
            - obs_list: list of observations for each agent
        Returns:
            - action_list: list of actions for each agent
            - log_prob_list: list of log probabilities of the chosen actions
        """

        action_list = []
        log_prob_list = []

        for actor, obs in zip(self.actor_models_list, obs_list):
            logits = actor(obs)

            distribution = torch.distributions.Categorical(logits=logits)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            action_list.append(action)
            log_prob_list.append(log_prob)

        return action_list, log_prob_list
    
    def choose_random_action(self):
        """Choose random action for all agents
        Returns:
            - action_list: list of random actions for each agent
        """

        action_list = np.random.randint(low=0, high=self.n_actions, size=self.num_agents)
        
        return action_list
    
    def get_prob_action_given_obs(self, action_list, obs_list, policy, get_entropy=False):
        """Get the probabilty of taking action in action_list given being obs using given policy
        Args:
            - obs_list: list of observations
            - action_list: list of actions
            - policy: actor network to use as policy
        Return:
            - prob_list: list of probabilities of selecting the given actions
        """

        logits = policy(obs_list)
        distribution = torch.distributions.Categorical(logits=logits)

        log_probs_list = distribution.log_prob(action_list)

        if get_entropy:
            entropy = distribution.entropy()

            return log_probs_list, entropy

        return log_probs_list

    
    def learn(self):
        """Update values of centralized value function and individual actors using shared experience
        """

        ### Update Actor Networks ###
        for n in range(self.num_agents):
            ### TODO Get batch of only agent n's trajectories
            states, next_states, obs_list, actions_list, rewards, dones = self.sample_buffer(self.batch_size) # Get batch of agent n's trajectories
        
            obs_list = torch.tensor(obs_list, dtype=torch.float32).to(self.device)
            actions_list = torch.tensor(actions_list, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            advantages = None # TODO still gotta do this with rewards-to-go

            ## Get Individual Loss ##
            current_log_probs, entropies = self.get_prob_action_given_obs(actions_list, obs_list, self.actor_models_list[n], get_entropy=True)
            old_log_probs = self.get_prob_action_given_obs(actions_list, obs_list, self.actor_prev_models_list[n])
            policy_ratio = torch.exp(current_log_probs - old_log_probs)
            individual_clipped_obj = min(policy_ratio * advantages, torch.clamp(policy_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages)
            individual_loss = -(individual_clipped_obj + entropies)

            ## Get Shared Experience Loss ##
            total_shared_loss = 0
            for not_n in range(self.num_agents):
                if not_n != n: # For each other agent
                    ### TODO Get batch of only agent not_n's trajectories
                    states, next_states, obs_list, actions_list, rewards, dones = self.sample_buffer(self.batch_size) # Get batch of agent not_n's trajectories

                    obs_list = torch.tensor(obs_list, dtype=torch.float32).to(self.device)
                    actions_list = torch.tensor(actions_list, dtype=torch.float32).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

                    advantages = None # TODO still gotta do this with rewards-to-go

                    current_log_probs_agent_n, entropies_agent_n = self.get_prob_action_given_obs(actions_list, obs_list, self.actor_models_list[n], get_entropy=True)
                    current_log_probs_agent_not_n, entropies_agent_not_n = self.get_prob_action_given_obs(actions_list, obs_list, self.actor_models_list[not_n], get_entropy=True)
                    old_log_probs = self.get_prob_action_given_obs(actions_list, obs_list, self.actor_prev_models_list[n])

                    policy_ratio = torch.exp(current_log_probs_agent_n - old_log_probs) # Agent n's policy ratio using agent not_n's trajetories
                    individual_clipped_obj = min(policy_ratio * advantages, torch.clamp(policy_ratio, 1.0-self.epsilon, 1.0+self.epsilon) * advantages)

                    shared_experience_ratio = torch.exp(current_log_probs_agent_n - current_log_probs_agent_not_n) # Importance sampling between agent n and not_n

                    # TODO I do not know exactly what to do with the entropy part, ill just do it on agent n
                    shared_loss = -(individual_clipped_obj + entropies_agent_n) * shared_experience_ratio

                    total_shared_loss += shared_loss

            mean_shared_loss = (total_shared_loss / (self.num_agents - 1))
            total_actor_loss = torch.mean(individual_loss + self.beta * mean_shared_loss)

            self.update_prev_actor_model(n) # Update prev network to current before optimizing current

            # Optimize policy network
            self.optimizers_actor_list[n].zero_grad()
            total_actor_loss.backward()
            self.optimizers_actor_list[n].step()

            self.update_target_model()

