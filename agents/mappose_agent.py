import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy

from agents.agent import Agent
from buffer import Buffer
from network import ActorNetwork, Critic_Network

class MAPPOSE(Agent):
    def __init__(self, state_dim, obs_dim, n_actions, num_agents, batch_size, lr, discount_factor, seq_size=16, beta = 1, epsilon = 0.1, tau=0.005):
        super().__init__(n_actions, lr, discount_factor)
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
        self.seq_size = seq_size

        # Initialize the state-value networks and their target networks
        self.critic_model = Critic_Network(map_size=state_dim)

        self.critic_model_target = Critic_Network(map_size=state_dim)
        self.critic_model_target.load_state_dict(self.critic_model.state_dict())

        # Initialize policy network for each agent
        self.actor_models_list = [ActorNetwork(input_size=obs_dim, n_actions=n_actions) for _ in range(num_agents)]
        self.actor_prev_models_list = [copy.deepcopy(actor_model) for actor_model in self.actor_models_list]

        # Set to device
        self.critic_model.to(self.device)
        self.critic_model_target.to(self.device)
        [actor_model.to(self.device) for actor_model in self.actor_models_list]
        [prev_actor_model.to(self.device) for prev_actor_model in self.actor_prev_models_list]

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
   
    def choose_action(self, obs_list, hidden_list):
        """Choose actions for all agents based on their observations
        Args:
            - obs_list: list of observations for each agent
        Returns:
            - action_list: list of actions for each agent
            - log_prob_list: list of log probabilities of the chosen actions
        """

        action_list = []
        log_prob_list = []
        next_hidden_states = []

        # Convert to tensors
        obs_list = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        hidden_list = torch.tensor(np.array(hidden_list), dtype=torch.float32).to(self.device)

        for actor, obs, h in zip(self.actor_models_list, obs_list, hidden_list):
            obs_input = torch.unsqueeze(obs, dim=0).unsqueeze(dim=0)  # Add batch and seq dimensions
            h_input = torch.unsqueeze(h, dim=0).unsqueeze(dim=0)  # Add num_layers dimension

            logits, h_next = actor(obs_input, h_input)

            distribution = torch.distributions.Categorical(logits=logits)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            action_list.append(action.item())
            log_prob_list.append(log_prob.item())
            next_hidden_states.append(h_next.squeeze().squeeze().detach().cpu().numpy())  # Remove singular dimension, convert to numpy & detach

        return action_list, log_prob_list, next_hidden_states
    
    def choose_random_action(self):
        """Choose random action for all agents
        Returns:
            - action_list: list of random actions for each agent
        """

        action_list = np.random.randint(low=0, high=self.n_actions, size=self.num_agents)
        
        return action_list
    
    def get_prob_action_given_obs(self, action_seq, obs_seq, policy, h_init=None, get_entropy=False):
        """Get the probabilty of taking action in action_list given being obs using given policy
        Args:
            - obs_seq: list of observations
            - action_seq: list of actions
            - policy: actor network to use as policy
        Return:
            - prob_list: list of probabilities of selecting the given actions
        """

        logits, _ = policy(obs_seq, h_init)
        distribution = torch.distributions.Categorical(logits=logits)

        log_probs_list = distribution.log_prob(action_seq)

        if get_entropy:
            entropy = distribution.entropy()

            return log_probs_list, entropy

        return log_probs_list
    

    def compute_rewards_to_go(self, rewards, dones, gamma=None):
        """
        Compute discounted rewards-to-go for each trajectory in the batch.

        Args:
            - rewards: Tensor of shape [batch_size, seq_len]
            - dones: Tensor of shape [batch_size, seq_len] 
            - gamma: discount factor
        Returns:
            - returns: Tensor of rewards-to-go from each timestep
        """

        if gamma is None:
            gamma = self.discount_factor

        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards).to(self.device)
        running_return = torch.zeros(batch_size).to(self.device)

        # Compute backward through time
        for t in reversed(range(seq_len)):
            # Reset running return where episode ended
            running_return = rewards[:, t] + gamma * running_return * (1 - dones[:, t])
            returns[:, t] = running_return

        return returns
    

    def compute_GAE_advantages(self, rewards, values, dones, gamma=None, lamda=0.95):
        """Compute Generalized Advantage Estimation (GAE) advantages - for each trajectory in the batch.
        Args:
            - rewards: Tensor of shape [batch_size, seq_len]
            - values: Tensor of shape [batch_size, seq_len]
            - dones: Tensor of shape [batch_size, seq_len]
            - gamma: discount factor
            - lamda: GAE lambda parameter
        Returns:
            - advantages: Tensor of GAE advantages
        """

        if gamma is None:
            gamma = self.discount_factor

        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = torch.zeros(batch_size).to(self.device)

        for t in reversed(range(seq_len)):
            mask = 1.0 - dones[:, t]
            last_advantage = last_advantage * mask
            delta = rewards[:, t] + gamma * values[:, t + 1] * mask - values[:, t]
            last_advantage = delta + gamma * lamda * last_advantage
            advantages[:, t] = last_advantage

        return advantages # shape [batch_size, seq_len]



    
    def learn(self, buffer: Buffer):
        """Update values of centralized value function and individual actors using shared experience
        Args:
            - buffer: sequential buffer for all agents
        """

        torch.autograd.set_detect_anomaly(True)

        ### Update Actor Networks ###
        for n in range(self.num_agents):
            print("Updating agent", n)
            states, obs_seq_n, actions_seq_n, rewards_seq_n, _, dones_seq_n, hidden_states_seq_n = buffer.sample_agent_batch(n, self.batch_size, window_size=self.seq_size) # Get batch of agent n's trajectories, should be shape [batch_size, seq_len, ...]

            print("States shape: ", states.shape)
            print("Obs shape: ", obs_seq_n.shape)
            print("Actions shape: ", actions_seq_n.shape)
            print("Rewards shape: ", rewards_seq_n.shape)
            print("Hiddens shape: ", hidden_states_seq_n.shape)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            obs_seq_n = torch.tensor(obs_seq_n, dtype=torch.float32).to(self.device)
            actions_seq_n = torch.tensor(actions_seq_n, dtype=torch.float32).to(self.device)
            rewards_seq_n = torch.tensor(rewards_seq_n, dtype=torch.float32).to(self.device)
            dones_seq_n = torch.tensor(dones_seq_n, dtype=torch.float32).to(self.device)
            hidden_states_seq_n = torch.tensor(hidden_states_seq_n, dtype=torch.float32).to(self.device)


            # TODO: Compute advantages using critic (GAE)

            # Compute reward-to-go
            print(rewards_seq_n.shape, dones_seq_n.shape)
            reward_to_go = self.compute_rewards_to_go(rewards_seq_n, dones_seq_n)  # shape: [batch_size, seq_len]

            # Get state values from critic
            with torch.no_grad():
                values_seq, _ = self.critic_model(states)      
                values_seq = values_seq.squeeze(-1)             

            # Compute advantages
            advantages = reward_to_go - values_seq

            ## Get Individual Loss ##
            current_log_probs, entropies = self.get_prob_action_given_obs(actions_seq_n, obs_seq_n, self.actor_models_list[n], get_entropy=True)
            with torch.no_grad():
                old_log_probs = self.get_prob_action_given_obs(actions_seq_n, obs_seq_n, self.actor_prev_models_list[n])
            policy_ratio = torch.exp(current_log_probs - old_log_probs)
            individual_clipped_obj = torch.min(policy_ratio * advantages, policy_ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * advantages)
            individual_loss = -(individual_clipped_obj + entropies)

            ## Get Shared Experience Loss ##
            total_shared_loss = 0
            for not_n in range(self.num_agents):
                if not_n != n: # For each other agent
                    print("Updating Shared Experience of Agent: ", not_n)
    
                    states, obs_seq_not_n, actions_seq_not_n, rewards_seq_not_n, _, dones_seq_not_n, hidden_states_seq_not_n = buffer.sample_agent_batch(not_n, self.batch_size, window_size=self.seq_size) # Get batch of agent n's trajectories, should be shape [batch_size, seq_len, ...]

                    states = torch.tensor(states, dtype=torch.float32).to(self.device)
                    obs_seq_not_n = torch.tensor(obs_seq_not_n, dtype=torch.float32).to(self.device)
                    actions_seq_not_n = torch.tensor(actions_seq_not_n, dtype=torch.float32).to(self.device)
                    rewards_seq_not_n = torch.tensor(rewards_seq_not_n, dtype=torch.float32).to(self.device)
                    dones_seq_not_n = torch.tensor(dones_seq_not_n, dtype=torch.float32).to(self.device)
                    hidden_states_seq_not_n = torch.tensor(hidden_states_seq_not_n, dtype=torch.float32).to(self.device)

                    advantages = torch.ones((self.batch_size, self.seq_size)).to(self.device) 

                    current_log_probs_agent_n, entropies_agent_n = self.get_prob_action_given_obs(actions_seq_not_n, obs_seq_not_n, self.actor_models_list[n], get_entropy=True)
                    with torch.no_grad():
                        current_log_probs_agent_not_n, entropies_agent_not_n = self.get_prob_action_given_obs(actions_seq_not_n, obs_seq_not_n, self.actor_models_list[not_n], get_entropy=True)
                        old_log_probs = self.get_prob_action_given_obs(actions_seq_not_n, obs_seq_not_n, self.actor_prev_models_list[n])

                    policy_ratio = torch.exp(current_log_probs_agent_n - old_log_probs) # Agent n's policy ratio using agent not_n's trajetories
                    individual_clipped_obj = torch.min(policy_ratio * advantages, policy_ratio.clamp(1.0-self.epsilon, 1.0+self.epsilon) * advantages)

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


        ### Update Critic Network ###
        values_seq, _ = self.critic_model(states)
        values_seq = values_seq.squeeze(-1)         
        critic_loss = F.mse_loss(values_seq, reward_to_go)


        # Optimize critic network
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        ## Update target network
        # self.update_target_model()

