import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import time

from agents.agent import Agent
from buffer import Buffer
from network import ActorNetwork, Critic_Network

class MAPPOSE(Agent):
    def __init__(self, state_dim, obs_dim, n_actions, num_agents, batch_size, lr, discount_factor, seq_size=32, beta = 1, epsilon = 0.1):
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
        """

        self.batch_size = batch_size
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


    def update_prev_actor_models(self):
        """Change the previous actor models
        """

        for agent_idx in range(self.num_agents):
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
    

    def convert_sample_to_tensor(self, array, dtype=torch.float32):
        """ Convert samples from buffer to torch tensors

        Args:
            array (_type_): array of all states, actions, rewards, etc.
            dtype (_type_, optional): data type. Defaults to torch.float32.
        """

        for i in range(len(array)):
            array[i] = torch.tensor(array[i], dtype=dtype).to(self.device)

        return array
    
    def get_clipped_objective(self, actions_seq, obs_seq, advantages, old_log_probs, agent_idx, return_log_probs=False):
        """Calculate PPO clipped objective for a given agent
        Args:
            - actions_seq: sequence of actions for agent n
            - obs_seq: sequence of observations for agent n
            - advantages: advantages computed for agent
            - agent_idx: agent index

        Returns:
            - individual_clipped_obj: clipped objective for agent n
            - entropies: entropy of the action distribution
        """

        current_log_probs, entropies = self.get_prob_action_given_obs(actions_seq, obs_seq, self.actor_models_list[agent_idx], get_entropy=True)
        if old_log_probs is None:
            with torch.no_grad():
                old_log_probs = self.get_prob_action_given_obs(actions_seq, obs_seq, self.actor_prev_models_list[agent_idx])

        policy_ratio = torch.exp(current_log_probs - old_log_probs)

        # print("Policy ratio shape:", policy_ratio.shape, "Advantages shape:", advantages.shape)

        individual_clipped_obj = torch.min(policy_ratio * advantages, policy_ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * advantages)

        if return_log_probs:
            return individual_clipped_obj, entropies, current_log_probs

        return individual_clipped_obj, entropies
    

    def compute_GAE_from_index(self, buffer: Buffer, start_idx, critic_model, gamma=0.99, lamda=0.95):
        """
        Compute GAE advantages for a single trajectory starting from start_idx to end of episode.
        Values are computed on the fly using critic_model.
        """

        advantages_list = []
        last_advantage = torch.tensor(0.0, device=self.device)

        # Get terminal timestep for this episode
        # _, _, _, terminal_idx = buffer.get_timestep_state_and_rewards(start_idx)
        sum_rewards, states, next_states, dones = buffer.get_timestep_state_and_rewards(start_idx)
        sum_rewards, states, next_states, dones = self.convert_sample_to_tensor([sum_rewards, states, next_states, dones])

        with torch.no_grad():
            value_t = critic_model(states).squeeze(-1)
            value_t_plus_1 = critic_model(next_states).squeeze(-1)

        # Zero next_value where terminal
        value_t_plus_1 = value_t_plus_1 * (1.0 - dones)

        # TD error
        deltas = sum_rewards + gamma * value_t_plus_1 - value_t

        # Compute GAE backwards (fast loop on GPU)
        advantages = torch.zeros_like(deltas)
        last_adv = 0
        for t in reversed(range(len(deltas))):
            last_adv = deltas[t] + gamma * lamda * (1 - dones[t]) * last_adv
            advantages[t] = last_adv

        # # Iterate backwards from terminal_idx to start_idx
        # for t in reversed(range(start_idx, terminal_idx + 1)):
        #     sum_rewards, state_t, state_t_plus_1, _ = buffer.get_timestep_state_and_rewards(t)
        #     state_t = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0).to(self.device)

        #     # Compute value estimates on the fly
        #     value_t = critic_model(state_t).squeeze()
        #     if state_t_plus_1 is not None:
        #         state_t_plus_1 = torch.tensor(state_t_plus_1, dtype=torch.float32).unsqueeze(0).to(self.device)
        #         value_t_plus_1 = critic_model(state_t_plus_1).squeeze()
        #     else:
        #         value_t_plus_1 = torch.zeros_like(value_t)

        #     # TD error
        #     delta = sum_rewards + gamma * value_t_plus_1 - value_t

        #     # GAE advantage
        #     last_advantage = delta + gamma * lamda * last_advantage
        #     advantages_list.insert(0, last_advantage)

        # # Stack to tensor
        # advantages_tensor = torch.stack(advantages_list[:self.seq_size])  # Limit to seq_size

        # return advantages_tensor

        return advantages[:self.seq_size]


    
    def learn(self, buffer: Buffer):
        """Update values of centralized value function and individual actors using shared experience
        Args:
            - buffer: sequential buffer for all agents
        Returns:
            - loss_list: list of actor losses for each agent
        """

        actor_loss_list = []

        agent_batches_list = []
        for n in range(self.num_agents):
            agent_batches_list.append(buffer.get_all_agent_batches(n, self.batch_size, window_size=self.seq_size))

        for batch_idx in range(len(agent_batches_list[0])):  # For each batch
            ### Update Actor Networks ###
            for n in range(self.num_agents):
                # print("Updating agent", n)
                # start_time = time.time()
                states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n, start_idxs_n = agent_batches_list[n][batch_idx]  # Get batch of agent n's trajectories, should be shape [batch_size, seq_len, ...]
                states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n = self.convert_sample_to_tensor([states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n], dtype=torch.float32)
                
                # end_buffer_sample_time = time.time()
                # print("Sample batch time:", round(end_buffer_sample_time - start_time, 4))

                advantages_n_list = []
                # Loop over batch of trajectories (start indices)
                for start_idx in start_idxs_n:
                    # compute GAE for each trajectory individually
                    advantages_traj = self.compute_GAE_from_index(buffer, start_idx, self.critic_model)
                    advantages_n_list.append(advantages_traj)
                advantages_n = torch.nn.utils.rnn.pad_sequence(advantages_n_list, batch_first=True, padding_value=0.0) # Pad or stack to get [batch_size, seq_len]
                
                # end_advantage_compute_time = time.time()
                # print("Compute advantages time:", round(end_advantage_compute_time - end_buffer_sample_time, 4))

                ## Get Individual Loss ##
                individual_clipped_obj, entropies = self.get_clipped_objective(actions_seq_n, obs_seq_n, advantages_n, old_log_probs_seq_n, n)
                individual_loss = -(individual_clipped_obj + entropies)

                # end_individual_loss_time = time.time()
                # print("Individual loss time:", round(end_individual_loss_time - end_advantage_compute_time, 4))

                ## Get Shared Experience Loss ##
                total_shared_loss = 0
                for not_n in range(self.num_agents):
                    if not_n != n: # For each other agent
                        # print("Updating Shared Experience of Agent: ", not_n)
                        states, obs_seq_not_n, actions_seq_not_n, rewards_seq_not_n, dones_seq_not_n, hidden_states_seq_not_n, old_log_probs_seq_not_n, start_idxs_not_n = agent_batches_list[not_n][batch_idx]
                        states, obs_seq_not_n, actions_seq_not_n, rewards_seq_not_n, dones_seq_not_n, hidden_states_seq_not_n, old_log_probs_seq_not_n = self.convert_sample_to_tensor([states, obs_seq_not_n, actions_seq_not_n, rewards_seq_not_n, dones_seq_not_n, hidden_states_seq_not_n, old_log_probs_seq_not_n], dtype=torch.float32)

                        # end_other_agent_sample_time = time.time()
                        # print("Other agent batch sample time:", round(end_other_agent_sample_time - end_individual_loss_time, 4))

                        advantages_not_n_list = []
                        # Loop over batch of trajectories for agent not_n
                        for start_idx in start_idxs_not_n:
                            advantages_traj = self.compute_GAE_from_index(buffer, start_idx, self.critic_model)
                            advantages_not_n_list.append(advantages_traj)
                        # Stack/pad to get [batch_size, seq_len]
                        advantages_not_n = torch.nn.utils.rnn.pad_sequence(advantages_not_n_list, batch_first=True, padding_value=0.0)

                        # end_other_agent_advantage_time = time.time()
                        # print("Other agent advantage compute time:", round(end_other_agent_advantage_time - end_other_agent_sample_time, 4))

                        shared_clipped_obj, entropies_n, current_log_probs_agent_n = self.get_clipped_objective(actions_seq_not_n, obs_seq_not_n, advantages_not_n, None, n, return_log_probs=True)
                        shared_experience_ratio = torch.exp(current_log_probs_agent_n - old_log_probs_seq_not_n) # Importance sampling between agent n and not_n
                        shared_loss = -(shared_clipped_obj + entropies_n) * shared_experience_ratio  # TODO I do not know exactly what to do with the entropy part, ill just do it on agent n

                        # end_other_agent_loss_time = time.time()
                        # print("Other agent loss compute time:", round(end_other_agent_loss_time - end_other_agent_advantage_time, 4))

                        total_shared_loss += shared_loss

                mean_shared_loss = (total_shared_loss / (self.num_agents - 1))
                total_actor_loss = torch.mean(individual_loss + self.beta * mean_shared_loss)
                actor_loss_list.append(total_actor_loss.item())

                # TODO need to only update the prev actor model after all training on buffer, and set old policy to be one that collected the data

                # Optimize policy network
                # POTENTIAL ISSUE!! - Updating each agent one at a time could cause issues since changes prob in loss
                self.optimizers_actor_list[n].zero_grad()
                total_actor_loss.backward()
                self.optimizers_actor_list[n].step()


            ### Update Critic Network ###
            reward_to_go_not_n = torch.tensor(buffer.get_rewards_to_go(window_size=self.seq_size, start_idxs=start_idxs_not_n), dtype=torch.float32).to(self.device)  # shape: [batch_size, seq_len]
            values_seq = self.critic_model(states).squeeze(-1)
            critic_loss = F.mse_loss(values_seq, reward_to_go_not_n)


            # Optimize critic network
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        return actor_loss_list, critic_loss.item()


