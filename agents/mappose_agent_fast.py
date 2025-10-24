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
    def __init__(self, state_dim, obs_dim, n_actions, num_agents, batch_size, lr, discount_factor, seq_size=32, beta = 1, epsilon = 0.2, entropy_coeff=0.005):
        super().__init__(n_actions, lr, discount_factor)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.beta = beta
        self.seq_size = seq_size
        self.entropy_coeff = entropy_coeff

        self.critic_model = Critic_Network(map_size=state_dim)
        self.critic_model_target = Critic_Network(map_size=state_dim)
        self.critic_model_target.load_state_dict(self.critic_model.state_dict())
        self.actor_models_list = [ActorNetwork(input_size=obs_dim, n_actions=n_actions) for _ in range(num_agents)]
        self.actor_prev_models_list = [copy.deepcopy(actor_model) for actor_model in self.actor_models_list]
        self.critic_model.to(self.device)
        self.critic_model_target.to(self.device)
        [actor_model.to(self.device) for actor_model in self.actor_models_list]
        [prev_actor_model.to(self.device) for prev_actor_model in self.actor_prev_models_list]
        self.optimizer_critic = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.optimizers_actor_list = [optim.Adam(actor_model.parameters(), lr=self.learning_rate) for actor_model in self.actor_models_list]

    def update_prev_actor_models(self):
        for agent_idx in range(self.num_agents):
            self.actor_prev_models_list[agent_idx].load_state_dict(self.actor_models_list[agent_idx].state_dict())

    def choose_action(self, obs_list, hidden_list):
        action_list = []
        log_prob_list = []
        next_hidden_states = []
        with torch.no_grad():
            obs_list = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
            hidden_list = torch.tensor(np.array(hidden_list), dtype=torch.float32).to(self.device)
            for actor, obs, h in zip(self.actor_models_list, obs_list, hidden_list):
                obs_input = obs.unsqueeze(0).unsqueeze(0)
                h_input = h.unsqueeze(0).unsqueeze(0).contiguous()
                logits, h_next = actor(obs_input, h_input)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                action_list.append(action.item())
                log_prob_list.append(log_prob.item())
                next_hidden_states.append(h_next.squeeze().cpu().numpy())
        return action_list, log_prob_list, next_hidden_states

    def choose_random_action(self):
        action_list = np.random.randint(low=0, high=self.n_actions, size=self.num_agents)
        return action_list, np.zeros(2), None

    def get_prob_action_given_obs(self, action_seq, obs_seq, policy, h_init=None, get_entropy=False):
        h_init = h_init.contiguous() if h_init is not None else None
        logits, _ = policy(obs_seq, h_init)
        distribution = torch.distributions.Categorical(logits=logits)
        action_seq = action_seq.long()
        log_probs_list = distribution.log_prob(action_seq)
        if get_entropy:
            entropy = distribution.entropy()
            return log_probs_list, entropy
        return log_probs_list

    def convert_sample_to_tensor(self, array, dtype=torch.float32):
        for i in range(len(array)):
            array[i] = torch.tensor(array[i], dtype=dtype).to(self.device)
        return array

    def get_clipped_objective(self, actions_seq, obs_seq, advantages, hidden_states, old_log_probs, agent_idx, return_log_probs=False):
        if hidden_states is not None:
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states = hidden_states[0, :, :].unsqueeze(dim=0)
            hidden_states = hidden_states.detach()
        current_log_probs, entropies = self.get_prob_action_given_obs(actions_seq, obs_seq, self.actor_models_list[agent_idx], h_init=hidden_states, get_entropy=True)
        if old_log_probs is None:
            with torch.no_grad():
                old_log_probs = self.get_prob_action_given_obs(actions_seq, obs_seq, self.actor_prev_models_list[agent_idx], h_init=hidden_states)
        else:
            old_log_probs = old_log_probs.detach()
        policy_ratio = torch.exp(current_log_probs - old_log_probs)
        individual_clipped_obj = torch.min(policy_ratio * advantages, policy_ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * advantages)
        if return_log_probs:
            return individual_clipped_obj, entropies, current_log_probs
        return individual_clipped_obj, entropies

    def compute_GAE_from_index(self, buffer: Buffer, start_idx, critic_model, gamma=0.99, lamda=0.95):
        sum_rewards, states, next_states, dones = buffer.get_timestep_state_and_rewards(start_idx)
        sum_rewards, states, next_states, dones = self.convert_sample_to_tensor([sum_rewards, states, next_states, dones])
        with torch.no_grad():
            value_t = critic_model(states).squeeze(-1)
            value_t_plus_1 = critic_model(next_states).squeeze(-1)
        value_t_plus_1 = value_t_plus_1 * (1.0 - dones)
        deltas = sum_rewards + gamma * value_t_plus_1 - value_t
        advantages = torch.zeros_like(deltas)
        last_adv = 0
        for t in reversed(range(len(deltas))):
            last_adv = deltas[t] + gamma * lamda * (1 - dones[t]) * last_adv
            advantages[t] = last_adv
        return advantages[:self.seq_size]

    def compute_all_GAEs(self, buffer: Buffer, critic_model, gamma=0.99, lamda=0.99):
        states_list, rewards_sum_list = buffer.get_all_states_and_summed_rewards()
        states_tensor, rewards_sum_tensor = self.convert_sample_to_tensor([states_list, rewards_sum_list])
        advantages_list_over_episodes = []
        values_list_over_episodes = []
        for states, rewards_sums in zip(states_tensor, rewards_sum_tensor):
            with torch.no_grad():
                values = critic_model(states).squeeze(-1)
            bootstrap_value = torch.tensor([0.0], device=values.device)
            values_b = torch.cat([values, bootstrap_value])
            deltas = (rewards_sums + (gamma * values_b[1:])) - values_b[:-1]
            advantages = torch.zeros_like(deltas)
            last_adv = 0
            for t in reversed(range(len(deltas))):
                last_adv = deltas[t] + gamma * lamda * last_adv
                advantages[t] = last_adv
            advantages_list_over_episodes.append(advantages)
            values_list_over_episodes.append(values)
        return advantages_list_over_episodes, values_list_over_episodes

    def learn(self, buffer: Buffer):
        actor_loss_list = []
        agent_batches_list = [buffer.get_all_agent_batches(n, self.batch_size, window_size=self.seq_size) for n in range(self.num_agents)]
        advantages_over_episodes, _ = self.compute_all_GAEs(buffer, self.critic_model)
        episode_length = buffer.end_episode_indices[0] + 1
        for batch_idx in range(len(agent_batches_list[0])):
            for n in range(self.num_agents):
                states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n, start_idxs_n = agent_batches_list[n][batch_idx]
                states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n = self.convert_sample_to_tensor([states, obs_seq_n, actions_seq_n, rewards_seq_n, dones_seq_n, hidden_states_seq_n, old_log_probs_seq_n], dtype=torch.float32)
                # Vectorized advantage assignment
                start_idxs_n = np.array(start_idxs_n)
                episode_idxs = start_idxs_n // episode_length
                within_ep_idxs = start_idxs_n % episode_length
                advantages_n = torch.zeros((len(start_idxs_n), self.seq_size), dtype=torch.float32).to(self.device)
                for i, (ep_idx, st_idx) in enumerate(zip(episode_idxs, within_ep_idxs)):
                    adv_traj = advantages_over_episodes[ep_idx][st_idx : st_idx + self.seq_size]
                    advantages_n[i, :len(adv_traj)] = adv_traj
                individual_clipped_obj, entropies = self.get_clipped_objective(actions_seq_n, obs_seq_n, advantages_n, hidden_states_seq_n, old_log_probs_seq_n, n)
                individual_loss = -(individual_clipped_obj + self.entropy_coeff * entropies)
                total_actor_loss = torch.mean(individual_loss)
                actor_loss_list.append(total_actor_loss.item())
                self.optimizers_actor_list[n].zero_grad()
                total_actor_loss.backward()
                self.optimizers_actor_list[n].step()
            reward_to_go_n = torch.tensor(buffer.get_rewards_to_go(window_size=self.seq_size, start_idxs=start_idxs_n), dtype=torch.float32).to(self.device)
            values_seq = self.critic_model(states).squeeze(-1)
            critic_loss = F.mse_loss(values_seq, reward_to_go_n)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
        return actor_loss_list, critic_loss.item()

    def save_all_models(self, path):
        os.makedirs(path) if not os.path.exists(path) else None
        for n in range(self.num_agents):
            torch.save(self.actor_models_list[n].state_dict(), f"{path}/actor_model_{n}.pth")
        torch.save(self.critic_model.state_dict(), f"{path}/critic_model.pth")
