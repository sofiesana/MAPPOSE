import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.agent import Agent
from network import QNetwork, PolicyNetwork

class SAC(Agent):
    def __init__(self, memory_size, state_dimensions, n_actions, batch_size, lr, alpha_lr, discount_factor, action_scale_factor=1, tau=0.005):
        super().__init__(memory_size, state_dimensions, n_actions, lr, discount_factor)
        self.batch_size = batch_size
        self.action_scale_value = action_scale_factor
        self.tau = tau
        self.alpha_lr = alpha_lr

        # Initialize the q-value networks and their target networks
        self.q_model_1 = QNetwork(num_input_neurons=state_dimensions + n_actions, hidden_layer_sizes=[256, 256])
        self.q_model_2 = QNetwork(num_input_neurons=state_dimensions + n_actions, hidden_layer_sizes=[256, 256])

        self.q_model_1_target = QNetwork(num_input_neurons=state_dimensions + n_actions, hidden_layer_sizes=[256, 256])
        self.q_model_2_target = QNetwork(num_input_neurons=state_dimensions + n_actions, hidden_layer_sizes=[256, 256])
        self.q_model_1_target.load_state_dict(self.q_model_1.state_dict())
        self.q_model_2_target.load_state_dict(self.q_model_2.state_dict())

        # Initialize polcy network
        self.policy_model = PolicyNetwork(num_input_neurons=state_dimensions, hidden_layer_sizes=[256,256], num_classes=n_actions)

        # Initialize alpha parameter
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.target_entropy = -1 * n_actions

        # Set to device
        self.q_model_1.to(self.device)
        self.q_model_2.to(self.device)
        self.q_model_1_target.to(self.device)
        self.q_model_2_target.to(self.device)
        self.policy_model.to(self.device)

        self.optimizer_q_1 = optim.Adam(self.q_model_1.parameters(), lr=self.learning_rate) # Using Adam since that is was is used in the paper
        self.optimizer_q_2 = optim.Adam(self.q_model_2.parameters(), lr=self.learning_rate)
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def get_alpha(self):
        return self.log_alpha.exp().item()

    def update_target_model(self): 
        target_1_state_dict = self.q_model_1_target.state_dict()
        target_2_state_dict = self.q_model_2_target.state_dict()
            
        q_model_1_state_dict = self.q_model_1.state_dict()
        q_model_2_state_dict = self.q_model_2.state_dict()

        for key in q_model_1_state_dict:
            target_1_state_dict[key] = q_model_1_state_dict[key] * self.tau + target_1_state_dict[key] * (1-self.tau)

        for key in q_model_2_state_dict:
            target_2_state_dict[key] = q_model_2_state_dict[key] * self.tau + target_2_state_dict[key] * (1-self.tau)

        self.q_model_1_target.load_state_dict(target_1_state_dict)
        self.q_model_2_target.load_state_dict(target_2_state_dict)


    def get_q_value(self, states, actions, model):
        states = self.state_numpy_to_tensor(np.array(states))
        
        q_value = model.forward(states, actions)

        return q_value

    
    def choose_action(self, states):
        states = self.state_numpy_to_tensor(np.array(states))
        means, stds = self.policy_model.forward(states)

        # Sample noise for reparameterization trick
        noise = torch.normal(torch.zeros_like(means), torch.ones_like(stds)).to(self.device)  # Sample noise from standard normal distribution
        reparameterized = means + stds * noise
        normalized_action = torch.tanh(reparameterized)  # normalize the output to [-1, 1]
        normalized_and_scaled_action = normalized_action * self.action_scale_value # Scale to match action space
        
        # Log-prob
        normal = torch.distributions.Normal(means, stds)
        log_prob = normal.log_prob(reparameterized)

        # Tanh correction
        log_prob -= torch.log(1 - (normalized_action ** 2) + 0.000001)

        # Sum over action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Shape: [batch_size, 1]


        return normalized_and_scaled_action, log_prob
    
    def choose_random_action(self):
        actions = np.random.uniform(-self.action_scale_value, self.action_scale_value, size=self.n_actions)
        actions = torch.tensor(actions)

        return actions, None
    
    
    def learn(self):
        states, next_states, actions, rewards, dones = self.sample_buffer(self.batch_size) # Get batch
        
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        ### Update Q-Value Networks ###
        current_q_value_1 = self.get_q_value(states, actions, model = self.q_model_1)
        current_q_value_2 = self.get_q_value(states, actions, model = self.q_model_2)

        # Sample next actions
        with torch.no_grad():
            sampled_next_actions, log_probs = self.choose_action(next_states)
            next_q_value_1 = self.get_q_value(next_states, sampled_next_actions, model = self.q_model_1_target)
            next_q_value_2 = self.get_q_value(next_states, sampled_next_actions, model = self.q_model_2_target)

            min_next_target_q_values = torch.min(next_q_value_1, next_q_value_2)

            # Calculate final targets
            pseudo_state_values = min_next_target_q_values - self.log_alpha.exp().item() * log_probs
            target_q_values = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * self.discount_factor * pseudo_state_values

        # Calulate loss
        loss_q_1 = F.mse_loss(current_q_value_1, target_q_values.detach()) # Using MSE over Huber since that is what is used in the paper
        loss_q_2 = F.mse_loss(current_q_value_2, target_q_values.detach())

        # Optimize q-value networks
        self.optimizer_q_1.zero_grad()
        loss_q_1.backward()
        self.optimizer_q_1.step()

        self.optimizer_q_2.zero_grad()
        loss_q_2.backward()
        self.optimizer_q_2.step()

        ### Update Policy Network ###
        sampled_actions, log_probs = self.choose_action(states)
        q_value_1 = self.get_q_value(states, sampled_actions, model = self.q_model_1)
        q_value_2 = self.get_q_value(states, sampled_actions, model = self.q_model_2)
        min_q_values = torch.min(q_value_1, q_value_2)

        policy_target = min_q_values - self.log_alpha.exp().item() * log_probs

        loss_policy = -torch.mean(policy_target)

        # Optimize policy network
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        ### Update Alpha Value ###
        loss_alpha = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Optimize alpha value
        self.optimizer_alpha.zero_grad()
        loss_alpha.backward()
        self.optimizer_alpha.step()

        self.update_target_model()