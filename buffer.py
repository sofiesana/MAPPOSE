from tracemalloc import start
import numpy as np
import torch



class Buffer:
    def __init__(self, size, n_agents, global_state_dim, observation_dim, hidden_state_dim=None):
        self.size = size
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim
        self.observation_dim = observation_dim
        self.hidden_state_dim = hidden_state_dim

        self.current_index = 0
        self.end_episode_indices = []

        self.global_states = np.zeros((size, n_agents, *global_state_dim))
        self.observations = np.zeros((size, n_agents, observation_dim))
        self.actions = np.zeros((size, n_agents))
        self.rewards = np.zeros((size, n_agents))
        self.dones = np.zeros((size, n_agents), dtype=bool)
        self.hidden_states = np.zeros((size, n_agents, hidden_state_dim))
        self.old_log_probs = np.zeros((size, n_agents))

        self.buffer_filled = False

    def print_attributes(self):
        print("############## Buffer Attributes:")
        print(f"Buffer size: {self.size}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Global state dimension: {self.global_state_dim}")
        print(f"Observation dimension: {self.observation_dim}")
        print(f"Current index: {self.current_index}")
        
    def store_transitions(self, global_states, observations, actions, rewards, dones, hidden_states, log_probs):
        self.global_states[self.current_index] = global_states
        self.observations[self.current_index] = observations
        self.actions[self.current_index] = actions
        self.rewards[self.current_index] = rewards
        self.dones[self.current_index] = dones
        self.hidden_states[self.current_index] = hidden_states
        self.old_log_probs[self.current_index] = log_probs

        if dones:
            # print("Storing new episode index at:", self.current_index)
            self.end_episode_indices.append(self.current_index)

        if not self.buffer_filled:
            if self.current_index == self.size - 1:
                self.buffer_filled = True
                self.current_index = 0
            else:
                self.current_index = (self.current_index + 1) % self.size
        else:
            self.current_index = (self.current_index + 1) % self.size

        if self.current_index in self.end_episode_indices:
            # new episode has been overwritten, remove from list
            self.end_episode_indices.remove(self.current_index)

    def increase_index(self):
        self.current_index = (self.current_index + 1) % self.size
        if self.current_index < self.size:
            self.current_index += 1

    def store_single_agent_transition(self, agent_index, global_state, observation, action, reward, done, hidden_state, log_prob):
        self.global_states[self.current_index, agent_index] = global_state
        self.observations[self.current_index, agent_index] = observation
        self.actions[self.current_index, agent_index] = action
        self.rewards[self.current_index, agent_index] = reward
        self.dones[self.current_index, agent_index] = done
        self.hidden_states[self.current_index, agent_index] = hidden_state
        self.old_log_probs[self.current_index, agent_index] = log_prob

        print("CAUTION: Storing single agent transition does NOT increment current_index.")
        print("please call increase_index() method after storing all agents' transitions for the current time step.")

    def print_buffer(self):
        # pprint buffer contents per agent, untruncated 
        print("############## Buffer Contents:")
        for ag in range(self.n_agents):
            print(f"--- Agent {ag} ---")
            for i in range(self.current_index):
                print(f"Index {i}: GS: {self.global_states[i, ag]}, Obs: {self.observations[i, ag]}, "
                      f"Act: {self.actions[i, ag]}, Rew: {self.rewards[i, ag]}, "
                      f"Next Obs: {self.next_observations[i, ag]}, Done: {self.dones[i, ag]}, "
                      f"Hidden State: {self.hidden_states[i, ag]}")
                
    def get_valid_start_indices_for_window(self, window_size):
        # Find all valid start indices (not crossing episode boundaries)
        if self.buffer_filled:
            max_index = self.size-1
        else:
            max_index = self.current_index

        valid_starts = []
        # print(" terminal indices:", self.end_episode_indices)
        for start in range(max_index):
            # Build window indices with wrapping
            window = [(start + i) % self.size for i in range(window_size)]
            # print(window)
            if not self.buffer_filled and (start + window_size > self.current_index):
                # print("window:", window, "skipped because it exceeds current_index")
                continue
            actual_end_episode_indices = [((idx + 1) % self.size) for idx in self.end_episode_indices]
            if any(idx in actual_end_episode_indices for idx in window[1:]):
                # print("window:", window, "skipped because it crosses episode boundary", actual_new_episode_indices)
                continue
            valid_starts.append(start)
        if not valid_starts:
            raise ValueError("No valid sequence found")
        # print("Valid start indices for window sampling:", valid_starts)
        return valid_starts
    
    def get_rewards_to_go(self, window_size, start_idxs, discount_factor=0.99):
        """Get rewards to go for each timestep in the window

        Args:
            window_size (int): size of sequence window
            start_idx (List): list of starting index of the window
            discount_factor (float, optional): discount factor. Defaults to 0.99.
        """
        rewards_to_go_array = np.zeros((len(start_idxs), window_size))

        # Calculate rewards to go
        for b, start_idx in enumerate(start_idxs):
            our_terminal_idx = None
            for terminal_idx in self.end_episode_indices:
                if terminal_idx >= start_idx:
                    our_terminal_idx = terminal_idx
                    break

            total_sequence_length = our_terminal_idx - start_idx + 1
            last_timestep_rewards_to_go = 0
            for t in reversed(range(total_sequence_length)):
                idx = start_idx + t
                all_agents_rewards = np.sum(self.rewards[idx])
                rewards_to_go = all_agents_rewards + discount_factor * last_timestep_rewards_to_go
                if t < window_size:
                    # print("Rewards to go at buffer index", idx, "for batch", b, "timestep", t, "is:", rewards_to_go)
                    rewards_to_go_array[b, t] = rewards_to_go
                last_timestep_rewards_to_go = rewards_to_go

        return rewards_to_go_array


    def sample_agent_batch(self, agent_index, batch_size, window_size=10):
        if window_size > self.size:
            raise ValueError(f"Window size, {window_size}, cannot be larger than buffer size, {self.size}.")
        
        if not self.buffer_filled and self.current_index < window_size:
            print("Not enough samples in buffer yet to sample the requested batch size.")
            return None # this may need to be handled differently
        
        # get batch of length window_size 
        batch_indices = np.zeros((batch_size, window_size), dtype=int) 
        
        valid_starts = self.get_valid_start_indices_for_window(window_size)
        for b in range(batch_size):
            start_index = np.random.choice(valid_starts)
            window = [(start_index + i) % self.size for i in range(window_size)]
            batch_indices[b] = window
            # get batch of length window_size
            # print("batch size:", batch_size
                #   , "window size:", window_size, "global state dim:", self.global_state_dim
                #   , "observation dim:", self.observation_dim)

        # print("Sampled batch indices for agent", agent_index, ":\n", batch_indices, "Batch shape:", batch_indices.shape)
        start_idxs = [indices[0] for indices in batch_indices]

        return (self.global_states[batch_indices, agent_index], 
                self.observations[batch_indices, agent_index], 
                self.actions[batch_indices, agent_index], 
                self.rewards[batch_indices, agent_index], 
                self.dones[batch_indices, agent_index], 
                self.hidden_states[batch_indices, agent_index], 
                self.old_log_probs[batch_indices, agent_index], start_idxs)
    
    def get_all_agent_batches(self, agent_index, batch_size, window_size=10, non_overlapping=True):
        if window_size > self.size:
            raise ValueError(f"Window size, {window_size}, cannot be larger than buffer size, {self.size}.")
        
        if not self.buffer_filled and self.current_index < window_size:
            print("Not enough samples in buffer yet to sample the requested batch size.")
            return None # this may need to be handled differently
        
        
        all_batches = []
        if non_overlapping:
            valid_starts = []
            i = 0
            while i + window_size <= self.current_index:
                #  check if i + window_size crosses a multiple of 500 (episode boundary)
                crosses_boundary = False
                for end_idx in self.end_episode_indices:
                    if i < end_idx < i + window_size - 1:
                        crosses_boundary = True
                        # print("  Skipping start index", i, "because it crosses episode boundary at", end_idx)
                        i = end_idx + 1  # jump to index after episode end
                        break
                # print("Found valid start index:", i)
                if not crosses_boundary:
                    valid_starts.append(i)
                    # Increment by window_size to ensure non-overlapping
                    i += window_size
        else:
            valid_starts = self.get_valid_start_indices_for_window(window_size)
        # Shuffle start indices
        rng = np.random.default_rng()
        rng.shuffle(valid_starts)
        for j in range(0, len(valid_starts), batch_size):
            batch_starts = valid_starts[j:j+batch_size]
            batch_indices = np.zeros((len(batch_starts), window_size), dtype=int)

            for b, start_index in enumerate(batch_starts):
                # start_index = np.random.choice(valid_starts)
                if non_overlapping:
                    window = list(range(start_index, start_index + window_size))
                    batch_indices[b] = window
                else:
                    window = [(start_index + i) % self.size for i in range(window_size)]
                    if window[0] > 99990:
                        print("Sampled start index at very high index:", window[0])
                    batch_indices[b] = window
            start_idxs = [indices[0] for indices in batch_indices]

            batch_data = (self.global_states[batch_indices, agent_index],
                          self.observations[batch_indices, agent_index],
                          self.actions[batch_indices, agent_index],
                          self.rewards[batch_indices, agent_index],
                          self.dones[batch_indices, agent_index],
                          self.hidden_states[batch_indices, agent_index],
                          self.old_log_probs[batch_indices, agent_index],
                          start_idxs)
            
            all_batches.append(batch_data)

        return all_batches  # list of batches
    
    def get_timestep_state_and_rewards(self, start_idx):
        """
        Get sum of rewards for all agents, state at timestep t and state at timestep t+1
        starting from a given index.
        """

        # Find the terminal timestep for this episode
        terminal_idx = min([idx for idx in self.end_episode_indices if idx >= start_idx])
        timesteps = list(range(start_idx, terminal_idx + 1))

        rewards = []
        states = []
        next_states = []
        dones = []

        for t in timesteps:
            # Get state at time t
            state_t = self.global_states[t, 0]  # shared global state for all agents
            states.append(state_t)

            # Get reward summed across agents
            rewards.append(np.sum(self.rewards[t]))

            # Determine if this timestep ends the episode
            done = (t == terminal_idx)
            dones.append(done)

            # Get next state
            if not done:
                next_states.append(self.global_states[t + 1, 0])
            else:
                # For terminal state, we can duplicate last state or fill with zeros
                next_states.append(np.zeros_like(state_t))

        # return sum_rewards, state_t, state_t_plus_1, terminal_idx
        return rewards, states, next_states, dones #, terminal_idx
    
    def get_all_states_and_summed_rewards(self):
        # print("End episode indices:", self.end_episode_indices)
        states_list_across_episodes = np.zeros((len(self.end_episode_indices), self.end_episode_indices[0]+1, self.global_state_dim[0])) # Shape: (num_episodes, max_episode_length, global_state_dim)
        rewards_list_across_episodes = np.zeros((len(self.end_episode_indices), self.end_episode_indices[0]+1)) # Shape: (num_episodes, max_episode_length)

        start_idx = 0
        for e, episode_end_idx in enumerate(self.end_episode_indices):
            # rewards, states, _, _ = self.get_timestep_state_and_rewards(start_idx)
            rewards_sum = np.sum(self.rewards[start_idx:episode_end_idx+1], axis=1)
            states = self.global_states[start_idx:episode_end_idx+1, 0, :]

            states_list_across_episodes[e, :len(states), :] = states
            rewards_list_across_episodes[e, :len(rewards_sum)] = rewards_sum

            start_idx = episode_end_idx + 1

        return states_list_across_episodes, rewards_list_across_episodes

