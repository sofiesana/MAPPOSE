from tracemalloc import start
import numpy as np



class Buffer:
    def __init__(self, size, n_agents, global_state_dim, observation_dim, hidden_state_dim=None):
        self.size = size
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim
        self.observation_dim = observation_dim
        self.hidden_state_dim = hidden_state_dim

        self.current_index = 0
        self.new_episode_indices = []

        self.global_states = np.zeros((size, n_agents, *global_state_dim))
        self.observations = np.zeros((size, n_agents, observation_dim))
        self.actions = np.zeros((size, n_agents))
        self.rewards = np.zeros((size, n_agents))
        self.next_observations = np.zeros((size, n_agents, observation_dim))
        self.dones = np.zeros((size, n_agents), dtype=bool)
        self.hidden_states = np.zeros((size, n_agents, hidden_state_dim))

        self.buffer_filled = False

    def print_attributes(self):
        print("############## Buffer Attributes:")
        print(f"Buffer size: {self.size}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Global state dimension: {self.global_state_dim}")
        print(f"Observation dimension: {self.observation_dim}")
        print(f"Current index: {self.current_index}")
        
    def store_transitions(self, global_states, observations, actions, rewards, next_observations, dones, hidden_states):
        self.global_states[self.current_index] = global_states
        self.observations[self.current_index] = observations
        self.actions[self.current_index] = actions
        self.rewards[self.current_index] = rewards
        self.next_observations[self.current_index] = next_observations
        self.dones[self.current_index] = dones
        self.hidden_states[self.current_index] = hidden_states
        
        if dones:
            # print("Storing new episode index at:", self.current_index)
            self.new_episode_indices.append(self.current_index)

        if not self.buffer_filled:
            if self.current_index == self.size - 1:
                self.buffer_filled = True
                self.current_index = 0
            else:
                self.current_index = (self.current_index + 1) % self.size
        else:
            self.current_index = (self.current_index + 1) % self.size

        if self.current_index in self.new_episode_indices:
            # new episode has been overwritten, remove from list
            self.new_episode_indices.remove(self.current_index)

    def increase_index(self):
        self.current_index = (self.current_index + 1) % self.size
        if self.current_index < self.size:
            self.current_index += 1

    def store_single_agent_transition(self, agent_index, global_state, observation, action, reward, next_observation, done, hidden_state):
        self.global_states[self.current_index, agent_index] = global_state
        self.observations[self.current_index, agent_index] = observation
        self.actions[self.current_index, agent_index] = action
        self.rewards[self.current_index, agent_index] = reward
        self.next_observations[self.current_index, agent_index] = next_observation
        self.dones[self.current_index, agent_index] = done
        self.hidden_states[self.current_index, agent_index] = hidden_state

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
            max_index = self.size
        else:
            max_index = self.current_index

        valid_starts = []
        # print(" terminal indices:", self.new_episode_indices)
        for start in range(max_index):
            # Build window indices with wrapping
            window = [(start + i) % self.size for i in range(window_size)]
            # print(window)
            if not self.buffer_filled and (start + window_size > self.current_index):
                # print("window:", window, "skipped because it exceeds current_index")
                continue
            actual_new_episode_indices = [((idx + 1) % self.size) for idx in self.new_episode_indices]
            if any(idx in actual_new_episode_indices for idx in window[1:]):
                # print("window:", window, "skipped because it crosses episode boundary", actual_new_episode_indices)
                continue
            valid_starts.append(start)
        if not valid_starts:
            raise ValueError("No valid sequence found")
        # print("Valid start indices for window sampling:", valid_starts)
        return valid_starts

    def sample_agent_batch(self, agent_index, batch_size, window_size=10):
        if window_size > self.size:
            raise ValueError(f"Window size, {window_size}, cannot be larger than buffer size, {self.size}.")
        
        if not self.buffer_filled and self.current_index < window_size:
            print("Not enough samples in buffer yet to sample the requested batch size.")
            return None # this may need to be handled differently
        
        
        # get batch of length window_size
        batch_indices = np.zeros((batch_size, window_size), dtype=int)
        print("batch size:", batch_size
              , "window size:", window_size, "global state dim:", self.global_state_dim
              , "observation dim:", self.observation_dim)

        valid_starts = self.get_valid_start_indices_for_window(window_size)
        for b in range(batch_size):
            start_index = np.random.choice(valid_starts)
            window = [(start_index + i) % self.size for i in range(window_size)]
            batch_indices[b] = window
        # print("Sampled batch indices for agent", agent_index, ":\n", batch_indices)

        return (self.global_states[batch_indices, agent_index],
                self.observations[batch_indices, agent_index],
                self.actions[batch_indices, agent_index],
                self.rewards[batch_indices, agent_index],
                self.next_observations[batch_indices, agent_index],
                self.dones[batch_indices, agent_index], 
                self.hidden_states[batch_indices, agent_index])
    
    def get_episode_state_and_rewards(self, episode, timestep):
        """
        Get sum of rewards for all agents, and the global state at a given timestep (and the previous timestep)
        within a specified episode.
        """


        episode_start = self.new_episode_indices[episode]
        episode_end = self.new_episode_indices[episode + 1]  # next episode start

        episode_length = episode_end - episode_start
        if timestep >= episode_length:
            raise ValueError(f"Timestep {timestep} out of range for episode {episode} (max {episode_length - 1}).")

        # --- Compute absolute buffer index ---
        index = episode_start + timestep

        # --- Get state at timestep t ---
        state_t = self.global_states[index]

        # --- Get previous state (if exists) ---
        state_t_minus_1 = self.global_states[index - 1] if timestep > 0 else None

        # --- Sum of rewards across all agents ---
        sum_rewards = np.sum(self.rewards[index])


        return sum_rewards, state_t, state_t_minus_1





