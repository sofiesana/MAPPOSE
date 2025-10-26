import gymnasium as gym
import numpy as np
from collections import Counter

from buffer import Buffer
from util import get_full_state
from agents.agent_factory import AgentFactory

# --- SETTINGS ---
N_COLLECTION_EPISODES = 25
SEQ_SIZE = 16  # sequence length for training
HIDDEN_STATE_DIM = 128
BUFFER_SIZE = 5000

# ------------------------------
# Environment Setup
# ------------------------------
def make_env():
    env = gym.make("rware-tiny-2ag-v2")
    return env

# ------------------------------
# Reward shaping
# ------------------------------
def shape_rewards(env, rewards, obs, actions, info):
    """
    Shape rewards for rware-tiny-2ag-v2:
      - Step penalty: -0.01 per step
      - Pickup reward: +1.0 for picking up requested shelf
    """
    n_agents = len(rewards)
    shaped_rewards = np.array(rewards, dtype=np.float32)

    # Step penalty
    shaped_rewards -= 0.01

    # Pickup reward
    if 'picked_up_shelf' in info:
        for i in range(n_agents):
            if info['picked_up_shelf'][i]:
                shaped_rewards[i] += 1.0

    return shaped_rewards

# ------------------------------
# Run a single episode and fill buffer
# ------------------------------
def run_episode(env, agent, buffer: Buffer, mode='train'):
    obs, _ = env.reset()
    n_agents = len(obs)
    hidden_states = np.zeros((n_agents, HIDDEN_STATE_DIM))
    terminated = False
    truncated = False
    ep_return = []
    step_counter = 0

    # For enhanced debugging
    cumulative_rewards = np.zeros(n_agents)
    hidden_state_trace = [[] for _ in range(n_agents)]
    action_counts = [Counter() for _ in range(n_agents)]

    # Track previous distances for reward shaping
    prev_distances = None

    while not (terminated or truncated):
        # Choose actions
        actions, log_probs, new_hidden_states = agent.choose_action(obs, hidden_states)

        # Step environment
        new_obs, rewards, terminated, truncated, info = env.step(actions)

        # Shape rewards
        rewards = shape_rewards(env, rewards, obs, actions, info)

        # Track cumulative rewards
        cumulative_rewards += rewards

        # Track hidden states
        for i in range(n_agents):
            hidden_state_trace[i].append(new_hidden_states[i].copy())

        # Track action counts
        for i, a in enumerate(actions):
            action_counts[i][a] += 1

        # Get global state
        global_state = get_full_state(env, flatten=True)

        # Store transitions in buffer
        buffer.store_transitions(
            global_states=np.array([global_state for _ in range(n_agents)]),
            observations=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array([terminated]*n_agents),
            hidden_states=np.array(hidden_states),
            log_probs=np.array(log_probs)
        )

        ep_return.append(rewards)
        obs = new_obs
        hidden_states = new_hidden_states
        step_counter += 1

        # Debug prints per step
        print(f"[Step {step_counter}]")
        print(f"Actions: {actions}")
        print(f"Shaped Rewards: {rewards}")
        print(f"Dones: {terminated}")
        print(f"Hidden states sample (agent 0): {hidden_states[0]}")
        print(f"Buffer index after store: {buffer.current_index}")
        print(f"Last end_episode_index: {buffer.end_episode_indices[-1] if buffer.end_episode_indices else 'None'}\n")

    return np.sum(ep_return), step_counter, terminated, cumulative_rewards, hidden_state_trace, action_counts

# ------------------------------
# Run multiple episodes
# ------------------------------
def collect_episodes(env, agent, num_episodes, buffer: Buffer, mode='train'):
    returns = []

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep+1}/{num_episodes} ===")
        ep_return, steps, terminated, cumulative_rewards, hidden_state_trace, action_counts = run_episode(env, agent, buffer, mode)
        returns.append(ep_return)

        print(f"Episode return: {ep_return}, steps: {steps}, terminated: {terminated}")
        print(f"Cumulative rewards per agent: {cumulative_rewards}")
        for i, counter in enumerate(action_counts):
            print(f"Agent {i} action counts: {dict(counter)}")

    return returns

# ------------------------------
# Debugging agent learn
# ------------------------------
def debug_learn(agent, buffer: Buffer):
    print("\n--- Debugging learn() ---")
    actor_losses, critic_loss = agent.learn(buffer)
    print("Actor losses:", actor_losses)
    print("Critic loss:", critic_loss)

# ------------------------------
# Main debug loop
# ------------------------------
def run_debug_environment():
    env = make_env()
    single_env = env

    state_dim = get_full_state(single_env, flatten=True).shape[0]
    observation_dim = env.observation_space[0].shape[0]
    n_agents = len(env.observation_space)

    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=single_env, batch_size=32)

    # Create buffer
    buffer = Buffer(
        size=BUFFER_SIZE,
        n_agents=n_agents,
        global_state_dim=(state_dim,),
        observation_dim=observation_dim,
        hidden_state_dim=HIDDEN_STATE_DIM
    )
    buffer.print_attributes()

    # Collect episodes
    returns = collect_episodes(env, agent, N_COLLECTION_EPISODES, buffer, mode='train')

    print("\n--- Buffer Inspection ---")
    print("Current index:", buffer.current_index)
    print("End episode indices:", buffer.end_episode_indices)
    print("Sample stored global state at index 0:", buffer.global_states[0])
    print("Sample stored observation at index 0:", buffer.observations[0])
    print("Sample stored action at index 0:", buffer.actions[0])
    print("Sample stored reward at index 0:", buffer.rewards[0])
    print("Sample stored done at index 0:", buffer.dones[0])
    print("Sample stored hidden state at index 0:", buffer.hidden_states[0])
    print("Sample stored log_prob at index 0:", buffer.old_log_probs[0])

    # Debug learn
    debug_learn(agent, buffer)

if __name__ == "__main__":
    run_debug_environment()
