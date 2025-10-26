import gymnasium as gym
import numpy as np
from collections import Counter

from buffer import Buffer
from util import get_full_state, get_true_coords
from agents.agent_factory import AgentFactory
from rware.warehouse import ImageLayer

# --- SETTINGS ---
N_COLLECTION_EPISODES = 50
SEQ_SIZE = 16
HIDDEN_STATE_DIM = 128
BUFFER_SIZE = 5000

# ------------------------------
# Environment Setup
# ------------------------------
def make_env():
    return gym.make("rware-tiny-2ag-v2")

# ------------------------------
# Print environment debug info
# ------------------------------
def print_env_debug(env):
    s = get_full_state(env, flatten=False)
    shelves_layer = s[0]  # SHELVES with requested shelves = 2
    agent_dir_layer = s[1]  # AGENT_DIRECTION

    # Detect agents carrying requested shelves: overlap of agent and requested shelf
    pickups = []
    for y, x in zip(*np.where((shelves_layer == 2) & (agent_dir_layer > 0))):
        pickups.append((y, x))

    print("\n--- Environment Debug ---")
    print("Shelves + Requested Shelves (1=shelf, 2=requested):\n", shelves_layer)
    print("\nAgent Directions (0=none, 1-4=dir):\n", agent_dir_layer)
    print(f"\nPickups detected at: {pickups}")

# ------------------------------
# Reward shaping
# ------------------------------
def shape_rewards(env, rewards, obs, actions, info):
    """
    +0.5 for picking up a requested shelf (detected as agent overlapping a requested shelf)
    """
    n_agents = len(rewards)
    shaped_rewards = np.array(rewards, dtype=np.float32)

    s = get_full_state(env, flatten=False)
    shelves_layer = s[0]
    agent_dir_layer = s[1]

    for idx, (y, x) in enumerate([(ag.y, ag.x) for ag in env.unwrapped.agents]):
        if shelves_layer[y, x] == 2 and agent_dir_layer[y, x] > 0:
            shaped_rewards[idx] += 0.5

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

    cumulative_rewards = np.zeros(n_agents)
    hidden_state_trace = [[] for _ in range(n_agents)]
    action_counts = [Counter() for _ in range(n_agents)]

    total_pickups = 0
    total_deliveries = 0

    while not (terminated or truncated):
        actions, log_probs, new_hidden_states = agent.choose_action(obs, hidden_states)
        new_obs, rewards, terminated, truncated, info = env.step(actions)

        rewards = shape_rewards(env, rewards, obs, actions, info)
        cumulative_rewards += rewards

        for i in range(n_agents):
            hidden_state_trace[i].append(new_hidden_states[i].copy())
            action_counts[i][actions[i]] += 1

        global_state = get_full_state(env, flatten=True)

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

        # Debug prints
        print(f"\n=== Step {step_counter} ===")
        print(f"Actions: {actions}")
        print(f"Shaped Rewards: {rewards}")
        print(f"Dones: {terminated}")
        print(f"Hidden state sample (agent 0): {hidden_states[0]}")
        print(f"Buffer index after store: {buffer.current_index}")
        print(f"Step info dict: {info}")

        # Environment debug
        s = get_full_state(env, flatten=False)
        shelves_layer = s[0]
        agent_dir_layer = s[1]

        for idx, (y, x) in enumerate([(ag.y, ag.x) for ag in env.unwrapped.agents]):
            notice = ""
            # Pickup
            if shelves_layer[y, x] == 2 and agent_dir_layer[y, x] > 0:
                notice += f"Agent {idx} picked up requested shelf (+0.5) "
                total_pickups += 1
            # Delivery
            if 'delivered_shelf' in info and info['delivered_shelf'][idx]:
                notice += f"Agent {idx} delivered a shelf (+1.0)"
                total_deliveries += 1
            if notice:
                print(notice)

    print(f"\n=== Episode Summary ===")
    print(f"Total pickups: {total_pickups}")
    print(f"Total deliveries: {total_deliveries}")

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
    state_dim = get_full_state(env, flatten=True).shape[0]
    observation_dim = env.observation_space[0].shape[0]
    n_agents = len(env.observation_space)

    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=32)

    buffer = Buffer(
        size=BUFFER_SIZE,
        n_agents=n_agents,
        global_state_dim=(state_dim,),
        observation_dim=observation_dim,
        hidden_state_dim=HIDDEN_STATE_DIM
    )
    buffer.print_attributes()

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

    debug_learn(agent, buffer)

if __name__ == "__main__":
    run_debug_environment()
