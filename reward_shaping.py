import gymnasium as gym
import numpy as np
from collections import Counter

from buffer import Buffer
from util import get_full_state, get_true_coords
from agents.agent_factory import AgentFactory

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
def shape_rewards(env, rewards, agent_positions):
    """
    +0.5 for picking up a requested shelf (detected as agent overlapping a requested shelf)
    """

    global_state = get_full_state(env, flatten=False)

    shelves = global_state[0]
    agents = global_state[1]

    shaped_rewards = np.array(rewards, dtype=np.float32)

    for idx, (y, x) in enumerate(agent_positions):
        # check that this agent is indeed on the grid and facing some direction
        if agents[y, x] > 0 and shelves[y, x] == 2:
            shaped_rewards[idx] += 0.03
            # testing if it works
            # print("\n🎉🎉🎉 CELEBRATION! 🎉🎉🎉")
            # print(f"Agent {idx} picked up a requested shelf at position (y={y}, x={x})")
            # print(f"Reward before shaping: {rewards[idx]}")
            # print(f"Reward after shaping: {shaped_rewards[idx]}")

            # # Print full layers
            # print("Full shelves layer:")
            # print(shelves)
            # print("Full agent direction layer:")
            # print(agents)

    return shaped_rewards