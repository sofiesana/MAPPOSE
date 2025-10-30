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
def shape_rewards(env, rewards, agent_positions, holding_shelf):
    """
    +0.5 for picking up a requested shelf (detected as agent overlapping a requested shelf)
    """

    global_state = get_full_state(env, flatten=False)
    shelves = global_state[0]
    agents = global_state[1]

    shaped_rewards = np.array(rewards, dtype=np.float32)

    for idx, (y, x) in enumerate(agent_positions):
            # Check for delivery: agent holding a shelf on a goal cell
            if holding_shelf[idx] == True and shelves[y, x] == 3:
                holding_shelf[idx] = False  # reset holding after delivery
                
                # print(f"Agent {idx} delivered shelf at {(y, x)}")

            # Check for pickup: agent on a requested shelf and not already holding
            if agents[y, x] > 0 and shelves[y, x] == 2:
                if not holding_shelf[idx]:
                    shaped_rewards[idx] += 0.1  # reward for pickup
                    holding_shelf[idx] = True
                    # print(f"Agent {idx} picked up requested shelf at {(y, x)}")

    
    return shaped_rewards, holding_shelf