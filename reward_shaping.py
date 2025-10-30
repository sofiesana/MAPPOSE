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
