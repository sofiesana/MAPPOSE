import gymnasium as gym
import numpy as np
from buffer import Buffer
from util import get_full_state
from agents.agent_factory import AgentFactory
from testing_out import run_episode


def test_buffer_new_functions():
    """Run a short episode, fill the buffer, and verify that it works correctly."""
    env = gym.make("rware-tiny-2ag-v2")

    n_agents = len(env.observation_space)
    global_state_dim = get_full_state(env, flatten=True).shape
    observation_dim = env.observation_space[0].shape[0]
    hidden_state_dim = 128

    # Initialize buffer and agent
    buffer = Buffer(100, n_agents, global_state_dim, observation_dim, hidden_state_dim)
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=64)

    print("\n🧪 Collecting data for buffer test...")
    run_episode(env, agent, "test", buffer)

    # -------------------------------
    # Buffer attribute inspection
    # -------------------------------
    print("\n✅ BUFFER INSPECTION:")
    buffer.print_attributes()
    print("Buffer filled:", buffer.buffer_filled)
    print("End of episode indices:", buffer.new_episode_indices)
    print("Sample rewards (first 5 timesteps):\n", buffer.rewards[:5])
    print("Sample done flags (first 5 timesteps):\n", buffer.dones[:5])

    # -------------------------------
    # Test: get_rewards_to_go
    # -------------------------------
    print("\n🔹 Testing get_rewards_to_go()...")
    try:
        if buffer.current_index >= 5:
            rewards_to_go = buffer.get_rewards_to_go(window_size=5, start_idxs=[0])
            print("Rewards-to-go shape:", rewards_to_go.shape)
            print("Rewards-to-go sample:", rewards_to_go[0])
        else:
            print("⚠️ Not enough timesteps collected to test rewards-to-go.")
    except Exception as e:
        print("❌ Error in get_rewards_to_go:", e)

    # -------------------------------
    # Test: get_all_states_and_summed_rewards
    # -------------------------------
    print("\n🔹 Testing get_all_states_and_summed_rewards()...")
    try:
        states, rewards = buffer.get_all_states_and_summed_rewards()
        print("States shape:", np.shape(states))
        print("Rewards shape:", np.shape(rewards))
        print("Sum of rewards per episode:", np.sum(rewards, axis=1))
    except Exception as e:
        print("❌ Error in get_all_states_and_summed_rewards:", e)

    # -------------------------------
    # Test: sample_agent_batch
    # -------------------------------
    print("\n🔹 Testing sample_agent_batch()...")
    try:
        batch = buffer.sample_agent_batch(agent_index=0, batch_size=2, window_size=5)
        if batch is not None:
            g_states, obs, acts, rews, dones, h_states, logps, start_idxs = batch
            print("Batch start indices:", start_idxs)
            print("Batch rewards shape:", rews.shape)
            print("First batch reward sequence:", rews[0])
            print("First batch done flags:", dones[0])
        else:
            print("⚠️ Not enough data in buffer yet to sample a batch.")
    except Exception as e:
        print("❌ Error in sample_agent_batch:", e)

    env.close()
    print("\n✅ Buffer testing complete.\n")


if __name__ == "__main__":
    test_buffer_new_functions()
