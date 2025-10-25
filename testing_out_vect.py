
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import rware
import numpy as np
from util import get_full_state
from new_buffer import Buffer
import time

from agents.agent_factory_vect import AgentFactory
from plotting import LiveLossPlotter
import os

N_COLLECTION_EPISODES = 1  # Number of episodes per environment
N_ENVS = 10  # Number of parallel environments
N_TRAIN_EPOCHS_PER_COLLECTION = 3
ITERS = 1000

class RwareRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # If reward is a list, sum it
        reward = float(np.sum(reward))
        return obs, reward, terminated, truncated, info

def inspect_environment(env):
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset()
    print("\nExample observation (type, shape):")
    if isinstance(obs, (list, tuple)):
        print(f"Number of agents: {len(obs)}")
        for i, ob in enumerate(obs):
            print(f"Agent {i} obs shape: {np.array(ob).shape} | type: {type(ob)}")
            print(f"Agent {i} obs sample:\n{np.array(ob)}\n")
    else:
        print(f"Obs shape: {np.array(obs).shape} | type: {type(obs)}")
        print(obs)



def run_vectorized_episodes(env, agent, num_episodes, plotter, buffer, mode='train'):
    """
    Run multiple episodes in parallel vectorized environments and store the returns.
    """
    print(f"Running {num_episodes} episodes in {mode} mode.")
    n_envs = env.num_envs
    n_agents = len(env.single_observation_space)
    returns = [[] for _ in range(n_envs)]
    steps = [0 for _ in range(n_envs)]

    # Initial hidden states: shape (n_envs, n_agents, hidden_state_dim)
    hidden_states = np.zeros((n_envs, n_agents, buffer.hidden_state_dim))
    obs, _ = env.reset()
    terminated = np.array([False] * n_envs)
    truncated = np.array([False] * n_envs)
    step_counts = np.zeros(n_envs, dtype=int)
    timestep = 0
    while not np.all(terminated | truncated):
        # print("Current timestep:", timestep)
        timestep += 1
        # print("Current terminated states:", terminated)
        # Only act for unfinished envs
        active_mask = ~(terminated | truncated)
        active_indices = np.where(active_mask)[0]
        obs_active = tuple([o[active_indices] for o in obs])
        hidden_active = hidden_states[active_indices]
        # print("Number of active envs:", len(active_indices))
        # print("hidden_active shape:", hidden_active.shape)

        # Agent acts for all active envs
        action, log_probs, new_hidden_states = agent.choose_action(obs_active, hidden_active)

        # Fill actions for all envs (inactive envs get dummy actions)
        full_action = np.zeros((n_envs, n_agents), dtype=int)
        full_log_probs = np.zeros((n_envs, n_agents))
        full_new_hidden_states = hidden_states.copy()
        for idx, i in enumerate(active_indices):
            full_action[i] = action[idx]
            full_log_probs[i] = log_probs[idx]
            full_new_hidden_states[i] = new_hidden_states[idx]

        actions_for_env = [list(full_action[i]) for i in range(n_envs)]
        # print("Actions for env len:", len(actions_for_env))
        
        new_obs, reward, term, trunc, info = env.step(actions_for_env)
        # print all episode info for each env to ensure it is working
        # for i in range(n_envs):
        #     print("Actions taken by env:", actions_for_env[i])
        #     print(f"Env {i} | Reward: {reward} | Terminated: {term[i]}")
        #     print("Observation:", new_obs[0][i], new_obs[1][i])

        # Explicitly enforce termination at 500 steps
        for i in range(n_envs):
            step_counts[i] += 1 if active_mask[i] else 0
            if step_counts[i] >= 500:
                term[i] = True

        global_state = [get_full_state(env.envs[i], flatten=True) for i in range(n_envs)]
        # print(reward.shape)

        buffer.store_transitions(
            global_states=np.array(global_state),
            observations=obs,
            actions=full_action,
            rewards=reward,
            dones=term,
            hidden_states=hidden_states,
            log_probs=full_log_probs
        )

        for i in active_indices:
            returns[i].append(np.sum(reward[i]))
            steps[i] += 1

        obs = new_obs
        hidden_states = full_new_hidden_states
        terminated = term
        truncated = trunc

    return [np.sum(r) for r in returns], steps, terminated


def make_env():
    env = gym.make("rware-tiny-2ag-v2")
    env = RwareRewardWrapper(env)
    return env

def run_episodes_vectorized(env, agent, plotter, mode='train'):
    """
    Run multiple episodes in parallel vectorized environments and store the returns.
    """
    if mode == 'test':
        agent.set_test_mode()

    global_state_dim = get_full_state(env.envs[0], flatten=True).shape
    n_agents = len(env.single_observation_space)
    observation_dim = env.single_observation_space[0].shape[0]
    hidden_state_dim = 128
    buffer = Buffer(max_episodes=N_COLLECTION_EPISODES, n_agents=n_agents, global_state_dim=global_state_dim,
                    obs_dim=observation_dim, hidden_state_dim=hidden_state_dim, n_envs=N_ENVS)

    pre_collect_time = time.time()
    returns, steps, terminated = run_vectorized_episodes(env, agent, N_COLLECTION_EPISODES, plotter, buffer, mode)
    print("Time to collect episodes:", round(time.time() - pre_collect_time, 4), "seconds")

    all_actor_loss_list = []
    all_critic_loss = []
    if mode == 'train':
        for epoch in range(N_TRAIN_EPOCHS_PER_COLLECTION):
            print(f"Training epoch {epoch + 1}/{N_TRAIN_EPOCHS_PER_COLLECTION}")
            actor_loss_list, critic_loss = agent.learn(buffer)
            all_actor_loss_list.extend(actor_loss_list)
            all_critic_loss.append(critic_loss)
        agent.update_prev_actor_models()
    elif mode == 'test':
        print(f"Average test return: {np.mean(agent.test_returns)}")

    return returns, all_actor_loss_list, all_critic_loss


def make_vector_env():
    return SyncVectorEnv([make_env for _ in range(N_ENVS)])


def run_environment(args):
    """Main function to set up and run the environment with the specified agent"""
    agent_factory = AgentFactory()
    env = make_vector_env()
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=3)
    os.mkdir("results") if not os.path.exists("results") else None

    mean_returns = np.zeros(ITERS)
    mean_actor_losses = np.zeros(ITERS)
    mean_critic_losses = np.zeros(ITERS)

    for iteration in range(ITERS):
        start_time = time.time()
        env.reset()
        print(f"Iteration {iteration + 1}/{ITERS}")

        returns, actor_loss_list, critic_loss = run_episodes_vectorized(env, agent, None, mode='train')

        mean_returns[iteration] = np.mean(returns)
        mean_actor_losses[iteration] = np.mean(actor_loss_list)
        mean_critic_losses[iteration] = np.mean(critic_loss)

        if (iteration+1) % 50 == 0:
            agent.save_all_models(f"models/agent_iteration_{iteration}")
            print(f"Saved models at iteration {iteration}")

        np.save("results/_mean_returns.npy", mean_returns)
        np.save("results/_mean_actor_losses.npy", mean_actor_losses)
        np.save("results/_mean_critic_losses.npy", mean_critic_losses)
        print("Training time for iteration", iteration + 1, ":", time.time() - start_time, "seconds")

    env.close()


if __name__ == "__main__":
    run_environment(None)