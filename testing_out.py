import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import rware
import numpy as np
from util import get_full_state
from buffer import Buffer
import time

from agents.agent_factory import AgentFactory
from plotting import LiveLossPlotter
import os

N_COLLECTION_EPISODES = 10
N_TRAIN_EPOCHS_PER_COLLECTION = 3
ITERS = 1000

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


def run_episode(env, agent, mode, buffer: Buffer):
    """Run a single episode and return the episode return"""

    n_agents = len(env.observation_space)
    
    observation, _ = env.reset()
    episode_ended = False
    ep_return = []
    step_counter = 0

    hidden_states = np.zeros((n_agents, buffer.hidden_state_dim))  # example initial hidden state 
    while not episode_ended:
        # env.render()

        action, log_probs, new_hidden_states = agent.choose_action(observation, hidden_states)
        # action, log_probs, new_hidden_states = agent.choose_random_action()
        # action = env.action_space.sample()  # Random action for placeholder

        new_observation, reward, terminated, truncated, info = env.step(action)
        global_state = get_full_state(env, flatten=True)
        
        buffer.store_transitions(
            global_states=global_state,
            observations=observation,
            actions=action,
            rewards=reward,
            dones=terminated,
            hidden_states=hidden_states,
            log_probs=log_probs
        )
        ep_return.append(reward)

            
        if terminated or truncated:
            episode_ended = True
            
        observation = new_observation
        hidden_states = new_hidden_states
        step_counter += 1

    # if mode == 'train':
    #     print("Training step...")
        # agent.store_transition(observation, action, reward, new_observation, terminated)
        
    return ep_return, step_counter, terminated

def run_episodes(env, agent, num_episodes, plotter, mode='train'):
    """Run multiple episodes and store the returns. If testing, no learning occurs."""

    returns = []

    global_state_dim = get_full_state(env, flatten=True).shape
    n_agents = len(env.observation_space)
    observation_dim = env.observation_space[0].shape[0]
    hidden_state_dim = 128  # example hidden state dimension for RNN
    buffer = Buffer(size=100000, n_agents=n_agents, global_state_dim=global_state_dim,
                    observation_dim=observation_dim, hidden_state_dim=hidden_state_dim)
    # buffer.print_attributes()

    if mode == 'test':
        agent.set_test_mode()

    pre_collect_time = time.time()

    for ep in range(num_episodes):
        print("Running episode ", ep + 1, "/", num_episodes)
        ep_return, _, terminated = run_episode(env, agent, mode, buffer)
        returns.append(ep_return)
        reward_sum = np.sum(ep_return)
        print(f"Episode {ep} | mean return: {reward_sum} | terminated: {bool(terminated)}")

    print("Time to collect episodes:", round(time.time() - pre_collect_time, 4), "seconds")

    if mode == 'train':
        all_actor_loss_list = []
        all_critic_loss = []
        for epoch in range(N_TRAIN_EPOCHS_PER_COLLECTION):
            print(f"Training epoch {epoch + 1}/{N_TRAIN_EPOCHS_PER_COLLECTION}")
            actor_loss_list, critic_loss = agent.learn(buffer)
            all_actor_loss_list.extend(actor_loss_list)
            all_critic_loss.append(critic_loss)

            print("Average actor loss this epoch:", np.mean(actor_loss_list), "Average critic loss this epoch:", critic_loss)

            # plotter.update(np.mean(actor_loss_list))
            # plotter.save("results/actor_loss_plot_{epoch}.png")

        agent.update_prev_actor_models() # Update prev network to current before optimizing current
    elif mode == 'test':
        print(f"Average test return: {np.mean(agent.test_returns)}")


    return returns, all_actor_loss_list, all_critic_loss

def make_env():
    return gym.make("rware-tiny-2ag-v2")

def run_environment(args):
    """Main function to set up and run the environment with the specified agent"""
    # set up looping through iters
    agent_factory = AgentFactory()
    # plotter = LiveLossPlotter()
    env = gym.make("rware-tiny-2ag-v2")
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=8)
    os.mkdir("results") if not os.path.exists("results") else None

    mean_returns = np.zeros(ITERS)
    mean_actor_losses = np.zeros(ITERS)
    mean_critic_losses = np.zeros(ITERS)

    for iteration in range(ITERS):
        start_time = time.time()
        env.reset()
        print(f"Iteration {iteration + 1}/{ITERS}")
    
        returns, actor_loss_list, critic_loss = run_episodes(env, agent, N_COLLECTION_EPISODES, None, mode='train')
        # plotter.update(np.mean(actor_loss_list))

        mean_returns[iteration] = np.mean(returns)
        mean_actor_losses[iteration] = np.mean(actor_loss_list)
        mean_critic_losses[iteration] = np.mean(critic_loss)

        mean_returns[iteration] = np.mean(returns)
        mean_actor_losses[iteration] = np.mean(actor_loss_list)
        mean_critic_losses[iteration] = np.mean(critic_loss)

        np.save(f"results/returns_iteration_{iteration}.npy", returns)
        np.save(f"results/actor_loss_iteration_{iteration}.npy", actor_loss_list)
        np.save(f"results/critic_loss_iteration_{iteration}.npy", critic_loss)

        np.save("results/_mean_returns.npy", mean_returns)
        np.save("results/_mean_actor_losses.npy", mean_actor_losses)
        np.save("results/_mean_critic_losses.npy", mean_critic_losses)
        print("Training time for iteration", iteration + 1, ":", time.time() - start_time, "seconds")
    
    env.close()


if __name__ == "__main__":
    run_environment(None)