import numpy as np
import time
import os
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt 


from buffer import Buffer
from agents.agent_factory import AgentFactory
from plotting import LiveLossPlotter
from util import get_full_state  # If not compatible, can be stubbed for now


N_COLLECTION_EPISODES = 10
N_TRAIN_EPOCHS_PER_COLLECTION = 3
ITERS = 1000


def inspect_environment(env):
    print("Observation space:", env.observation_space(env.agents[0]))
    print("Action space:", env.action_space(env.agents[0]))

    obs, _ = env.reset()
    print("\nExample observation (type, shape):")
    for agent in env.agents:
        print(f"Agent {agent} obs shape: {np.array(obs[agent]).shape} | type: {type(obs[agent])}")
        print(f"Agent {agent} obs sample:\n{np.array(obs[agent])}\n")


def run_episode(env, agent, mode, buffer: Buffer):
    """Run a single episode in MPE and store transitions"""

    env.reset()
    n_agents = len(env.agents)
    episode_ended = False
    ep_return = []
    step_counter = 0

    # Collect initial observations
    observations, _ = env.reset()
    hidden_states = np.zeros((n_agents, buffer.hidden_state_dim))

    while env.agents:  # PettingZoo removes agents as they terminate
        env.render()
        # Convert observations dict → list (sorted by agent order)
        obs_list = [observations[a] for a in env.agents]

        actions, log_probs, new_hidden_states = agent.choose_action(obs_list, hidden_states)

        # Convert actions list → dict
        actions_dict = {agent_name: act for agent_name, act in zip(env.agents, actions)}

        # Step environment
        new_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

        # Convert dicts to arrays/lists
        reward_list = [rewards[a] for a in env.agents]
        done_list = [terminations[a] or truncations[a] for a in env.agents]

        # print("Rewards:", reward_list)
        # print("Dones:", done_list)

        # Use a placeholder for global state if get_full_state is not MPE-compatible
        global_state = np.concatenate([obs for obs in obs_list])  # simple approximation

        done = False
        if len(done_list) == 0:
            done = True
            reward_list = [0.0 for _ in range(n_agents)]

        # Check if reward list is empty
        # if len(reward_list) != 0:
        buffer.store_transitions(
            global_states=global_state,
            observations=obs_list,
            actions=actions,
            rewards=reward_list,
            dones=done,
            hidden_states=hidden_states,
            log_probs=log_probs,
        )

        ep_return.append(np.mean(reward_list))

        # Update loop vars
        observations = new_observations
        hidden_states = new_hidden_states
        step_counter += 1

        # End episode if all agents done
        if all(done_list):
            episode_ended = True
            break

    return ep_return, step_counter, episode_ended


def run_episodes(env, agent, num_episodes, plotter, mode='train'):
    """Run multiple episodes in the MPE environment."""
    returns = []

    first_obs, _ = env.reset()
    n_agents = len(env.possible_agents)
    observation_dim = first_obs[env.agents[0]].shape[0]
    global_state_dim = (n_agents * observation_dim,)
    hidden_state_dim = 128

    buffer = Buffer(
        size=100000,
        n_agents=n_agents,
        global_state_dim=global_state_dim,
        observation_dim=observation_dim,
        hidden_state_dim=hidden_state_dim,
    )

    if mode == 'test':
        agent.set_test_mode()

    for ep in range(num_episodes):
        # print(f"Running episode {ep + 1}/{num_episodes}")
        ep_return, _, terminated = run_episode(env, agent, mode, buffer)
        returns.append(np.sum(ep_return))
        # print(f"Episode {ep} | mean return: {np.sum(ep_return)} | terminated: {bool(terminated)}")

    if mode == 'train':
        all_actor_loss_list = []
        all_critic_loss = []
        for epoch in range(N_TRAIN_EPOCHS_PER_COLLECTION):
            print(f"Training epoch {epoch + 1}/{N_TRAIN_EPOCHS_PER_COLLECTION}")
            actor_loss_list, critic_loss = agent.learn(buffer)
            all_actor_loss_list.extend(actor_loss_list)
            all_critic_loss.append(critic_loss)
            print(f"Average actor loss: {np.mean(actor_loss_list)} | Critic loss: {critic_loss}")

        agent.update_prev_actor_models()

    return returns, all_actor_loss_list, all_critic_loss


def make_env():
    # Example: 3-agent simple spread
    return simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)


def run_environment(args=None):
    """Main loop for MPE test"""
    env = make_env()
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=512)
    os.makedirs("results", exist_ok=True)

    mean_returns = np.zeros(ITERS)
    mean_actor_losses = np.zeros(ITERS)
    mean_critic_losses = np.zeros(ITERS)

    # plt.ion()
    # fig, axs = plt.subplots(1, 3, figsize=(10, 4))  # 3 subplots stacked vertically

    # # Lines for live updating
    # line_return, = axs[0].plot([], [], label='Mean Return', color='b')
    # line_actor, = axs[1].plot([], [], label='Mean Actor Loss', color='r')
    # line_critic, = axs[2].plot([], [], label='Mean Critic Loss', color='g')

    # Labels and titles
    # axs[0].set_ylabel('Mean Return')
    # axs[0].set_title('Training Progress (Avg Return per Iteration)')
    # axs[0].legend()

    # axs[1].set_ylabel('Mean Actor Loss')
    # axs[1].set_title('Actor Loss over Iterations')
    # axs[1].legend()

    # axs[2].set_xlabel('Iteration')
    # axs[2].set_ylabel('Mean Critic Loss')
    # axs[2].set_title('Critic Loss over Iterations')
    # axs[2].legend()

    # plt.tight_layout()
    # plt.show(block=False)

    for iteration in range(ITERS):
        print(f"Iteration {iteration + 1}/{ITERS}")
        returns, actor_loss_list, critic_loss = run_episodes(env, agent, N_COLLECTION_EPISODES, None, mode='train')

        mean_returns[iteration] = np.mean(returns)
        mean_actor_losses[iteration] = np.mean(actor_loss_list)
        mean_critic_losses[iteration] = np.mean(critic_loss)

        print("     Mean Return:", mean_returns[iteration])

        # np.save(f"results/returns_iteration_{iteration}.npy", returns)
        # np.save(f"results/actor_loss_iteration_{iteration}.npy", actor_loss_list)
        # np.save(f"results/critic_loss_iteration_{iteration}.npy", critic_loss)
        # np.save("results/_mean_returns.npy", mean_returns)
        # np.save("results/_mean_actor_losses.npy", mean_actor_losses)
        # np.save("results/_mean_critic_losses.npy", mean_critic_losses)

        # Update lines
        # iterations = np.arange(iteration + 1)
        # line_return.set_xdata(iterations)
        # line_return.set_ydata(mean_returns[:iteration + 1])

        # line_actor.set_xdata(iterations)
        # line_actor.set_ydata(mean_actor_losses[:iteration + 1])

        # line_critic.set_xdata(iterations)
        # line_critic.set_ydata(mean_critic_losses[:iteration + 1])

        # # Rescale axes
        # for ax in axs:
        #     ax.relim()
        #     ax.autoscale_view()

        # plt.pause(0.1)

    # plt.ioff()
    # plt.show()
    env.close()


if __name__ == "__main__":
    run_environment()
