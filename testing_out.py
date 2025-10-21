import gymnasium as gym
import rware
import numpy as np
from util import get_full_state
from buffer import Buffer

from agents.agent_factory import AgentFactory
from plotting import LiveLossPlotter

N_TRAIN_EPISODES = 8
N_TEST_EPISODES = 3
N_TRAIN_EPOCHS = 5
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
        # pause until key press
        # input("Press Enter to continue...")

        action, log_probs, hidden_states = agent.choose_action(observation, hidden_states)
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
        # buffer.print_attributes()
        # buffer.print_buffer()
        # batch = buffer.sample_agent_batch(agent_index=0, batch_size=10, window_size=5)

        # you can also store single agent transition if needed
        # dummy_single_agent_transition = {
        #     "global_state": global_state,
        #     "observation": observation[0],
        #     "action": action[0],
        #     "reward": reward[0],
        #     "next_observation": new_observation[0],
        #     "done": terminated,
        #     "hidden_state": dummy_hidden_state[0]
        # }

        # buffer.store_single_agent_transition(
        #     agent_index=0,
        #     global_state=dummy_single_agent_transition["global_state"],
        #     observation=dummy_single_agent_transition["observation"],
        #     action=dummy_single_agent_transition["action"],
        #     reward=dummy_single_agent_transition["reward"],
        #     next_observation=dummy_single_agent_transition["next_observation"],
        #     done=dummy_single_agent_transition["done"],
        #     hidden_state=dummy_single_agent_transition["hidden_state"]
        # )

        # to check if batching is working:
        # print("actions of Sampled batch for agent 0:", batch[5] if batch is not None else "No batch sampled")
        # print("Sampled batch for agent 0:", batch if batch is not None else "No batch sampled")

        ep_return.append(reward)

            
        if terminated or truncated:
            episode_ended = True
            
        observation = new_observation
        step_counter += 1

    if mode == 'train':
        print("Training step...")
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

    for ep in range(num_episodes):
        print("Running episode ", ep + 1, "/", num_episodes)
        ep_return, _, terminated = run_episode(env, agent, mode, buffer)
        returns.append(ep_return)
        reward_sum = np.sum(ep_return)
        print(f"Episode {ep} | mean return: {reward_sum} | terminated: {bool(terminated)}")

    if mode == 'train':
        # agent.save_rewards()
        # agent.save_model()
        for epoch in range(N_TRAIN_EPOCHS):
            print(f"Training epoch {epoch + 1}/{N_TRAIN_EPOCHS}")
            actor_loss_list, critic_loss = agent.learn(buffer)
            # print("Actor 1 loss:", actor_loss_list[0], " ---  Actor 2 loss:", actor_loss_list[1], " ---  Critic loss:", critic_loss)
            plotter.update(np.mean(actor_loss_list))

        agent.update_prev_actor_models() # Update prev network to current before optimizing current
    elif mode == 'test':
        print(f"Average test return: {np.mean(agent.test_returns)}")
        # agent.save_test_returns()

    return returns, actor_loss_list, critic_loss


def run_environment(args):
    """Main function to set up and run the environment with the specified agent"""
    # set up looping through iters
    agent_factory = AgentFactory()
    plotter = LiveLossPlotter()
    for iteration in range(ITERS):
        print(f"Iteration {iteration + 1}/{ITERS}")
        # add iteration to args
        env = gym.make("rware-tiny-2ag-v2")
        agent = agent_factory.create_agent(agent_type="MAPPOSE", env=env, batch_size=16)
        # agent = 0
        # make_data_folder(agent.path)
        
        # Training phase
        # print(f"Training {agent.agentName} agent...")
        returns, actor_loss_list, critic_loss = run_episodes(env, agent, N_TRAIN_EPISODES, plotter, mode='train')
        
        # Testing phase
        # print(f"Testing {agent.agentName} agent...")
        # run_episodes(env, agent, N_TEST_EPISODES, mode='test')
    env.close()


if __name__ == "__main__":
    run_environment(None)
    # inspect_environment(gym.make("rware-tiny-2ag-v2"))