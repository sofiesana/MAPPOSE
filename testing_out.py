import gymnasium as gym
import rware
import numpy as np
from util import get_full_state
from buffer import Buffer

N_TRAIN_EPISODES = 3
N_TEST_EPISODES = 3
ITERS = 1

def run_episode(env, agent, mode):
    """Run a single episode and return the episode return"""

    global_state_dim = len(get_full_state(env))
    
    n_agents = len(env.observation_space)
    print("Number of agents:", n_agents)
    
    observation_dim = env.observation_space[0].shape[0]
    print("Observation Space:", observation_dim)

    buffer = Buffer(size=1000, n_agents=n_agents, global_state_dim=global_state_dim,
                    observation_dim=observation_dim)
    buffer.print_attributes()
    
    observation, _ = env.reset()
    episode_ended = False
    ep_return = []
    step_counter = 0
    
    while not episode_ended:
        env.render()
        step_counter += 1
        # action = agent.choose_action(observation)
        action = env.action_space.sample()  # Random action for placeholder
        new_observation, reward, terminated, truncated, info = env.step(action)
        global_state = get_full_state(env)
        
        buffer.store_transitions(
            global_states=global_state,
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=new_observation,
            dones=terminated
        )
        # buffer.print_buffer()
        batch = buffer.sample_agent_batch(agent_index=0, batch_size=10)
        # print("Sampled batch for agent 0:", batch if batch is not None else "No batch sampled")

        ep_return.append(reward)
        
        if mode == 'train':
            agent.store_transition(observation, action, reward, new_observation, terminated)
            agent.learn()
            
        if terminated or truncated:
            episode_ended = True
            
        observation = new_observation
        
    return ep_return, step_counter, terminated

def run_episodes(env, agent, num_episodes, mode='train'):
    """Run multiple episodes and store the returns. If testing, no learning occurs."""

    returns = []

    if mode == 'test':
        agent.set_test_mode()

    for ep in range(num_episodes):
        ep_return, _, terminated = run_episode(env, agent, mode)
        returns.append(ep_return)
        if mode == 'train':
            agent.store_return(ep_return)
        elif mode == 'test':
            agent.store_test_return(ep_return)
        # print(f"Episode {ep} | return: {ep_return} | terminated: {bool(terminated)}")

    if mode == 'train':
        agent.save_rewards()
        agent.save_model()
    elif mode == 'test':
        print(f"Average test return: {np.mean(agent.test_returns)}")
        agent.save_test_returns()

    return returns


def run_environment(args):
    """Main function to set up and run the environment with the specified agent"""
    # set up looping through iters
    for iteration in range(ITERS):
        print(f"Iteration {iteration + 1}/{ITERS}")
        # add iteration to args
        env = gym.make("rware-tiny-2ag-v2")
        agent = 0
        # make_data_folder(agent.path)
        
        # Training phase
        # print(f"Training {agent.agentName} agent...")
        run_episodes(env, agent, N_TRAIN_EPISODES, mode='none')
        
        # Testing phase
        # print(f"Testing {agent.agentName} agent...")
        # run_episodes(env, agent, N_TEST_EPISODES, mode='test')


if __name__ == "__main__":
    run_environment(None)