import gymnasium as gym
import rware
import numpy as np

N_TRAIN_EPISODES = 3
N_TEST_EPISODES = 3
ITERS = 5

def run_episode(env, agent, mode):
    """Run a single episode and return the episode return"""
    observation, _ = env.reset()
    episode_ended = False
    ep_return = []
    step_counter = 0
    
    while not episode_ended:
        env.render()
        step_counter += 1
        # action = agent.choose_action(observation)
        action = env.action_space.sample()  # Random action for placeholder
        new_observation, reward, terminated, truncated, _ = env.step(action)
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
        print(f"Episode {ep} | return: {ep_return} | terminated: {bool(terminated)}")

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