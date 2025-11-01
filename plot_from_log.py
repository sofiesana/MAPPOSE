import re
import matplotlib.pyplot as plt
import numpy as np

# Path to your log file
log_path = "output_random.log"

# --- Regex patterns ---
iteration_re = re.compile(r"Iteration (\d+)/\d+")
mean_return_re = re.compile(r"mean return: ([0-9.]+)")
true_mean_return_re = re.compile(r"true mean return: ([0-9.]+)")
loss_re = re.compile(r"Mean Actor Loss:\s*(-?[0-9.]+(?:e-?\d+)?)\s*Mean Critic Loss:\s*([0-9.]+(?:e-?\d+)?)")

def clean_log_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        content = infile.read()

    # Regex to match everything from "Final global state shape:" through the array block
    cleaned_content = re.sub(
        r"Final global state shape:.*?Final global state array:\s*\[\[\[.*?\]\]\]",
        "",
        content,
        flags=re.DOTALL
    )

    # Remove possible leftover blank lines
    cleaned_content = re.sub(r'\n\s*\n+', '\n', cleaned_content).strip()

    with open(output_path, 'w') as outfile:
        outfile.write(cleaned_content + '\n')

def extract_log_data(log_path, get_returns=True, get_losses=True):
    # --- Storage ---
    iterations = []
    mean_returns = []
    actor_losses = []
    critic_losses = []

    current_iteration = None
    current_returns = []

    with open(log_path, "r") as f:
        for line in f:
            # Detect iteration start
            iter_match = iteration_re.search(line)
            if iter_match:
                # Save previous iteration data
                if current_iteration is not None and current_returns:
                    iterations.append(current_iteration)
                    mean_returns.append(sum(current_returns) / len(current_returns))
                current_iteration = int(iter_match.group(1))
                current_returns = []
                continue

            # Extract per-episode mean returns
            if get_returns:
                return_match = mean_return_re.search(line)
                if return_match:
                    current_returns.append(float(return_match.group(1)))

            # Extract actor/critic losses
            if get_losses:
                loss_match = loss_re.search(line)
                if loss_match:
                    actor_losses.append(float(loss_match.group(1)))
                    critic_losses.append(float(loss_match.group(2)))

    # Handle last iteration
    if current_iteration is not None and current_returns:
        iterations.append(current_iteration)
        mean_returns.append(sum(current_returns) / len(current_returns))

    return (np.array(iterations), np.array(mean_returns), np.array(actor_losses), np.array(critic_losses))


def get_mappose_and_random(mappose_log_paths, random_log_path):
    # --- Storage ---
    iterations = []
    mappose_mean_returns = []
    random_mean_returns = []

    for log_path in mappose_log_paths:
        iters, returns, _, _ = extract_log_data(log_path, get_losses=False)
        iterations = iters  # Assuming all logs have the same iterations
        mappose_mean_returns.append(returns)

    iters, returns, _, _ = extract_log_data(random_log_path, get_losses=False)
    random_mean_returns = returns

    return (np.array(iterations), np.array(mappose_mean_returns), np.array(random_mean_returns))

def plot_mappose_vs_random(iterations, mappose_returns, random_returns):
    # Plot with mean and std shading (for MAPPOSE)

    mean_mappose = np.mean(mappose_returns, axis=0)
    smooth_mean_mappose = moving_average(mean_mappose, window_size=20)
    std_mappose = np.std(mappose_returns, axis=0)
    smooth_std_mappose = moving_average(std_mappose, window_size=20)

    smooth_random_returns = moving_average(random_returns, window_size=20)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, smooth_mean_mappose, label='MAPPOSE Mean Return', color='blue')
    plt.fill_between(iterations, smooth_mean_mappose - smooth_std_mappose, smooth_mean_mappose + smooth_std_mappose, color='blue', alpha=0.2, label='MAPPOSE Std Dev')
    plt.plot(iterations, smooth_random_returns, label='Random Mean Return', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Return')
    plt.title('MAPPOSE vs Random Agent Mean Returns (With 20-Iteration Moving Average)')
    plt.legend()
    plt.grid()
    plt.savefig("final_plots/mappose_vs_random_mean_return.png")
    plt.show()

def moving_average(data, window_size):
    # Simple moving average
    # If idxex < window_size, average over available data
    averaged = []
    for i in range(len(data)):
        if i < window_size:
            averaged.append(np.mean(data[:i+1]))
        else:
            averaged.append(np.mean(data[i-window_size+1:i+1]))
    return np.array(averaged)

def plot_reward_shaping_effect(np_file_path):
    rewards = np.load(np_file_path)
    smooth_rewards = moving_average(rewards, window_size=20)


    plt.figure(figsize=(10, 6))
    plt.plot(smooth_rewards, label='Reward Shaping Values', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Mean Returns (with Reward Shaping)')
    plt.title('Mean Returns with Reward Shaping Over Iterations (With 20-Iteration Moving Average)')
    plt.grid()
    plt.savefig("final_plots/reward_shaping_values.png")
    plt.show()

# Get data
folder_path = "results_log_files/"
mappose_log_paths = [f"{folder_path}mappose final {i}.log" for i in range(1, 3)]
random_log_path = folder_path + "random final.log"
iterations, mappose_returns, random_returns = get_mappose_and_random(mappose_log_paths, random_log_path)

# Plot comparison
plot_mappose_vs_random(iterations, mappose_returns, random_returns)

# Plot reward shaping values
reward_shaping_np_file = folder_path + "reward_shaping_mean_returns.npy"
plot_reward_shaping_effect(reward_shaping_np_file)


