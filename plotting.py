import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class LiveLossPlotter:
    def __init__(self):
        self.losses = []
        self.steps = []
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        plt.ion()  # interactive mode on

    def update(self, loss):
        # Append the new loss
        self.losses.append(loss)

        # Clear previous plot
        clear_output(wait=True)
        self.ax.clear()

        # Use the index of the list as the x-axis
        self.ax.plot(range(len(self.losses)), self.losses, label="Actor Loss", color='tab:blue')
        self.ax.set_xlabel("Updates")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Live Actor Loss")
        self.ax.legend()
        self.ax.grid(True)

        # Show updated figure
        display(self.fig)
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()
