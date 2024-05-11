import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from monte_carlo import run_monte_carlo
from sarsa import run_sarsa
from q_learning import run_q_learning
from volcano_crossing import VolcanoCrossing
import numpy as np


class RLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reinforcement Learning App")

        self.slip_probability_var = tk.DoubleVar()
        self.epsilon_var = tk.DoubleVar()
        self.num_episodes_var = tk.IntVar()

        ttk.Label(self.root, text="Slip Probability:").grid(row=0, column=0)
        ttk.Scale(self.root, from_=0.0, to=0.3, variable=self.slip_probability_var,
                  orient="horizontal").grid(row=0, column=1)

        ttk.Label(self.root, text="Epsilon:").grid(row=1, column=0)
        ttk.Scale(self.root, from_=0.1, to=1.0, variable=self.epsilon_var,
                  orient="horizontal").grid(row=1, column=1)

        ttk.Label(self.root, text="Number of Episodes:").grid(row=2, column=0)
        ttk.Entry(self.root, textvariable=self.num_episodes_var).grid(
            row=2, column=1)

        ttk.Button(self.root, text="Run Monte Carlo",
                   command=self.run_monte_carlo).grid(row=3, column=0)
        ttk.Button(self.root, text="Run SARSA",
                   command=self.run_sarsa).grid(row=3, column=1)
        ttk.Button(self.root, text="Run Q-Learning",
                   command=self.run_q_learning).grid(row=3, column=2)

        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Canvas to display the Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=3)

    def run_monte_carlo(self):
        slip_probability = self.slip_probability_var.get()
        epsilon = self.epsilon_var.get()
        num_episodes = self.num_episodes_var.get()

        env = VolcanoCrossing(slip_probability=slip_probability)
        Q_values, avg_utility = run_monte_carlo(
            env, epsilon=epsilon, num_episodes=num_episodes)

        self.display_results(Q_values)

        print("Monte Carlo Results:")
        print("Q-values:\n", Q_values)
        print("Average Utility:", avg_utility)

    def run_sarsa(self):
        slip_probability = self.slip_probability_var.get()
        epsilon = self.epsilon_var.get()
        num_episodes = self.num_episodes_var.get()

        env = VolcanoCrossing(slip_probability=slip_probability)
        Q_values, avg_utility = run_sarsa(
            env, epsilon=epsilon, num_episodes=num_episodes)

        self.display_results(Q_values)

        print("SARSA Results:")
        print("Q-values:\n", Q_values)
        print("Average Utility:", avg_utility)

    def run_q_learning(self):
        slip_probability = self.slip_probability_var.get()
        epsilon = self.epsilon_var.get()
        num_episodes = self.num_episodes_var.get()

        env = VolcanoCrossing(slip_probability=slip_probability)
        Q_values, avg_utility = run_q_learning(
            env, epsilon=epsilon, num_episodes=num_episodes)

        self.display_results(Q_values)

        print("Q-Learning Results:")
        print("Q-values:\n", Q_values)
        print("Average Utility:", avg_utility)

    def display_results(self, Q_values):
        # Clear the previous plot
        self.ax.clear()

        # Plot Q-values as a heatmap
        im = self.ax.imshow(Q_values.max(axis=2),
                            cmap='viridis', origin='lower')

        # Add colorbar
        self.fig.colorbar(im)

        # Update canvas
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RLApp(root)
    root.mainloop()
