from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_curve(values, output, title, ylabel):
    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _save(fig, output)


def plot_goal_success(goal_rates, output="goal_success_bar.png"):
    fig, ax = plt.subplots()
    names = list(goal_rates)
    ax.bar(names, [goal_rates[n] for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title("Goal Success")
    return _save(fig, output)


def plot_transition_heatmap(matrix, labels, output="transition_matrix_heatmap.png"):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Target goal")
    ax.set_ylabel("Start pose")
    ax.set_title("Transition Success Matrix")
    fig.colorbar(im, ax=ax)
    return _save(fig, output)
