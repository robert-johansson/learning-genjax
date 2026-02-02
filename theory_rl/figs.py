"""Matplotlib visualization for theory-based RL model."""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .simulations import (
    sim_acquisition,
    sim_discrimination,
    sim_contingency_reversal,
    sim_pree,
    sim_contingency_degradation,
    sim_posterior_dynamics,
    _smooth,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")

THEORY_COLOR = "#2171B5"
QLEARN_COLOR = "#CB181D"


def _setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["figure.dpi"] = 150


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_with_sem(ax, trials, mean, sem, color, label, window=5):
    """Plot smoothed mean with SEM shading."""
    mean_s = _smooth(np.array(mean), window)
    sem_s = _smooth(np.array(sem), window)
    trials = np.array(trials)
    ax.plot(trials, mean_s, color=color, label=label)
    ax.fill_between(trials, mean_s - sem_s, mean_s + sem_s,
                    color=color, alpha=0.15)


def plot_fig1():
    """Figure 1: Basic acquisition — reward rate over trials."""
    _setup_style()
    results = sim_acquisition()
    T = results["n_trials"]
    trials = np.arange(T)

    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_with_sem(ax, trials, results["theory"]["reward_mean"],
                   results["theory"]["reward_sem"], THEORY_COLOR,
                   "Theory-based agent")
    _plot_with_sem(ax, trials, results["qlearn"]["reward_mean"],
                   results["qlearn"]["reward_sem"], QLEARN_COLOR,
                   "Q-learning agent")

    ax.set_xlabel("Trial", fontsize=13)
    ax.set_ylabel("Reward rate", fontsize=13)
    ax.set_title("Acquisition: single context, fixed contingency", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, "fig1_acquisition.png")


def plot_fig2():
    """Figure 2: Discrimination — P(correct) per context."""
    _setup_style()
    results = sim_discrimination()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    context_labels = ["Context 0 (light)", "Context 1 (tone)"]

    for i, ctx in enumerate([0, 1]):
        data = results[f"context_{ctx}"]
        trials = np.array(data["trials"])
        _plot_with_sem(axes[i], trials, data["theory_mean"], data["theory_sem"],
                       THEORY_COLOR, "Theory-based")
        _plot_with_sem(axes[i], trials, data["qlearn_mean"], data["qlearn_sem"],
                       QLEARN_COLOR, "Q-learning")
        axes[i].set_xlabel("Trial (within context)", fontsize=13)
        axes[i].set_title(context_labels[i], fontsize=14)
        axes[i].set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("P(correct action)", fontsize=13)
    axes[0].legend(fontsize=10)
    fig.suptitle("Discrimination: context-dependent action selection", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_discrimination.png")


def plot_fig3():
    """Figure 3: Contingency reversal — flagship figure."""
    _setup_style()
    results = sim_contingency_reversal()
    n_phase1 = results["n_phase1"]
    T = results["n_total"]
    trials = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot_with_sem(ax, trials, results["theory"]["correct_mean"],
                   results["theory"]["correct_sem"], THEORY_COLOR,
                   "Theory-based agent", window=9)
    _plot_with_sem(ax, trials, results["qlearn"]["correct_mean"],
                   results["qlearn"]["correct_sem"], QLEARN_COLOR,
                   "Q-learning agent", window=9)

    ax.axvline(x=n_phase1, color="gray", linestyle="--", linewidth=1.5,
               label="Contingency reversal")
    ax.set_xlabel("Trial", fontsize=13)
    ax.set_ylabel("P(correct action)", fontsize=13)
    ax.set_title("Contingency reversal: theory 0 (identity) → theory 1 (rotate +1)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, "fig3_contingency_reversal.png")


def plot_fig4():
    """Figure 4: Partial reinforcement extinction effect (PREE)."""
    _setup_style()
    results = sim_pree()
    n_acq = results["n_acq"]
    n_ext = results["n_ext"]
    n_total = n_acq + n_ext

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, (label, key) in enumerate([("CRF (p=0.9)", "crf"),
                                       ("PRF (p=0.5)", "prf")]):
        data = results[key]
        trials = np.arange(n_total)
        _plot_with_sem(axes[i], trials, data["reward_mean"], data["reward_sem"],
                       THEORY_COLOR, "Reward rate", window=7)
        axes[i].axvline(x=n_acq, color="gray", linestyle="--", linewidth=1.5)
        axes[i].set_xlabel("Trial", fontsize=13)
        axes[i].set_title(label, fontsize=14)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].text(n_acq / 2, 0.95, "Acquisition", ha="center", fontsize=9,
                     color="gray")
        axes[i].text(n_acq + n_ext / 2, 0.95, "Extinction", ha="center",
                     fontsize=9, color="gray")

    axes[0].set_ylabel("Reward rate", fontsize=13)
    fig.suptitle("Partial Reinforcement Extinction Effect", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_pree.png")


def plot_fig5():
    """Figure 5: Contingency degradation."""
    _setup_style()
    results = sim_contingency_degradation()
    T = results["n_trials"]
    trials = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, (label, key) in enumerate([("Contingent", "contingent"),
                                       ("Degraded", "degraded")]):
        data = results[key]
        _plot_with_sem(axes[i], trials, data["correct_mean"], data["correct_sem"],
                       THEORY_COLOR, "P(correct)", window=7)
        axes[i].set_xlabel("Trial", fontsize=13)
        axes[i].set_title(label, fontsize=14)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].axhline(y=1.0 / 3, color="gray", linestyle=":", alpha=0.5,
                        label="Chance")

    axes[0].set_ylabel("P(correct action)", fontsize=13)
    axes[0].legend(fontsize=10)
    fig.suptitle("Contingency degradation: free rewards impair learning",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_contingency_degradation.png")


def plot_fig6():
    """Figure 6: Posterior dynamics during contingency reversal."""
    _setup_style()
    results = sim_posterior_dynamics()
    n_phase1 = results["n_phase1"]
    T = results["n_total"]
    posteriors = np.array(results["posteriors_mean"])  # (T, 6)
    labels = results["theory_labels"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Stacked area plot
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    ax.stackplot(np.arange(T), posteriors.T, labels=labels, colors=colors,
                 alpha=0.85)

    ax.axvline(x=n_phase1, color="black", linestyle="--", linewidth=2,
               label="Contingency reversal")
    ax.set_xlabel("Trial", fontsize=13)
    ax.set_ylabel("P(theory)", fontsize=13)
    ax.set_title("Posterior dynamics: belief shifting at contingency change",
                 fontsize=13)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    _save(fig, "fig6_posterior_dynamics.png")


ALL_PLOTS = {
    1: plot_fig1,
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
    6: plot_fig6,
}


def plot_all():
    """Generate all figures."""
    for num, fn in sorted(ALL_PLOTS.items()):
        print(f"Generating Figure {num}...")
        fn()
    print("All figures generated.")
