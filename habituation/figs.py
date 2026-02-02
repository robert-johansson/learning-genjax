"""Matplotlib visualization for each figure from Gershman 2024."""

import os

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .simulations import (
    sim_simple_habituation,
    sim_frequency_intensity,
    sim_common_test,
    sim_spontaneous_recovery,
    sim_potentiation,
    sim_stimulus_specificity,
    sim_dishabituation,
    sim_length_scale_effects,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")


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


def plot_fig2():
    """Figure 2: Simple habituation illustration."""
    _setup_style()
    responses, means, stds, t_vals = sim_simple_habituation()
    t_flat = t_vals.flatten()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot posterior mean with error bars
    ax.errorbar(t_flat, means, yerr=stds, fmt="k-", capsize=3,
                label="Posterior mean +/- std")
    # Plot stimulus points
    ax.scatter(t_flat, jnp.ones(len(t_flat)) * 0.3,
               c="green", s=60, zorder=5, label="Stimulus")
    # Threshold line
    ax.axhline(y=0.5, color="steelblue", linestyle="--", linewidth=1.5,
               label="Threshold $\\psi$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.set_title("Figure 2: Illustration of the model")
    ax.legend(fontsize=8)
    _save(fig, "fig2_simple_habituation.png")


def plot_fig3():
    """Figure 3: Stimulus frequency and intensity effects."""
    _setup_style()
    results = sim_frequency_intensity()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False)
    layout = [
        (("Low", "Low"), (0, 0)),
        (("High", "Low"), (0, 1)),
        (("Low", "High"), (1, 0)),
        (("High", "High"), (1, 1)),
    ]
    for (i_label, f_label), (r, c) in layout:
        ax = axes[r, c]
        resp = results[(i_label, f_label)]
        ax.plot(jnp.arange(len(resp)), resp)
        ax.set_title(f"Intensity: {i_label}, Frequency: {f_label}")
        ax.set_xlabel("Repetition")
        if c == 0:
            ax.set_ylabel("Normalized response")

    fig.suptitle("Figure 3: Stimulus frequency and intensity effects",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_frequency_intensity.png")


def plot_fig4():
    """Figure 4: Common test procedure."""
    _setup_style()
    results = sim_common_test()

    fig, ax = plt.subplots(figsize=(6, 4))
    for f_label, (intervals, resp) in results.items():
        ax.plot(intervals, resp, label=f"{f_label} frequency")

    ax.set_xlabel("Test interval")
    ax.set_ylabel("Normalized response")
    ax.set_title("Figure 4: Common test procedure")
    ax.legend()
    _save(fig, "fig4_common_test.png")


def plot_fig5():
    """Figure 5: Spontaneous recovery."""
    _setup_style()
    results = sim_spontaneous_recovery()

    fig, ax = plt.subplots(figsize=(6, 4))
    for N, (delays, resp) in sorted(results.items()):
        ax.plot(delays, resp, label=f"N={N}")

    ax.set_xlabel("Delay")
    ax.set_ylabel("Normalized response")
    ax.set_title("Figure 5: Spontaneous recovery")
    ax.legend()
    _save(fig, "fig5_spontaneous_recovery.png")


def plot_fig6():
    """Figure 6: Potentiation."""
    _setup_style()
    norm1, norm2 = sim_potentiation()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(jnp.arange(len(norm1)), norm1, label="1st series")
    ax.plot(jnp.arange(len(norm2)), norm2, label="2nd series")

    ax.set_xlabel("Repetition")
    ax.set_ylabel("Normalized response")
    ax.set_title("Figure 6: Potentiation")
    ax.legend()
    _save(fig, "fig6_potentiation.png")


def plot_fig7():
    """Figure 7: Stimulus specificity."""
    _setup_style()
    distances, resp = sim_stimulus_specificity()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(distances, resp)
    ax.set_xlabel("Stimulus distance")
    ax.set_ylabel("Normalized response")
    ax.set_title("Figure 7: Stimulus specificity")
    _save(fig, "fig7_stimulus_specificity.png")


def plot_fig8():
    """Figure 8: Dishabituation."""
    _setup_style()
    results = sim_dishabituation()

    fig, ax = plt.subplots(figsize=(6, 4))
    conditions = ["None", "Weak", "Strong", "Repeat"]
    values = [results[c] for c in conditions]
    ax.bar(conditions, values, color=["steelblue", "cornflowerblue",
                                       "royalblue", "navy"])
    ax.set_xlabel("Dishabituation condition")
    ax.set_ylabel("Normalized response")
    ax.set_title("Figure 8: Dishabituation")
    _save(fig, "fig8_dishabituation.png")


def plot_fig9():
    """Figure 9: Short length-scale effects."""
    _setup_style()
    results = sim_length_scale_effects(length_scale_val=0.001)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False)
    layout = [
        (("Low", "Low"), (0, 0)),
        (("High", "Low"), (0, 1)),
        (("Low", "High"), (1, 0)),
        (("High", "High"), (1, 1)),
    ]
    for (i_label, f_label), (r, c) in layout:
        ax = axes[r, c]
        resp = results[(i_label, f_label)]
        ax.plot(jnp.arange(len(resp)), resp)
        ax.set_title(f"Intensity: {i_label}, Frequency: {f_label}")
        ax.set_xlabel("Repetition")
        if c == 0:
            ax.set_ylabel("Normalized response")

    fig.suptitle("Figure 9: Short length-scale ($\\lambda=0.001$)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig9_short_length_scale.png")


def plot_fig10():
    """Figure 10: Long length-scale effects."""
    _setup_style()
    results = sim_length_scale_effects(length_scale_val=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False)
    layout = [
        (("Low", "Low"), (0, 0)),
        (("High", "Low"), (0, 1)),
        (("Low", "High"), (1, 0)),
        (("High", "High"), (1, 1)),
    ]
    for (i_label, f_label), (r, c) in layout:
        ax = axes[r, c]
        resp = results[(i_label, f_label)]
        ax.plot(jnp.arange(len(resp)), resp)
        ax.set_title(f"Intensity: {i_label}, Frequency: {f_label}")
        ax.set_xlabel("Repetition")
        if c == 0:
            ax.set_ylabel("Normalized response")

    fig.suptitle("Figure 10: Long length-scale ($\\lambda=100$)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig10_long_length_scale.png")


ALL_PLOTS = {
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
    6: plot_fig6,
    7: plot_fig7,
    8: plot_fig8,
    9: plot_fig9,
    10: plot_fig10,
}


def plot_all():
    """Generate all figures."""
    for num, fn in sorted(ALL_PLOTS.items()):
        print(f"Generating Figure {num}...")
        fn()
    print("All figures generated.")
