"""Matplotlib visualization for Gershman 2025 conditioning model."""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .simulations import (
    sim_rw_spacing,
    sim_rw_contingency,
    sim_rw_different_lr,
    sim_rate_estimation_error,
    sim_timescale_invariance,
    sim_informativeness,
    sim_spacing_effect,
    sim_contingency_degradation,
    sim_acquisition_extinction,
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
    """Figure 2: RW model failures — spacing and contingency invariance."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    # Left: spacing
    spacing = sim_rw_spacing()
    for label, (trials, r_hat) in spacing.items():
        ax1.plot(trials, r_hat, label=label)
    ax1.set_xlabel("Trial", fontsize=15)
    ax1.set_ylabel(r"$\hat{r}$", fontsize=15)
    ax1.legend(fontsize=15)

    # Right: contingency
    contingency = sim_rw_contingency()
    for label, (trials, r_hat) in contingency.items():
        ax2.plot(trials, r_hat, label=label)
    ax2.set_xlabel("Trial", fontsize=15)
    ax2.set_ylabel(r"$\hat{r}$", fontsize=15)
    ax2.legend(fontsize=15)

    fig.tight_layout()
    _save(fig, "fig2_rw_failures.png")


def plot_fig3():
    """Figure 3: RW with different learning rates still fails."""
    _setup_style()
    results = sim_rw_different_lr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    for label, (trials, r_hat) in results["spacing"].items():
        ax1.plot(trials, r_hat, label=label)
    ax1.set_xlabel("Trial", fontsize=15)
    ax1.set_ylabel(r"$\hat{r}$", fontsize=15)
    ax1.legend(fontsize=15)

    for label, (trials, r_hat) in results["contingency"].items():
        ax2.plot(trials, r_hat, label=label)
    ax2.set_xlabel("Trial", fontsize=15)
    ax2.set_ylabel(r"$\hat{r}$", fontsize=15)
    ax2.legend(fontsize=15)

    fig.tight_layout()
    _save(fig, "fig3_rw_different_lr.png")


def plot_fig4():
    """Figure 4: Proportional estimation error converging to 0."""
    _setup_style()
    results = sim_rate_estimation_error()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(results["time"], results["bg_error"], label="Background")
    ax.plot(results["time"], results["cs_error"], label="CS")

    ax.set_xlabel("Time (s)", fontsize=15)
    ax.set_ylabel("Proportional error", fontsize=15)
    ax.legend(fontsize=15)
    fig.tight_layout()
    _save(fig, "fig4_rate_estimation.png")


def plot_fig5():
    """Figure 5: Timescale invariance — decision variable curves."""
    _setup_style()
    results = sim_timescale_invariance()

    # Color cycle matching Gershman's winter colormap
    colors = plt.cm.winter(np.linspace(0, 1, 3))
    conditions = ["fixed_iti", "fixed_ratio"]
    titles = ["ITI = 48 s", "Informativeness = 6"]
    data_points = results["data_points"]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))

    for k, (cond, title) in enumerate(zip(conditions, titles)):
        panel = results[cond]
        for j, (label, (trials, dvs)) in enumerate(panel.items()):
            axs[k].plot(trials, dvs, color=colors[j], label=label)
            # Overlay data point from Gibbon et al. (1977)
            dp_trial = data_points[cond][j]
            if dp_trial < len(dvs):
                axs[k].plot(dp_trial, dvs[dp_trial - 1], "o",
                           color=colors[j], markersize=10)
        axs[k].set_xlabel("Trial", fontsize=15)
        axs[k].set_ylabel("Decision variable", fontsize=15)
        axs[k].set_title(title, fontsize=15)
        axs[k].set_ylim([1, 7])
    axs[0].legend(fontsize=12)

    fig.tight_layout()
    _save(fig, "fig5_timescale_invariance.png")


def plot_fig6():
    """Figure 6: Acquisition speed vs informativeness with curve fits."""
    _setup_style()
    datasets = sim_informativeness()

    fig, axs = plt.subplots(len(datasets), 1, figsize=(8, 11))

    for i, ds in enumerate(datasets):
        axs[i].loglog(ds["data_inf"], ds["data_R"], "o")
        axs[i].loglog(ds["inf_range"], ds["fit1"], "--",
                      label="$R^* = k/(C/T-1)$")
        axs[i].loglog(ds["inf_range"], ds["fit2"], "-",
                      label="$R^* = k/(C/T)$")
        axs[i].set_title(ds["name"], fontweight="bold")
        axs[i].set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 400])
        axs[i].set_xticklabels([1, 2, 5, 10, 20, 50, 100, 200, 400])
        axs[i].set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 400, 1000])
        axs[i].set_yticklabels([1, 2, 5, 10, 20, 50, 100, 200, 400, 1000])

    axs[0].legend(["Data", "$R^* = k/(C/T-1)$", "$R^* = k/(C/T)$"],
                  fontsize=12)
    fig.text(0.5, 0.07, "Informativeness (log scale)", ha="center",
             fontsize=15)
    fig.text(0.02, 0.5, "Reinforcements to acquisition (log scale)",
             va="center", rotation="vertical", fontsize=15)

    fig.tight_layout(rect=[0.05, 0.1, 1, 1])
    _save(fig, "fig6_informativeness.png")


def plot_fig7():
    """Figure 7: Spacing effect — our model."""
    _setup_style()
    results = sim_spacing_effect()

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (trials, dvs) in results.items():
        ax.plot(trials, dvs, label=label)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Decision variable (log info gain)")
    ax.set_title("Spacing effect (rate estimation model)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig7_spacing_effect.png")


def plot_fig8():
    """Figure 8: Contingency degradation — our model."""
    _setup_style()
    results = sim_contingency_degradation()

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (trials, dvs) in results.items():
        ax.plot(trials, dvs, label=label)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Decision variable (log info gain)")
    ax.set_title("Contingency degradation (rate estimation model)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig8_contingency_degradation.png")


def plot_fig9():
    """Figure 9: Acquisition, extinction, and spontaneous recovery."""
    _setup_style()
    results = sim_acquisition_extinction()

    fig, ax = plt.subplots(figsize=(9, 4))

    colors = {"acquisition": "steelblue", "extinction": "coral",
              "recovery": "seagreen"}
    labels = {"acquisition": "Acquisition (CS+US)",
              "extinction": "Extinction (CS only)",
              "recovery": "Reacquisition (CS+US)"}

    for phase in ["acquisition", "extinction", "recovery"]:
        trials, rates = results[phase]
        ax.plot(trials, rates, color=colors[phase], label=labels[phase],
                linewidth=2.5)

    # Mark phase boundaries
    ax.axvline(x=20, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=40, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Trial")
    ax.set_ylabel(r"CS rate estimate ($\hat{\lambda}_{CS}$)")
    ax.set_title("Acquisition, extinction, and reacquisition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig9_acquisition_extinction.png")


ALL_PLOTS = {
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
    6: plot_fig6,
    7: plot_fig7,
    8: plot_fig8,
    9: plot_fig9,
}


def plot_all():
    """Generate all figures."""
    for num, fn in sorted(ALL_PLOTS.items()):
        print(f"Generating Figure {num}...")
        fn()
    print("All figures generated.")
