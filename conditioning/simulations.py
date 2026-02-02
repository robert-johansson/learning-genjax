"""All simulations for Gershman 2025 conditioning model.

Each function builds event sequences, runs the Bayesian rate estimator or
Rescorla-Wagner model, and returns results ready for plotting.

Matches the reference implementation at github.com/sjgershm/rate_estimation.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from .core import (
    compute_conditioning_response,
    compute_rw_response,
    build_delay_conditioning_events,
    build_rw_delay_events,
    generate_poisson_events,
)

# Default parameters from the paper
R_0 = 0.1
N_0 = 1.0
ETA = 0.7
DT = 0.5

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# RW failure demonstrations (paper Figs 2-3)
# ---------------------------------------------------------------------------

def sim_rw_spacing(n_trials=80):
    """Figure 2 (left): RW model — learning invariant to ITI changes.

    Plots r_hat (US prediction at CS time), not raw weights.
    RW predicts identical r_hat curves regardless of ITI.

    Returns:
        dict mapping ITI label -> (trial_indices, r_hat_per_trial).
    """
    ISI = 2
    results = {}

    for label, ITI in [("ITI = 5", 5), ("ITI = 10", 10)]:
        events = build_rw_delay_events(n_trials, ISI, ITI)
        weights, predictions = compute_rw_response(events, n_stimuli=2,
                                                     alpha_lr=0.1)

        # Extract r_hat at US delivery (where r==1 in non-degraded trials)
        # US is at trial_time == ISI, i.e. every C steps at index ISI
        C = ISI + ITI
        us_indices = jnp.arange(n_trials) * C + ISI
        us_indices = jnp.minimum(us_indices, predictions.shape[0] - 1)
        r_hat = predictions[us_indices]
        results[label] = (jnp.arange(1, n_trials + 1), r_hat)

    return results


def sim_rw_contingency(n_trials=80):
    """Figure 2 (right): RW model — contingency degradation.

    Plots r_hat at US time. Standard RW is largely invariant to
    contingency degradation.

    Returns:
        dict mapping condition -> (trial_indices, r_hat_per_trial).
    """
    ISI = 2
    ITI = 10
    results = {}

    for label, degrade in [("Contingent", False), ("Degraded", True)]:
        events = build_rw_delay_events(n_trials, ISI, ITI, degrade=degrade)
        weights, predictions = compute_rw_response(events, n_stimuli=2,
                                                     alpha_lr=0.1)
        C = ISI + ITI
        us_indices = jnp.arange(n_trials) * C + ISI
        us_indices = jnp.minimum(us_indices, predictions.shape[0] - 1)
        r_hat = predictions[us_indices]
        results[label] = (jnp.arange(1, n_trials + 1), r_hat)

    return results


def sim_rw_different_lr(n_trials=80):
    """Figure 3: RW with different CS/context learning rates still fails.

    Uses alpha_bg=0.01 (background, index 1) and alpha_cs=0.1 (CS, index 0).
    Background has lower learning rate but RW still can't differentiate
    spacing or contingency.

    Returns:
        dict with 'spacing' and 'contingency' sub-dicts.
    """
    # Gershman's code: lrate=[0.01, 0.1] with [bg, CS] ordering
    # Our ordering is [CS, bg], so alpha = [0.1, 0.01]
    alpha = jnp.array([0.1, 0.01])
    ISI = 2
    results = {}

    # Spacing
    spacing = {}
    for label, ITI in [("ITI = 5", 5), ("ITI = 10", 10)]:
        events = build_rw_delay_events(n_trials, ISI, ITI)
        _, predictions = compute_rw_response(events, n_stimuli=2,
                                              alpha_lr=alpha)
        C = ISI + ITI
        us_indices = jnp.arange(n_trials) * C + ISI
        us_indices = jnp.minimum(us_indices, predictions.shape[0] - 1)
        spacing[label] = (jnp.arange(1, n_trials + 1), predictions[us_indices])
    results["spacing"] = spacing

    # Contingency
    contingency = {}
    ITI = 10
    for label, degrade in [("Contingent", False), ("Degraded", True)]:
        events = build_rw_delay_events(n_trials, ISI, ITI, degrade=degrade)
        _, predictions = compute_rw_response(events, n_stimuli=2,
                                              alpha_lr=alpha)
        C = ISI + ITI
        us_indices = jnp.arange(n_trials) * C + ISI
        us_indices = jnp.minimum(us_indices, predictions.shape[0] - 1)
        contingency[label] = (jnp.arange(1, n_trials + 1),
                               predictions[us_indices])
    results["contingency"] = contingency

    return results


# ---------------------------------------------------------------------------
# Rate estimation model demonstrations (paper Figs 4-6)
# ---------------------------------------------------------------------------

def sim_rate_estimation_error():
    """Figure 4: Proportional estimation error over time.

    Generates Poisson events with known rates (lambda_bg=0.5, lambda_cs=1.5),
    runs the Bayesian rate estimator, and plots |lambda_hat - lambda| / lambda.

    Matches Gershman's cell-2: ISI=2, ITI=5, lambda=[0.5, 1.5], 25000 seconds.

    Returns:
        dict with 'time', 'bg_error', 'cs_error' arrays.
    """
    ISI = 2
    ITI = 5
    lambda_bg = 0.5
    lambda_cs = 1.5
    total_time = 25000

    events = generate_poisson_events(ISI, ITI, lambda_bg, lambda_cs,
                                      total_time, step_size=DT, seed=1)

    lambda_hats, _, _ = compute_conditioning_response(
        events, n_stimuli=2, r_0=R_0, n_0=N_0, eta=ETA, dt=DT
    )

    # Sample at 1-second intervals for plotting
    sample_indices = jnp.arange(0, lambda_hats.shape[0], int(1.0 / DT))
    sample_indices = jnp.minimum(sample_indices, lambda_hats.shape[0] - 1)

    cs_est = lambda_hats[sample_indices, 0]
    bg_est = lambda_hats[sample_indices, 1]

    cs_error = jnp.abs(cs_est - lambda_cs) / lambda_cs
    bg_error = jnp.abs(bg_est - lambda_bg) / lambda_bg
    time = jnp.arange(len(sample_indices))

    return {
        "time": time,
        "bg_error": bg_error,
        "cs_error": cs_error,
    }


def sim_timescale_invariance(r_0=R_0, n_0=N_0, eta=ETA, dt=DT):
    """Figure 5: Decision variable curves over trials.

    Left: Fixed ITI=48, ISI=[4,8,16] (informativeness varies).
    Right: Fixed informativeness=6, ISI=[4,8,16] (ITI scales with ISI).

    Overlays data from Gibbon et al. (1977) pigeon autoshaping.

    Returns:
        dict with 'fixed_iti' and 'fixed_ratio' sub-dicts, plus 'data_points'.
    """
    n_trials = 80
    ISIs = [4, 8, 16]
    Inf_val = 6
    results = {}

    # Data points from Gallistel & Gibbon (2000), Figure 11
    # [trial index for ISI=4, ISI=8, ISI=16]
    data_points = {
        "fixed_iti": [19, 45, 71],
        "fixed_ratio": [39, 45, 36],
    }

    # Left panel: Fixed ITI=48
    fixed_iti = {}
    ITI = 48
    for ISI in ISIs:
        label = f"ISI = {ISI} s"
        C = ISI + ITI
        events = build_delay_conditioning_events(n_trials, ISI, ITI, dt)
        _, dvs, _ = compute_conditioning_response(events, n_stimuli=2,
                                                    r_0=r_0, n_0=n_0,
                                                    eta=eta, dt=dt)
        trial_len = int(ISI / dt) + int(ITI / dt)
        trial_indices = jnp.arange(n_trials) * trial_len + int(ISI / dt) - 1
        trial_indices = jnp.minimum(trial_indices, dvs.shape[0] - 1)
        fixed_iti[label] = (jnp.arange(1, n_trials + 1), dvs[trial_indices])
    results["fixed_iti"] = fixed_iti

    # Right panel: Fixed informativeness (C/T = Inf_val)
    fixed_ratio = {}
    for ISI in ISIs:
        ITI = ISI * (Inf_val - 1)
        label = f"ISI = {ISI} s"
        events = build_delay_conditioning_events(n_trials, ISI, ITI, dt)
        _, dvs, _ = compute_conditioning_response(events, n_stimuli=2,
                                                    r_0=r_0, n_0=n_0,
                                                    eta=eta, dt=dt)
        trial_len = int(ISI / dt) + int(ITI / dt)
        trial_indices = jnp.arange(n_trials) * trial_len + int(ISI / dt) - 1
        trial_indices = jnp.minimum(trial_indices, dvs.shape[0] - 1)
        fixed_ratio[label] = (jnp.arange(1, n_trials + 1), dvs[trial_indices])
    results["fixed_ratio"] = fixed_ratio

    results["data_points"] = data_points
    return results


def sim_informativeness():
    """Figure 6: Acquisition speed vs informativeness — real data + curve fits.

    Loads three experimental datasets and fits two models:
      acq1: log(R*) = log(k) - log(C/T - 1)   [Gallistel & Harris 2024]
      acq2: log(R*) = log(k) - log(C/T)        [new, from this paper]

    Computes BIC for model comparison.

    Returns:
        dict with 'datasets' list, each containing data, fits, and BIC.
    """
    import pandas as pd
    from scipy.optimize import curve_fit

    def acq1(Inf, p):
        """R* = k/(C/T - 1): Gallistel & Harris model."""
        return np.log(p) - np.log(Inf - 1)

    def acq2(Inf, p):
        """R* = k/(C/T): new model from this paper."""
        return np.log(p) - np.log(Inf)

    def compute_BIC(p, x, y, fun):
        n = len(y)
        k = len(p)
        residuals = y - fun(x, *p)
        sse = np.sum(residuals**2)
        return n * np.log(sse / n) + k * np.log(n)

    dataset_files = ["GibbonBalsam81.csv", "Balsam24.csv",
                      "HarrisGallistel24.csv"]
    dataset_names = ["Gibbon & Balsam (1981)", "Balsam et al. (2024)",
                      "Harris & Gallistel (2024)"]

    Inf_range = np.logspace(np.log10(1.5), np.log10(400), 100)
    results = []

    for fname, name in zip(dataset_files, dataset_names):
        path = os.path.join(DATA_DIR, fname)
        data = pd.read_csv(path).dropna()

        # Fit model 1: R* = k/(C/T - 1)
        p1, _ = curve_fit(acq1, data.Inf, np.log(data.R))
        fit1 = np.exp(acq1(Inf_range, p1[0]))
        BIC1 = compute_BIC(p1, data.Inf, np.log(data.R), acq1)

        # Fit model 2: R* = k/(C/T)
        p2, _ = curve_fit(acq2, data.Inf, np.log(data.R))
        fit2 = np.exp(acq2(Inf_range, p2[0]))
        BIC2 = compute_BIC(p2, data.Inf, np.log(data.R), acq2)

        # Bayes factor approximation
        BF = 0.5 * (BIC1 - BIC2)
        prob_model2 = 1.0 / (1.0 + np.exp(-BF))

        results.append({
            "name": name,
            "data_inf": np.array(data.Inf),
            "data_R": np.array(data.R),
            "inf_range": Inf_range,
            "fit1": fit1,
            "fit2": fit2,
            "BIC1": BIC1,
            "BIC2": BIC2,
            "prob_model2": prob_model2,
        })

    return results


# ---------------------------------------------------------------------------
# Our model demonstrations (bonus figures showing what RW can't do)
# ---------------------------------------------------------------------------

def sim_spacing_effect(n_trials=30, r_0=R_0, n_0=N_0, eta=ETA, dt=DT):
    """Figure 7: Spacing effect — longer ITI = faster acquisition.

    Our model predicts that more widely spaced trials lead to faster
    growth of the decision variable, because the background rate
    estimate decreases with longer ITI exposure without reinforcement.

    Returns:
        dict mapping ITI label -> (trials, decision_vars).
    """
    T = 1.0
    results = {}

    for label, I in [("ITI=2", 2.0), ("ITI=5", 5.0), ("ITI=10", 10.0)]:
        events = build_delay_conditioning_events(n_trials, T, I, dt)
        _, dvs, _ = compute_conditioning_response(events, n_stimuli=2,
                                                    r_0=r_0, n_0=n_0,
                                                    eta=eta, dt=dt)
        trial_len = int(T / dt) + int(I / dt)
        trial_indices = jnp.arange(n_trials) * trial_len + int(T / dt) - 1
        trial_indices = jnp.minimum(trial_indices, dvs.shape[0] - 1)
        results[label] = (jnp.arange(n_trials), dvs[trial_indices])

    return results


def sim_contingency_degradation(n_trials=30, r_0=R_0, n_0=N_0, eta=ETA,
                                 dt=DT):
    """Figure 8: Contingency degradation — US during ITI slows acquisition.

    Adding unreinforced US during ITI increases the background rate estimate,
    reducing the information gain from the CS.

    Returns:
        dict mapping condition -> (trials, decision_vars).
    """
    T = 1.0
    I = 5.0
    results = {}

    for label, extra_us in [("Contingent", False), ("Degraded", True)]:
        events = build_delay_conditioning_events(n_trials, T, I, dt,
                                                  extra_us_during_iti=extra_us)
        _, dvs, _ = compute_conditioning_response(events, n_stimuli=2,
                                                    r_0=r_0, n_0=n_0,
                                                    eta=eta, dt=dt)
        trial_len = int(T / dt) + int(I / dt)
        trial_indices = jnp.arange(n_trials) * trial_len + int(T / dt) - 1
        trial_indices = jnp.minimum(trial_indices, dvs.shape[0] - 1)
        results[label] = (jnp.arange(n_trials), dvs[trial_indices])

    return results


def sim_acquisition_extinction(n_acq=20, n_ext=20, n_recovery_test=10,
                                r_0=R_0, n_0=N_0, eta=ETA, dt=DT):
    """Figure 9: Acquisition, extinction, and spontaneous recovery.

    Phase 1: Acquisition (CS + US pairings) — CS rate rises
    Phase 2: Extinction (CS without US) — CS rate falls
    Phase 3: After a gap, reacquisition — CS rate rises again (faster)

    Tracks the CS rate estimate (lambda_CS), which is more interpretable
    than the decision variable for extinction dynamics.

    Returns:
        dict with 'acquisition', 'extinction', 'recovery' phases,
        each containing (trial_indices, cs_rates).
    """
    T = 1.0
    I = 5.0

    # Acquisition: CS-US pairings
    acq_events = build_delay_conditioning_events(n_acq, T, I, dt)

    # Extinction: CS without US (same timing, r=0)
    ext_events_list = []
    for _ in range(n_ext):
        n_cs_steps = max(1, int(T / dt))
        for _ in range(n_cs_steps):
            ext_events_list.append([1.0, 1.0, 0.0])
        n_iti_steps = max(1, int(I / dt))
        for _ in range(n_iti_steps):
            ext_events_list.append([0.0, 1.0, 0.0])
    ext_events = jnp.array(ext_events_list)

    # Recovery gap: just background for a while
    gap_steps = int(20.0 / dt)
    gap_events = jnp.zeros((gap_steps, 3))
    gap_events = gap_events.at[:, 1].set(1.0)

    # Recovery test: CS-US pairings again
    test_events = build_delay_conditioning_events(n_recovery_test, T, I, dt)

    # Concatenate all phases
    all_events = jnp.concatenate([acq_events, ext_events, gap_events,
                                   test_events])

    lambda_hats, _, _ = compute_conditioning_response(all_events, n_stimuli=2,
                                                       r_0=r_0, n_0=n_0,
                                                       eta=eta, dt=dt)

    # Extract CS rate at end of each trial for each phase
    trial_len = int(T / dt) + int(I / dt)
    acq_offset = 0
    ext_offset = acq_events.shape[0]
    gap_offset = ext_offset + ext_events.shape[0]
    test_offset = gap_offset + gap_events.shape[0]

    def extract_cs_rates(offset, n_trials_phase):
        indices = offset + jnp.arange(n_trials_phase) * trial_len + int(T / dt) - 1
        indices = jnp.minimum(indices, lambda_hats.shape[0] - 1)
        return lambda_hats[indices, 0]

    acq_rates = extract_cs_rates(acq_offset, n_acq)
    ext_rates = extract_cs_rates(ext_offset, n_ext)
    test_rates = extract_cs_rates(test_offset, n_recovery_test)

    return {
        "acquisition": (jnp.arange(n_acq), acq_rates),
        "extinction": (jnp.arange(n_ext) + n_acq, ext_rates),
        "recovery": (jnp.arange(n_recovery_test) + n_acq + n_ext, test_rates),
    }
