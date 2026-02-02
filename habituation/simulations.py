"""All paper simulations (Figures 2-10) from Gershman 2024.

Each function builds stimulus inputs and signal values, runs the GP filter,
and returns normalized responses ready for plotting.

Note on frequencies: the paper uses alpha=0.3, lambda=1.0, psi=0.5 throughout.
The stimulus frequency controls how tightly packed stimuli are relative to the
kernel length-scale, which determines habituation depth. Higher frequency
(smaller ISI relative to lambda) produces deeper habituation.
"""

import jax
import jax.numpy as jnp

from .core import (
    compute_habituation_response,
    gp_posterior,
    normalize_response,
    response,
    response_to_isolated_stimulus,
)

# Default parameters from the paper
ALPHA = 0.3
LENGTH_SCALE = 1.0
PSI = 0.5


def sim_simple_habituation(n_reps=10, frequency=25, intensity=0.3,
                           alpha=ALPHA, length_scale=LENGTH_SCALE, psi=PSI):
    """Figure 2: Simple habituation — repeated stimulus, response decreases.

    Returns:
        responses: (n_reps,) normalized responses.
        means: (n_reps,) posterior means.
        stds: (n_reps,) posterior standard deviations.
        t: (n_reps, 1) time points.
    """
    t = jnp.linspace(1, n_reps, n_reps).reshape(-1, 1) / frequency
    h = jnp.ones(n_reps) * intensity

    raw, means, stds = compute_habituation_response(t, h, alpha, length_scale, psi)
    normalized = normalize_response(raw, intensity, alpha, length_scale, psi)
    return normalized, means, stds, t


def sim_frequency_intensity(n_reps=10, alpha=ALPHA, length_scale=LENGTH_SCALE,
                            psi=PSI):
    """Figure 3: 2x2 frequency x intensity effects.

    Low intensity + high frequency => stronger habituation.
    High intensity + high frequency => stronger sensitization.

    Returns:
        dict mapping (intensity_label, freq_label) -> normalized responses.
    """
    frequencies = {"Low": 2, "High": 10}
    intensities = {"Low": 0.3, "High": 0.7}

    results = {}
    for i_label, intensity in intensities.items():
        for f_label, freq in frequencies.items():
            t = jnp.linspace(1, n_reps, n_reps).reshape(-1, 1) / freq
            h = jnp.ones(n_reps) * intensity
            raw, _, _ = compute_habituation_response(t, h, alpha, length_scale, psi)
            normalized = normalize_response(raw, intensity, alpha, length_scale, psi)
            results[(i_label, f_label)] = normalized
    return results


def sim_common_test(n_hab=10, alpha=ALPHA, length_scale=LENGTH_SCALE, psi=PSI):
    """Figure 4: Common test procedure.

    After habituation at low or high frequency, test at various intervals.
    Davis (1970) showed low-frequency stimuli produce weaker test responses.

    Returns:
        dict with 'Low' and 'High' frequency keys, each mapping to
        (test_intervals, test_responses).
    """
    frequencies = {"Low": 2, "High": 10}
    intensity = 0.3
    test_intervals = jnp.linspace(0.1, 0.5, 10)

    results = {}
    for f_label, freq in frequencies.items():
        t_hab = jnp.linspace(1, n_hab, n_hab).reshape(-1, 1) / freq
        h_hab = jnp.ones(n_hab) * intensity

        def test_at_interval(ti):
            t_test = t_hab[-1, 0] + ti
            z_all = jnp.concatenate([t_hab, jnp.array([[t_test]])])
            h_all = jnp.concatenate([h_hab, jnp.array([intensity])])
            post_mean, post_var = gp_posterior(z_all, h_all,
                                               jnp.array([[t_test]]),
                                               alpha, length_scale)
            return response(post_mean, post_var, psi)[0]

        test_responses = jax.vmap(test_at_interval)(test_intervals)
        normalized = normalize_response(test_responses, intensity,
                                        alpha, length_scale, psi)
        results[f_label] = (test_intervals, normalized)
    return results


def sim_spontaneous_recovery(series_lengths=(15, 20), alpha=ALPHA,
                             length_scale=LENGTH_SCALE, psi=PSI):
    """Figure 5: Spontaneous recovery after rest.

    After habituation series of different lengths, test at various delays.
    Recovery is slower after longer series.

    Returns:
        dict mapping series_length -> (delays, test_responses).
    """
    intensity = 0.3
    frequency = 50
    n_delay_points = 12
    delays = jnp.linspace(0.05, 0.5, n_delay_points)

    results = {}
    for N in series_lengths:
        t_hab = jnp.linspace(1, N, N).reshape(-1, 1) / frequency
        h_hab = jnp.ones(N) * intensity

        def test_at_delay(delay):
            t_test = t_hab[-1, 0] + delay
            z_all = jnp.concatenate([t_hab, jnp.array([[t_test]])])
            h_all = jnp.concatenate([h_hab, jnp.array([intensity])])
            post_mean, post_var = gp_posterior(z_all, h_all,
                                               jnp.array([[t_test]]),
                                               alpha, length_scale)
            return response(post_mean, post_var, psi)[0]

        test_responses = jax.vmap(test_at_delay)(delays)
        normalized = normalize_response(test_responses, intensity,
                                        alpha, length_scale, psi)
        results[N] = (delays, normalized)
    return results


def sim_potentiation(n_reps=9, recovery_delay=0.5, frequency=50,
                     intensity=0.3, alpha=ALPHA, length_scale=LENGTH_SCALE,
                     psi=PSI):
    """Figure 6: Potentiation — 2nd series habituates faster than 1st.

    Returns:
        (series1_responses, series2_responses) — both normalized.
    """
    # 1st series
    t1 = jnp.linspace(1, n_reps, n_reps).reshape(-1, 1) / frequency
    h1 = jnp.ones(n_reps) * intensity
    raw1, _, _ = compute_habituation_response(t1, h1, alpha, length_scale, psi)
    norm1 = normalize_response(raw1, intensity, alpha, length_scale, psi)

    # 2nd series: append after a recovery delay (long enough for response
    # to return to baseline, but posterior variance remains lower)
    t2_start = t1[-1, 0] + recovery_delay
    t2 = jnp.linspace(1, n_reps, n_reps).reshape(-1, 1) / frequency + t2_start
    t_both = jnp.concatenate([t1, t2])
    h_both = jnp.concatenate([h1, jnp.ones(n_reps) * intensity])

    raw_both, _, _ = compute_habituation_response(t_both, h_both,
                                                  alpha, length_scale, psi)
    raw2 = raw_both[n_reps:]
    norm2 = normalize_response(raw2, intensity, alpha, length_scale, psi)

    return norm1, norm2


def sim_stimulus_specificity(n_hab=10, frequency=10, intensity=0.3,
                             alpha=ALPHA, length_scale=LENGTH_SCALE, psi=PSI):
    """Figure 7: Stimulus specificity — graded recovery to novel stimuli.

    Uses 2D input z = [t, s] where s is stimulus identity.
    Habituate to stimulus s=0, then test at various stimulus distances.

    Returns:
        (distances, test_responses) — normalized.
    """
    distances = jnp.linspace(0.0, 1.0, 11)

    t_hab = jnp.linspace(1, n_hab, n_hab) / frequency
    z_hab = jnp.column_stack([t_hab, jnp.zeros(n_hab)])
    h_hab = jnp.ones(n_hab) * intensity
    t_test_time = t_hab[-1] + 1.0 / frequency

    def test_at_distance(d):
        z_test = jnp.array([[t_test_time, d]])
        z_all = jnp.concatenate([z_hab, z_test])
        h_all = jnp.concatenate([h_hab, jnp.array([intensity])])
        post_mean, post_var = gp_posterior(z_all, h_all, z_test,
                                           alpha, length_scale)
        return response(post_mean, post_var, psi)[0]

    test_responses = jax.vmap(test_at_distance)(distances)
    normalized = normalize_response(test_responses, intensity,
                                    alpha, length_scale, psi)
    return distances, normalized


def sim_dishabituation(n_hab=20, frequency=50, intensity_familiar=0.3,
                       alpha=ALPHA, length_scale=LENGTH_SCALE, psi=PSI):
    """Figure 8: Dishabituation — novel stimulus restores response.

    Conditions:
    - None: no dishabituator, just test the familiar stimulus
    - Weak: 1 weak novel (s=1.0, h=0.3) then test familiar
    - Strong: 1 strong novel (s=1.0, h=0.7) then test familiar
    - Repeat: 20 interleaved rounds of (strong novel, familiar test),
      showing the last familiar test — habituation of dishabituation

    The interleaved "Repeat" protocol captures habituation of dishabituation:
    the accumulated familiar test data (all at h=0.3 < threshold) eventually
    outweighs the novel stimuli's dishabituating effect, bringing the
    familiar response back below the single-novel level.

    Returns:
        dict mapping condition_name -> normalized response to familiar stimulus.
    """
    isi = 1.0 / frequency

    # Habituation phase
    t_hab = jnp.linspace(1, n_hab, n_hab) / frequency
    z_hab = jnp.column_stack([t_hab, jnp.zeros(n_hab)])
    h_hab = jnp.ones(n_hab) * intensity_familiar
    t_base = t_hab[-1]

    baseline = float(response_to_isolated_stimulus(intensity_familiar, alpha,
                                                   length_scale, psi))
    results = {}

    # Condition 1: No dishabituator
    z_all = jnp.concatenate([z_hab, jnp.array([[t_base + isi, 0.0]])])
    h_all = jnp.concatenate([h_hab, jnp.array([intensity_familiar])])
    raw, _, _ = compute_habituation_response(z_all, h_all, alpha, length_scale, psi)
    results["None"] = 100.0 * float(raw[-1]) / baseline

    # Condition 2: Weak novel stimulus (s=1.0, h=0.3), then test familiar
    z_all = jnp.concatenate([z_hab,
                              jnp.array([[t_base + isi, 1.0]]),
                              jnp.array([[t_base + 2 * isi, 0.0]])])
    h_all = jnp.concatenate([h_hab, jnp.array([0.3, intensity_familiar])])
    raw, _, _ = compute_habituation_response(z_all, h_all, alpha, length_scale, psi)
    results["Weak"] = 100.0 * float(raw[-1]) / baseline

    # Condition 3: Strong novel stimulus (s=1.0, h=0.7), then test familiar
    z_all = jnp.concatenate([z_hab,
                              jnp.array([[t_base + isi, 1.0]]),
                              jnp.array([[t_base + 2 * isi, 0.0]])])
    h_all = jnp.concatenate([h_hab, jnp.array([0.7, intensity_familiar])])
    raw, _, _ = compute_habituation_response(z_all, h_all, alpha, length_scale, psi)
    results["Strong"] = 100.0 * float(raw[-1]) / baseline

    # Condition 4: Repeated interleaved (novel, familiar_test) rounds
    n_rounds = 20
    event_indices = jnp.arange(2 * n_rounds)
    is_novel = (event_indices % 2 == 0)
    event_t = t_base + (event_indices + 1) * isi
    event_s = jnp.where(is_novel, 1.0, 0.0)
    event_h = jnp.where(is_novel, 0.7, intensity_familiar)
    events_z = jnp.column_stack([event_t, event_s])
    z_all = jnp.concatenate([z_hab, events_z])
    h_all = jnp.concatenate([h_hab, event_h])
    raw, _, _ = compute_habituation_response(z_all, h_all, alpha, length_scale, psi)
    results["Repeat"] = 100.0 * float(raw[-1]) / baseline

    return results


def sim_length_scale_effects(length_scale_val, n_reps=10, alpha=ALPHA, psi=PSI):
    """Figures 9-10: Effects of short/long length-scale on habituation.

    Fig 9: length_scale=0.001 (very short — no generalization across time)
    Fig 10: length_scale=100 (very long — near-constant mean)

    Returns:
        Same format as sim_frequency_intensity.
    """
    return sim_frequency_intensity(n_reps=n_reps, alpha=alpha,
                                   length_scale=length_scale_val, psi=psi)
