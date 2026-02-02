"""All simulations for theory-based RL model.

Each function builds context/theory sequences, runs both agents across
multiple seeds (vmapped), and returns mean +/- SEM for plotting.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .core import (
    build_theory_table,
    simulate_theory_agent,
    simulate_qlearning_agent,
    N_CONTEXTS,
    N_ACTIONS,
)


def _run_both_agents(context_seq, true_theory_seq, n_seeds=20, seed_base=0,
                     temperature=0.5, alpha=0.1, epsilon=0.1):
    """Run both agents across multiple seeds and return summary statistics.

    Returns:
        dict with 'theory' and 'qlearn' keys, each containing:
            'reward_mean': (T,) mean reward rate (rolling window).
            'reward_sem': (T,) SEM of reward rate.
            'correct_mean': (T,) mean P(correct action).
            'correct_sem': (T,) SEM of P(correct action).
            'posteriors_mean': (T, 6) mean posterior (theory agent only).
    """
    theory_table = build_theory_table()
    T = context_seq.shape[0]

    def run_theory(seed_idx):
        key = jax.random.PRNGKey(seed_base + seed_idx)
        trial_data, posteriors, action_values = simulate_theory_agent(
            key, theory_table, context_seq, true_theory_seq,
            temperature=temperature)
        return trial_data, posteriors, action_values

    def run_qlearn(seed_idx):
        key = jax.random.PRNGKey(seed_base + seed_idx)
        trial_data, q_tables = simulate_qlearning_agent(
            key, theory_table, context_seq, true_theory_seq,
            alpha=alpha, epsilon=epsilon)
        return trial_data, q_tables

    # Vmap across seeds
    theory_results = jax.vmap(run_theory)(jnp.arange(n_seeds))
    qlearn_results = jax.vmap(run_qlearn)(jnp.arange(n_seeds))

    # theory_results: (trial_data: (S,T,3), posteriors: (S,T,6), action_values: (S,T,3))
    # qlearn_results: (trial_data: (S,T,3), q_tables: (S,T,3,3))

    theory_rewards = theory_results[0][:, :, 2].astype(jnp.float32)  # (S, T)
    qlearn_rewards = qlearn_results[0][:, :, 2].astype(jnp.float32)

    # Compute correct action per trial for scoring
    theory_actions = theory_results[0][:, :, 1]  # (S, T)
    qlearn_actions = qlearn_results[0][:, :, 1]

    # Correct action under the true theory at each trial
    correct_actions = jnp.array([
        jnp.argmax(theory_table[true_theory_seq[t], context_seq[t], :])
        for t in range(T)
    ])  # (T,)

    theory_correct = (theory_actions == correct_actions[None, :]).astype(jnp.float32)
    qlearn_correct = (qlearn_actions == correct_actions[None, :]).astype(jnp.float32)

    def summarize(data):
        """data: (S, T) -> mean, sem over seeds."""
        mean = jnp.mean(data, axis=0)
        sem = jnp.std(data, axis=0) / jnp.sqrt(data.shape[0])
        return mean, sem

    theory_reward_mean, theory_reward_sem = summarize(theory_rewards)
    qlearn_reward_mean, qlearn_reward_sem = summarize(qlearn_rewards)
    theory_correct_mean, theory_correct_sem = summarize(theory_correct)
    qlearn_correct_mean, qlearn_correct_sem = summarize(qlearn_correct)

    return {
        "theory": {
            "reward_mean": theory_reward_mean,
            "reward_sem": theory_reward_sem,
            "correct_mean": theory_correct_mean,
            "correct_sem": theory_correct_sem,
            "posteriors_mean": jnp.mean(theory_results[1], axis=0),  # (T, 6)
        },
        "qlearn": {
            "reward_mean": qlearn_reward_mean,
            "reward_sem": qlearn_reward_sem,
            "correct_mean": qlearn_correct_mean,
            "correct_sem": qlearn_correct_sem,
        },
    }


def _smooth(arr, window=5):
    """Simple rolling mean for smoother plots."""
    arr = np.array(arr, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


# ---------------------------------------------------------------------------
# Simulation 1: Basic acquisition
# ---------------------------------------------------------------------------

def sim_acquisition(n_trials=60, n_seeds=30):
    """Figure 1: Basic learning — both agents converge on correct action.

    Single context (0), fixed theory (0), so correct action = 0.

    Returns:
        dict with 'theory' and 'qlearn' reward rate curves.
    """
    context_seq = jnp.zeros(n_trials, dtype=jnp.int32)
    true_theory_seq = jnp.zeros(n_trials, dtype=jnp.int32)

    results = _run_both_agents(context_seq, true_theory_seq, n_seeds=n_seeds)
    return {
        "n_trials": n_trials,
        "theory": results["theory"],
        "qlearn": results["qlearn"],
    }


# ---------------------------------------------------------------------------
# Simulation 2: Discrimination
# ---------------------------------------------------------------------------

def sim_discrimination(n_trials=120, n_seeds=30):
    """Figure 2: Context-dependent action selection.

    Two contexts (0, 1) alternate every trial. Theory 0 (identity):
    context 0 -> action 0, context 1 -> action 1.

    Returns:
        dict with per-context correct action probabilities.
    """
    # Alternating contexts
    context_seq = jnp.array([t % 2 for t in range(n_trials)], dtype=jnp.int32)
    true_theory_seq = jnp.zeros(n_trials, dtype=jnp.int32)

    theory_table = build_theory_table()

    def run_theory(seed_idx):
        key = jax.random.PRNGKey(seed_idx)
        trial_data, posteriors, action_values = simulate_theory_agent(
            key, theory_table, context_seq, true_theory_seq)
        return trial_data

    def run_qlearn(seed_idx):
        key = jax.random.PRNGKey(seed_idx)
        trial_data, _ = simulate_qlearning_agent(
            key, theory_table, context_seq, true_theory_seq)
        return trial_data

    n_seeds_arr = jnp.arange(n_seeds)
    theory_data = jax.vmap(run_theory)(n_seeds_arr)  # (S, T, 3)
    qlearn_data = jax.vmap(run_qlearn)(n_seeds_arr)

    # Correct action per context under theory 0
    # context 0 -> action 0, context 1 -> action 1
    results = {}
    for ctx in [0, 1]:
        mask = context_seq == ctx
        trial_indices = jnp.where(mask, size=n_trials // 2)[0]

        theory_actions_ctx = theory_data[:, trial_indices, 1]  # (S, T/2)
        qlearn_actions_ctx = qlearn_data[:, trial_indices, 1]

        correct_action = ctx  # under theory 0 identity
        theory_correct = (theory_actions_ctx == correct_action).astype(jnp.float32)
        qlearn_correct = (qlearn_actions_ctx == correct_action).astype(jnp.float32)

        results[f"context_{ctx}"] = {
            "theory_mean": jnp.mean(theory_correct, axis=0),
            "theory_sem": jnp.std(theory_correct, axis=0) / jnp.sqrt(n_seeds),
            "qlearn_mean": jnp.mean(qlearn_correct, axis=0),
            "qlearn_sem": jnp.std(qlearn_correct, axis=0) / jnp.sqrt(n_seeds),
            "trials": jnp.arange(trial_indices.shape[0]),
        }

    return results


# ---------------------------------------------------------------------------
# Simulation 3: Contingency reversal (flagship)
# ---------------------------------------------------------------------------

def sim_contingency_reversal(n_phase1=60, n_phase2=60, n_seeds=50):
    """Figure 3: Contingency reversal — theory agent adapts rapidly.

    Phase 1: Theory 0 (identity) for 60 trials, all 3 contexts cycling.
    Phase 2: Theory 1 (rotate +1) for 60 trials. All 3 context-action
             mappings change, so Q-learning must relearn everything.

    The theory agent shifts posterior mass and adapts within ~10 trials.
    The Q-learner must unlearn and relearn, taking ~30-50 trials.

    Returns:
        dict with reward/correct curves, reversal point.
    """
    n_total = n_phase1 + n_phase2
    # Cycle through all 3 contexts
    context_seq = jnp.array([t % N_CONTEXTS for t in range(n_total)], dtype=jnp.int32)
    true_theory_seq = jnp.concatenate([
        jnp.zeros(n_phase1, dtype=jnp.int32),
        jnp.full(n_phase2, 1, dtype=jnp.int32),  # rotate +1: all mappings change
    ])

    results = _run_both_agents(context_seq, true_theory_seq, n_seeds=n_seeds)
    results["n_phase1"] = n_phase1
    results["n_total"] = n_total
    return results


# ---------------------------------------------------------------------------
# Simulation 4: Partial reinforcement extinction effect (PREE)
# ---------------------------------------------------------------------------

def sim_pree(n_acq=60, n_ext=40, n_seeds=30):
    """Figure 4: Partial reinforcement extinction effect.

    CRF (continuous reinforcement): p=0.9 during acquisition, then extinction.
    PRF (partial reinforcement): p=0.5 during acquisition, then extinction.

    PREE: partially reinforced actions extinguish slower because the agent
    maintains more uncertainty about the correct theory.

    Returns:
        dict with CRF and PRF conditions, reward rates over time.
    """
    theory_table_crf = build_theory_table()  # p=0.9

    # PRF table: correct action gets p=0.5, others get p=0.1
    theory_table_prf = build_theory_table()
    theory_table_prf = jnp.where(theory_table_prf > 0.5, 0.5, theory_table_prf)

    # Extinction table: all actions give p=0.1 (no reward)
    theory_table_ext = jnp.full_like(theory_table_crf, 0.1)

    n_total = n_acq + n_ext
    context_seq = jnp.zeros(n_total, dtype=jnp.int32)
    true_theory_seq = jnp.zeros(n_total, dtype=jnp.int32)

    def run_condition(theory_table_acq, seed_idx):
        key = jax.random.PRNGKey(seed_idx)

        # Run acquisition phase with acquisition table
        trial_data_acq, posteriors_acq, _ = simulate_theory_agent(
            key, theory_table_acq, context_seq[:n_acq], true_theory_seq[:n_acq])

        # Continue with extinction table, using final posterior from acquisition
        key2 = jax.random.fold_in(key, 999)
        trial_data_ext, posteriors_ext, _ = simulate_theory_agent(
            key2, theory_table_ext, context_seq[:n_ext], true_theory_seq[:n_ext])

        rewards_acq = trial_data_acq[:, 2].astype(jnp.float32)
        rewards_ext = trial_data_ext[:, 2].astype(jnp.float32)
        return jnp.concatenate([rewards_acq, rewards_ext])

    seeds = jnp.arange(n_seeds)
    crf_rewards = jax.vmap(lambda s: run_condition(theory_table_crf, s))(seeds)  # (S, T)
    prf_rewards = jax.vmap(lambda s: run_condition(theory_table_prf, s))(seeds)

    def summarize(data):
        return jnp.mean(data, axis=0), jnp.std(data, axis=0) / jnp.sqrt(n_seeds)

    return {
        "n_acq": n_acq,
        "n_ext": n_ext,
        "crf": {"reward_mean": summarize(crf_rewards)[0],
                "reward_sem": summarize(crf_rewards)[1]},
        "prf": {"reward_mean": summarize(prf_rewards)[0],
                "reward_sem": summarize(prf_rewards)[1]},
    }


# ---------------------------------------------------------------------------
# Simulation 5: Contingency degradation
# ---------------------------------------------------------------------------

def sim_contingency_degradation(n_trials=80, n_seeds=30):
    """Figure 5: Non-contingent rewards weaken learning.

    Contingent: rewards only from correct action (standard theory table).
    Degraded: additional free rewards (p=0.3) on all actions.

    Returns:
        dict with contingent and degraded conditions.
    """
    theory_table_contingent = build_theory_table()

    # Degraded: raise floor probability so even wrong actions sometimes reward
    theory_table_degraded = build_theory_table()
    theory_table_degraded = jnp.where(
        theory_table_degraded < 0.5, 0.3, theory_table_degraded
    )

    context_seq = jnp.array([t % N_CONTEXTS for t in range(n_trials)], dtype=jnp.int32)
    true_theory_seq = jnp.zeros(n_trials, dtype=jnp.int32)

    def run_condition(table, seed_idx):
        key = jax.random.PRNGKey(seed_idx)
        trial_data, posteriors, _ = simulate_theory_agent(
            key, table, context_seq, true_theory_seq)
        correct_actions = jnp.array([
            jnp.argmax(table[0, context_seq[t], :]) for t in range(n_trials)
        ])
        correct = (trial_data[:, 1] == correct_actions).astype(jnp.float32)
        return correct

    seeds = jnp.arange(n_seeds)
    contingent_correct = jax.vmap(lambda s: run_condition(theory_table_contingent, s))(seeds)
    degraded_correct = jax.vmap(lambda s: run_condition(theory_table_degraded, s))(seeds)

    def summarize(data):
        return jnp.mean(data, axis=0), jnp.std(data, axis=0) / jnp.sqrt(n_seeds)

    return {
        "n_trials": n_trials,
        "contingent": {"correct_mean": summarize(contingent_correct)[0],
                       "correct_sem": summarize(contingent_correct)[1]},
        "degraded": {"correct_mean": summarize(degraded_correct)[0],
                     "correct_sem": summarize(degraded_correct)[1]},
    }


# ---------------------------------------------------------------------------
# Simulation 6: Posterior dynamics (reversal with full posterior)
# ---------------------------------------------------------------------------

def sim_posterior_dynamics(n_phase1=60, n_phase2=60, n_seeds=10):
    """Figure 6: Full posterior dynamics during contingency reversal.

    Same protocol as sim_contingency_reversal but returns the full
    posterior over theories for visualization.

    Returns:
        dict with mean posterior trajectory.
    """
    n_total = n_phase1 + n_phase2
    context_seq = jnp.array([t % N_CONTEXTS for t in range(n_total)], dtype=jnp.int32)
    true_theory_seq = jnp.concatenate([
        jnp.zeros(n_phase1, dtype=jnp.int32),
        jnp.full(n_phase2, 1, dtype=jnp.int32),  # rotate +1
    ])

    theory_table = build_theory_table()

    def run_one(seed_idx):
        key = jax.random.PRNGKey(seed_idx)
        _, posteriors, _ = simulate_theory_agent(
            key, theory_table, context_seq, true_theory_seq)
        return jnp.exp(posteriors)  # (T, 6) convert log-posterior to probabilities

    posteriors_all = jax.vmap(run_one)(jnp.arange(n_seeds))  # (S, T, 6)
    posteriors_mean = jnp.mean(posteriors_all, axis=0)  # (T, 6)

    return {
        "n_phase1": n_phase1,
        "n_total": n_total,
        "posteriors_mean": posteriors_mean,
        "theory_labels": [
            "T0: identity", "T1: rot+1", "T2: rot+2",
            "T3: swap(1,2)", "T4: swap(0,1)", "T5: reverse"
        ],
    }
