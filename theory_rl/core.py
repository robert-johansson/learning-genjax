"""Theory table, agents (lax.scan folds), GenJAX generative model, and validation.

Implements theory-based reinforcement learning from Tomov, Tsividis, Pouncy,
Tenenbaum & Gershman (2023). "The Neural Architecture of Theory-Based
Reinforcement Learning." Neuron, 111(8):1331-1344.

The agent infers hidden contingency rules (theories) mapping (context, action)
pairs to reward probabilities. This is the third instance of the organism-as-fold
pattern, after habituation (GP kernel) and conditioning (rate estimation).
"""

import jax
import jax.numpy as jnp

import genjax
from genjax import gen

# Number of contexts, actions, and theories
N_CONTEXTS = 3
N_ACTIONS = 3
N_THEORIES = 6

# Reward probabilities
P_REWARD = 0.9
P_NO_REWARD = 0.1


# ---------------------------------------------------------------------------
# Theory table
# ---------------------------------------------------------------------------

def build_theory_table():
    """Build (6, 3, 3) reward probability array for all theories.

    Each theory maps (context, action) -> reward probability.
    The "correct" action for context c under theory k gets P_REWARD;
    all other actions get P_NO_REWARD.

    Theory 0 (identity):   context c -> action c
    Theory 1 (rotate +1):  context c -> action (c+1) % 3
    Theory 2 (rotate +2):  context c -> action (c+2) % 3
    Theory 3 (swap 1,2):   0->0, 1->2, 2->1
    Theory 4 (swap 0,1):   0->1, 1->0, 2->2
    Theory 5 (reverse):    0->2, 1->1, 2->0

    Returns:
        (6, 3, 3) JAX array where table[k, c, a] = P(reward | theory k, context c, action a).
    """
    # Define the correct action for each (theory, context) pair
    mappings = jnp.array([
        [0, 1, 2],  # identity
        [1, 2, 0],  # rotate +1
        [2, 0, 1],  # rotate +2
        [0, 2, 1],  # swap 1,2
        [1, 0, 2],  # swap 0,1
        [2, 1, 0],  # reverse
    ])  # (6, 3) — mappings[k, c] = correct action

    table = jnp.full((N_THEORIES, N_CONTEXTS, N_ACTIONS), P_NO_REWARD)
    for k in range(N_THEORIES):
        for c in range(N_CONTEXTS):
            table = table.at[k, c, mappings[k, c]].set(P_REWARD)

    return table


# ---------------------------------------------------------------------------
# Theory-based agent (pure fold via lax.scan)
# ---------------------------------------------------------------------------

def simulate_theory_agent(key, theory_table, context_seq, true_theory_seq,
                          temperature=0.5, stickiness=0.95):
    """Theory-based agent via lax.scan — Bayesian posterior with change detection.

    State: log_posterior over theories (6,)
    Input per step: context from environment
    Action selection: Thompson sampling (sample theory, act greedily)
    Update: HMM-style prediction (mix with uniform for change detection),
            then Bayesian categorical update given (context, action, reward)

    The stickiness parameter models the agent's prior that the current
    theory persists: P(stay) = stickiness, P(change to any) = (1-stickiness)/K.
    This prevents the posterior from becoming so concentrated that it
    cannot adapt when contingencies actually change.

    Args:
        key: JAX PRNG key.
        theory_table: (6, 3, 3) reward probability table.
        context_seq: (T,) int array of contexts per trial.
        true_theory_seq: (T,) int array of true theory per trial.
        temperature: softmax temperature for action selection (lower = greedier).
        stickiness: probability of theory persistence per trial (0-1).

    Returns:
        trial_data: (T, 3) int array [context, action, reward].
        posteriors: (T, 6) log-posterior over theories at each step.
        action_values: (T, 3) expected reward per action at each step.
    """
    T = context_seq.shape[0]
    K = theory_table.shape[0]

    def step(carry, t):
        key, log_posterior = carry
        context = context_seq[t]
        true_theory = true_theory_seq[t]

        # --- HMM prediction step: mix with uniform for change detection ---
        # P_pred(k) = stickiness * P_prev(k) + (1-stickiness)/K
        posterior_prev = jnp.exp(log_posterior)
        posterior_pred = stickiness * posterior_prev + (1.0 - stickiness) / K
        log_posterior_pred = jnp.log(posterior_pred + 1e-10)

        # --- Expected action values under predicted posterior ---
        expected_rewards = jnp.einsum('k,ka->a', posterior_pred,
                                       theory_table[:, context, :])

        # --- Action selection (Thompson sampling) ---
        key, k1, k2 = jax.random.split(key, 3)
        sampled_theory = jax.random.categorical(k1, log_posterior_pred)
        # Act greedily under sampled theory (with tiny noise for tie-breaking)
        action_probs = theory_table[sampled_theory, context, :]
        action = jnp.argmax(action_probs + jax.random.uniform(k2, (N_ACTIONS,)) * 1e-6)

        # --- Environment response ---
        key, k3 = jax.random.split(key)
        reward_prob = theory_table[true_theory, context, action]
        reward = jax.random.bernoulli(k3, reward_prob).astype(jnp.int32)

        # --- Bayesian update (likelihood * predicted prior) ---
        p_k = theory_table[:, context, action]  # (6,) P(reward|theory k, c, a)
        log_lik = (reward * jnp.log(p_k + 1e-10)
                   + (1 - reward) * jnp.log(1 - p_k + 1e-10))
        log_posterior_new = log_posterior_pred + log_lik
        log_posterior_new = log_posterior_new - jax.scipy.special.logsumexp(log_posterior_new)

        new_carry = (key, log_posterior_new)
        output = (context, action, reward, log_posterior_new, expected_rewards)
        return new_carry, output

    init_log_posterior = jnp.full(K, -jnp.log(float(K)))
    init_carry = (key, init_log_posterior)
    _, (contexts, actions, rewards, posteriors, action_values) = jax.lax.scan(
        step, init_carry, jnp.arange(T)
    )

    trial_data = jnp.stack([contexts, actions, rewards], axis=-1)
    return trial_data, posteriors, action_values


# ---------------------------------------------------------------------------
# Q-learning agent (pure fold via lax.scan, for comparison)
# ---------------------------------------------------------------------------

def simulate_qlearning_agent(key, theory_table, context_seq, true_theory_seq,
                             alpha=0.1, epsilon=0.1):
    """Q-learning agent via lax.scan — epsilon-greedy.

    State: Q-table (3, 3) — Q[context, action]
    Update: Q[c,a] += alpha * (reward - Q[c,a])

    Args:
        key: JAX PRNG key.
        theory_table: (6, 3, 3) reward probability table.
        context_seq: (T,) int array of contexts per trial.
        true_theory_seq: (T,) int array of true theory per trial.
        alpha: learning rate.
        epsilon: exploration probability.

    Returns:
        trial_data: (T, 3) int array [context, action, reward].
        q_tables: (T, 3, 3) Q-values over time.
    """
    T = context_seq.shape[0]

    def step(carry, t):
        key, Q = carry
        context = context_seq[t]
        true_theory = true_theory_seq[t]

        # --- Action selection (epsilon-greedy) ---
        key, k1, k2, k3 = jax.random.split(key, 4)
        q_vals = Q[context]

        # Greedy action (with tie-breaking noise)
        greedy_action = jnp.argmax(q_vals + jax.random.uniform(k1, (N_ACTIONS,)) * 1e-6)
        random_action = jax.random.randint(k2, (), 0, N_ACTIONS)
        explore = jax.random.bernoulli(k3, epsilon)
        action = jnp.where(explore, random_action, greedy_action)

        # --- Environment response ---
        key, k4 = jax.random.split(key)
        reward_prob = theory_table[true_theory, context, action]
        reward = jax.random.bernoulli(k4, reward_prob).astype(jnp.int32)

        # --- Q-update ---
        q_old = Q[context, action]
        q_new = q_old + alpha * (reward - q_old)
        Q_new = Q.at[context, action].set(q_new)

        new_carry = (key, Q_new)
        output = (context, action, reward, Q_new)
        return new_carry, output

    init_Q = jnp.full((N_CONTEXTS, N_ACTIONS), 0.5)
    init_carry = (key, init_Q)
    _, (contexts, actions, rewards, q_tables) = jax.lax.scan(
        step, init_carry, jnp.arange(T)
    )

    trial_data = jnp.stack([contexts, actions, rewards], axis=-1)
    return trial_data, q_tables


# ---------------------------------------------------------------------------
# GenJAX generative model
# ---------------------------------------------------------------------------

@gen
def operant_theory_model(contexts, actions, theory_table):
    """GenJAX generative model for Bayesian operant learning.

    Prior: theta ~ Categorical(uniform over 6 theories)
    Observations: For each trial, reward_t ~ Bernoulli(theory_table[theta, c_t, a_t])

    Args:
        contexts: (T,) int array of contexts.
        actions: (T,) int array of actions taken.
        theory_table: (6, 3, 3) reward probability table.

    Returns:
        theta (sampled theory index).
    """
    log_prior = jnp.full(N_THEORIES, -jnp.log(float(N_THEORIES)))
    theta = genjax.categorical(log_prior) @ "theta"

    reward_probs = theory_table[theta, contexts, actions]
    obs = genjax.bernoulli(reward_probs) @ "rewards"

    return theta


# ---------------------------------------------------------------------------
# Validation: IS vs exact categorical posterior
# ---------------------------------------------------------------------------

def validate_is_vs_exact(n_samples=50000, seed_val=42):
    """Compare importance sampling posterior with exact Bayesian categorical posterior.

    Uses a short trial sequence where we can compute the exact posterior
    analytically. We sample theta from the categorical prior and weight
    by the Bernoulli likelihood — the resulting IS posterior should match
    the exact categorical posterior.

    Args:
        n_samples: number of importance samples.
        seed_val: random seed.

    Returns:
        dict with exact and IS posterior distributions, and max error.
    """
    theory_table = build_theory_table()

    # Test scenario: 10 trials, context 0, alternating actions, theory 0 is true
    n_trials = 10
    contexts = jnp.zeros(n_trials, dtype=jnp.int32)
    actions = jnp.array([0, 1, 2, 0, 0, 1, 0, 2, 0, 0], dtype=jnp.int32)
    # Generate rewards from theory 0: context 0 -> action 0 rewarded
    rewards = jnp.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=jnp.int32)

    # --- Exact posterior ---
    log_prior = jnp.full(N_THEORIES, -jnp.log(float(N_THEORIES)))
    log_posterior_exact = log_prior.copy()
    for t in range(n_trials):
        p_k = theory_table[:, contexts[t], actions[t]]
        log_lik = (rewards[t] * jnp.log(p_k + 1e-10)
                   + (1 - rewards[t]) * jnp.log(1 - p_k + 1e-10))
        log_posterior_exact = log_posterior_exact + log_lik
    log_posterior_exact = log_posterior_exact - jax.scipy.special.logsumexp(log_posterior_exact)
    posterior_exact = jnp.exp(log_posterior_exact)

    # --- Importance sampling ---
    key = jax.random.PRNGKey(seed_val)

    def single_is(key):
        # Sample theta from prior
        theta = jax.random.categorical(key, log_prior)
        # Compute log likelihood
        reward_probs = theory_table[theta, contexts, actions]
        log_lik = jnp.sum(
            rewards * jnp.log(reward_probs + 1e-10)
            + (1 - rewards) * jnp.log(1 - reward_probs + 1e-10)
        )
        return theta, log_lik

    keys = jax.random.split(key, n_samples)
    all_theta, all_log_w = jax.vmap(single_is)(keys)

    # Normalize importance weights
    log_w_normalized = all_log_w - jax.scipy.special.logsumexp(all_log_w)
    weights = jnp.exp(log_w_normalized)

    # Compute IS posterior: weighted histogram
    posterior_is = jnp.zeros(N_THEORIES)
    for k in range(N_THEORIES):
        posterior_is = posterior_is.at[k].set(jnp.sum(weights * (all_theta == k)))

    max_error = float(jnp.max(jnp.abs(posterior_exact - posterior_is)))

    return {
        "exact_posterior": posterior_exact,
        "is_posterior": posterior_is,
        "max_error": max_error,
        "n_samples": n_samples,
    }
