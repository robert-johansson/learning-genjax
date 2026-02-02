"""Analytical GP filter, GenJAX generative model, and validation.

Implements the habituation model from Gershman 2024 (iScience 27, 110523).
The model treats habituation as Bayes-optimal filtering of a latent
Gaussian process observed through noisy signals.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm

import genjax
from genjax import gen, seed


# ---------------------------------------------------------------------------
# Analytical GP filter (pure JAX)
# ---------------------------------------------------------------------------

def sq_exp_kernel(z1, z2, length_scale):
    """Squared-exponential covariance matrix.

    k(z, z') = exp(-||z - z'||^2 / (2 * length_scale^2))

    Args:
        z1: (N, D) array of input locations.
        z2: (M, D) array of input locations.
        length_scale: kernel length-scale (scalar).

    Returns:
        (N, M) covariance matrix.
    """
    z1 = jnp.atleast_2d(z1)
    z2 = jnp.atleast_2d(z2)
    sq_dists = jnp.sum((z1[:, None, :] - z2[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (2.0 * length_scale ** 2))


def gp_posterior(z_obs, h_obs, z_test, alpha, length_scale, jitter=1e-6):
    """Compute GP posterior mean and variance at test points.

    Implements Equations 6-7 from the paper:
        x_hat = k_t^T (K + alpha I)^{-1} h
        sigma^2 = k_{t,t} - k_t^T (K + alpha I)^{-1} k_t

    Args:
        z_obs: (T, D) observed input locations.
        h_obs: (T,) observed signal values.
        z_test: (M, D) test input locations.
        alpha: observation noise variance.
        length_scale: kernel length-scale.
        jitter: diagonal jitter for numerical stability.

    Returns:
        (posterior_mean, posterior_var) — arrays of shape (M,).
    """
    z_obs = jnp.atleast_2d(z_obs)
    z_test = jnp.atleast_2d(z_test)

    K = sq_exp_kernel(z_obs, z_obs, length_scale)
    K_noisy = K + (alpha + jitter) * jnp.eye(K.shape[0])
    k_star = sq_exp_kernel(z_obs, z_test, length_scale)  # (T, M)
    k_test = sq_exp_kernel(z_test, z_test, length_scale)  # (M, M)

    L = jax.scipy.linalg.cholesky(K_noisy, lower=True)
    alpha_vec = jax.scipy.linalg.cho_solve((L, True), h_obs)
    v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)

    posterior_mean = k_star.T @ alpha_vec  # (M,)
    posterior_var = jnp.diag(k_test) - jnp.sum(v ** 2, axis=0)  # (M,)
    posterior_var = jnp.maximum(posterior_var, 1e-10)
    return posterior_mean, posterior_var


def response(posterior_mean, posterior_var, psi):
    """Compute response probability (Equation 5).

    y_t = Phi((x_hat_t - psi) / sigma_t)

    where Phi is the standard normal CDF.
    """
    sigma = jnp.sqrt(posterior_var)
    return jax_norm.cdf((posterior_mean - psi) / sigma)


def response_to_isolated_stimulus(h_value, alpha, length_scale, psi):
    """Response to a single isolated stimulus (for normalization).

    The posterior after one observation at t=0 with value h:
        mean = k(0,0) / (k(0,0) + alpha) * h = h / (1 + alpha)
        var  = k(0,0) - k(0,0)^2 / (k(0,0) + alpha)
             = 1 - 1/(1+alpha) = alpha/(1+alpha)
    """
    post_mean = h_value / (1.0 + alpha)
    post_var = alpha / (1.0 + alpha)
    return response(post_mean, post_var, psi)


def normalize_response(raw_responses, h_value, alpha, length_scale, psi):
    """Normalize responses: raw * 100 / response_to_isolated_stimulus."""
    baseline = response_to_isolated_stimulus(h_value, alpha, length_scale, psi)
    return 100.0 * raw_responses / baseline


def compute_habituation_response(z_all, h_all, alpha=0.3, length_scale=1.0,
                                 psi=0.5):
    """Online filtering loop: at each t, compute posterior from stimuli 0..t,
    predict response to stimulus t.

    Uses lax.scan over a padded/masked GP solve: the full T×T kernel matrix
    is computed once, and at step t, observations beyond index t are masked
    out via a large diagonal addition (making them uninformative).

    Args:
        z_all: (T, D) input locations (time, possibly with stimulus dimension).
        h_all: (T,) signal values at each time point.
        alpha: observation noise variance.
        length_scale: kernel length-scale.
        psi: response threshold.

    Returns:
        responses: (T,) array of raw (un-normalized) responses.
        means: (T,) posterior means.
        stds: (T,) posterior standard deviations.
    """
    z_all = jnp.atleast_2d(z_all)
    T = z_all.shape[0]

    K_full = sq_exp_kernel(z_all, z_all, length_scale)
    K_base = K_full + (alpha + 1e-6) * jnp.eye(T)
    indices = jnp.arange(T)

    def step(_, t):
        # Mask observations beyond index t with large diagonal
        mask = jnp.where(indices <= t, 0.0, 1e10)
        K_noisy = K_base + jnp.diag(mask)

        # Zero out unobserved signals
        h_masked = jnp.where(indices <= t, h_all, 0.0)

        # Solve for posterior at z_all[t]
        k_star = jax.lax.dynamic_slice(K_full, (0, t), (T, 1))
        k_tt = K_full[t, t]

        L = jax.scipy.linalg.cholesky(K_noisy, lower=True)
        alpha_vec = jax.scipy.linalg.cho_solve((L, True), h_masked)
        v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)

        post_mean = (k_star.T @ alpha_vec)[0]
        post_var = jnp.maximum(k_tt - jnp.sum(v ** 2), 1e-10)
        sigma = jnp.sqrt(post_var)
        r = jax_norm.cdf((post_mean - psi) / sigma)

        return None, (r, post_mean, sigma)

    _, (responses, means, stds) = jax.lax.scan(step, None, indices)
    return responses, means, stds


# ---------------------------------------------------------------------------
# GenJAX generative model
# ---------------------------------------------------------------------------

@gen
def habituation_gp_model(z_inputs, alpha, length_scale):
    """GenJAX generative model for the habituation GP.

    Prior: x_bar ~ GP(0, k)  where k is squared-exponential
    Observations: obs ~ N(x_bar, alpha * I)

    Args:
        z_inputs: (T, D) input locations.
        alpha: observation noise variance (scalar).
        length_scale: kernel length-scale (scalar).

    Returns:
        x_bar: the latent GP function values.
    """
    T = z_inputs.shape[0]
    K = sq_exp_kernel(z_inputs, z_inputs, length_scale)
    K = K + 1e-6 * jnp.eye(T)

    # Prior: draw latent function values from GP
    x_bar = genjax.multivariate_normal(jnp.zeros(T), K) @ "x_bar"

    # Observation model: noisy observations of x_bar
    obs = genjax.multivariate_normal(x_bar, alpha * jnp.eye(T)) @ "obs"

    return x_bar


# ---------------------------------------------------------------------------
# Validation: GenJAX importance sampling vs analytical posterior
# ---------------------------------------------------------------------------

def validate_genjax_vs_analytical(n_samples=5000, seed_val=42):
    """Compare GenJAX importance sampling posterior with analytical GP posterior.

    Runs importance sampling by constraining observations in the generative
    model, then computes weighted posterior statistics. Compares against
    the closed-form GP posterior.

    Args:
        n_samples: number of importance samples.
        seed_val: random seed.

    Returns:
        dict with analytical and IS posterior statistics, and errors.
    """
    alpha = 0.3
    length_scale = 1.0
    psi = 0.5
    T = 5

    # Stimulus: repeated stimulus at regular intervals
    z_inputs = jnp.linspace(0.1, 0.5, T).reshape(-1, 1)
    h_obs = jnp.ones(T) * 0.3

    # --- Analytical posterior ---
    anal_mean, anal_var = gp_posterior(z_inputs, h_obs, z_inputs,
                                       alpha, length_scale)

    # --- GenJAX importance sampling ---
    key = jax.random.PRNGKey(seed_val)

    def single_is(key):
        trace, log_w = seed(habituation_gp_model.generate)(
            key,
            {"obs": h_obs},
            z_inputs, alpha, length_scale
        )
        x_bar = trace.get_choices()["x_bar"]
        return x_bar, log_w

    keys = jax.random.split(key, n_samples)
    all_x_bar, all_log_w = jax.vmap(single_is)(keys)

    # Normalize importance weights
    log_w_normalized = all_log_w - jax.scipy.special.logsumexp(all_log_w)
    weights = jnp.exp(log_w_normalized)

    # Weighted posterior statistics
    is_mean = jnp.sum(weights[:, None] * all_x_bar, axis=0)
    is_var = jnp.sum(weights[:, None] * (all_x_bar - is_mean[None, :]) ** 2,
                     axis=0)

    mean_error = jnp.max(jnp.abs(is_mean - anal_mean))
    var_error = jnp.max(jnp.abs(is_var - anal_var))

    return {
        "analytical_mean": anal_mean,
        "analytical_var": anal_var,
        "is_mean": is_mean,
        "is_var": is_var,
        "mean_error": float(mean_error),
        "var_error": float(var_error),
        "n_samples": n_samples,
    }
