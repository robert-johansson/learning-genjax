# A Minimal Organism as a Pure Functional Fold: Habituation via Gaussian Process Filtering in GenJAX

## Abstract

We present a purely functional implementation of the Gershman (2024)
habituation model, in which an organism's entire lifetime of
perception, learning, and behavioral response is expressed as a
stateless fold over a sequence of stimuli. The model treats
habituation — the progressive decline in response to a repeated
stimulus — as Bayes-optimal filtering of a latent Gaussian process.
Our implementation in JAX and GenJAX eliminates all mutable state:
the organism's memory is encoded in a masked kernel matrix, inference
is a Cholesky decomposition within a `lax.scan` body, and
multi-condition experiments are parallelized via `vmap`. We show that
nine classical habituation phenomena emerge from this single
functional pipeline without any special-cased behavioral mechanisms.

---

## 1. Introduction

Habituation is the simplest form of learning: an organism presented
with a repeated, inconsequential stimulus gradually stops responding
to it. Despite its simplicity, habituation exhibits a rich set of
empirical regularities — spontaneous recovery after rest, stimulus
specificity, potentiation of subsequent learning, dishabituation by
novel stimuli — that have historically been described by separate
mechanistic accounts (Thompson & Spencer, 1966; Rankin et al., 2009).

Gershman (2024) proposed a unifying framework: habituation as
Bayes-optimal filtering. The organism maintains a Gaussian process
(GP) belief over a latent world state and responds when its posterior
estimate exceeds a threat threshold. All classical habituation
phenomena emerge as consequences of a single inference algorithm
operating on a single generative model.

The original implementation used Python with scikit-learn's
`GaussianProcessRegressor`, an object-oriented, stateful API. Each
timestep mutated a GP object via `.fit()`, accumulated results into a
Python list via `.append()`, and hid the posterior equations behind
an opaque interface.

We re-implement this model in a purely functional style using JAX
and GenJAX. The key insight is that the organism's lifetime can be
expressed as:

```
(prior, stimuli) ──> fold ──> (responses, beliefs)
```

with no mutable state anywhere in the pipeline. This is not merely
an aesthetic choice: the functional form makes the mathematical
structure of the model transparent, enables hardware acceleration
via XLA compilation, and permits trivially parallel experimental
conditions via `vmap`.

### 1.1 Contributions

1. A `lax.scan`-based implementation of the GP filtering organism,
   replacing imperative loops and mutable state with a pure fold.
2. A `vmap`-based experimental harness that evaluates multiple
   stimulus conditions in parallel without Python-level iteration.
3. A GenJAX generative model that makes the probabilistic structure
   explicit, validated against the analytical posterior via
   importance sampling.
4. Reproduction of all nine figures from Gershman (2024),
   demonstrating that the functional implementation produces
   identical behavioral phenomena.

---

## 2. Methods

### 2.1 The Generative Model

The organism assumes the world is governed by a latent function
$\bar{x}(z)$ drawn from a Gaussian process with zero mean and
squared-exponential kernel:

```
Prior:        x_bar ~ GP(0, k)
Kernel:       k(z, z') = exp(-||z - z'||^2 / (2 * lambda^2))
Observations: h_t ~ N(x_bar(z_t), alpha)
Response:     y_t = Phi((x_hat_t - psi) / sigma_t)
```

where $z_t$ is the input location (time, or time + stimulus
identity), $h_t$ is the observed signal, $\alpha$ is observation
noise, $\psi$ is a response threshold, and $\Phi$ is the standard
normal CDF. The posterior mean $\hat{x}_t$ and variance
$\sigma_t^2$ are computed via the standard GP equations:

```
x_hat = k_*^T (K + alpha * I)^{-1} h
sigma^2 = k_{**} - k_*^T (K + alpha * I)^{-1} k_*
```

Throughout, we use the paper's default parameters: $\alpha = 0.3$,
$\lambda = 1.0$, $\psi = 0.5$.

### 2.2 Architecture: The Organism as a Pure Function

The entire system is composed from pure functions with no side
effects. The architecture has three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     LAYER 3: EXPERIMENTS                     │
│                                                              │
│  sim_common_test    sim_spontaneous_recovery    sim_dishab   │
│       │                     │                       │        │
│       └────── jax.vmap ─────┘                       │        │
│              over test conditions          vectorized build   │
│                     │                           │            │
├─────────────────────┼───────────────────────────┼────────────┤
│                     LAYER 2: ORGANISM LIFETIME               │
│                                                              │
│              compute_habituation_response                     │
│                         │                                    │
│                   jax.lax.scan                                │
│                    over stimuli                               │
│                         │                                    │
├─────────────────────────┼────────────────────────────────────┤
│                     LAYER 1: SINGLE-STEP INFERENCE           │
│                                                              │
│  sq_exp_kernel ──> gp_posterior ──> response                  │
│       │                 │               │                    │
│    (T,T) K        Cholesky solve    Phi(z-score)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1** contains the pure mathematical building blocks:

- `sq_exp_kernel(z1, z2, lambda)` — computes the covariance matrix.
  Pure function: two arrays in, one array out.
- `gp_posterior(z_obs, h_obs, z_test, alpha, lambda)` — computes
  posterior mean and variance via Cholesky decomposition. Pure
  function: arrays and scalars in, two arrays out.
- `response(mean, var, psi)` — applies the normal CDF decision rule.
  Pure function: arrays in, array out.

**Layer 2** is the organism's lifetime, expressed as a `lax.scan`:

```
stimuli:   [ s_0,  s_1,  s_2, ..., s_{T-1} ]
               │     │     │          │
               ▼     ▼     ▼          ▼
scan:      [ step, step, step, ..., step   ]
               │     │     │          │
               ▼     ▼     ▼          ▼
outputs:   [ r_0,  r_1,  r_2, ..., r_{T-1} ]
```

**Layer 3** contains the experimental conditions, parallelized via
`vmap` where iterations are independent.

### 2.3 The Scan Body: One Timestep of the Organism

At timestep $t$, the organism has experienced stimuli $0$ through
$t$. The scan body computes the posterior given this experience and
produces a behavioral response. The full T×T kernel matrix is
computed once before the scan; at each step, a diagonal mask renders
future observations uninformative:

```
         Kernel matrix at step t=2 (T=5)
         ┌                               ┐
         │  k00+a   k01    k02  │  k03    k04   │
         │  k10    k11+a   k12  │  k13    k14   │
         │  k20     k21   k22+a │  k23    k24   │
         │─────────────────────-┼────────────────│
         │  k30     k31    k32  │ k33+BIG  k34   │
         │  k40     k41    k42  │  k43   k44+BIG │
         └                               ┘
               observed (0..t)      masked (t+1..T-1)

         BIG = 1e10 (makes masked entries uninformative)
         a   = alpha + jitter
```

The masked observations have their signal values zeroed out and
their diagonal entries inflated to $10^{10}$, making them
effectively independent with infinite variance. The Cholesky
decomposition of this padded matrix yields a posterior that is
numerically equivalent to solving the smaller $(t+1) \times (t+1)$
system, but with a fixed matrix shape that JAX can compile once.

The scan body in full:

```
step : (carry=None, t: int) -> (carry=None, (r_t, mu_t, sigma_t))

  1.  mask      = where(indices <= t, 0, 1e10)
  2.  K_noisy   = K_base + diag(mask)
  3.  h_masked  = where(indices <= t, h_all, 0)
  4.  k_star    = K_full[:, t]              # cross-covariance
  5.  L         = cholesky(K_noisy)         # O(T^3)
  6.  alpha_vec = cho_solve(L, h_masked)    # posterior weights
  7.  v         = solve_triangular(L, k_star)
  8.  mu_t      = k_star^T @ alpha_vec      # posterior mean
  9.  var_t     = k_tt - sum(v^2)           # posterior variance
  10. sigma_t   = sqrt(var_t)
  11. r_t       = Phi((mu_t - psi) / sigma_t)  # response

  return None, (r_t, mu_t, sigma_t)
```

Note that the carry is `None` — no mutable state is threaded
between timesteps. The organism's "memory" is implicit: at step $t$,
the mask reveals the first $t+1$ rows of the pre-computed kernel
matrix. The accumulating experience is encoded in which entries are
unmasked, not in a mutating state variable.

### 2.4 Parallel Experiments via vmap

Several experiments test the organism under varying conditions
(different test intervals, recovery delays, stimulus distances).
Since each condition is independent, we replace Python loops with
`jax.vmap`:

```
                     PYTHON LOOP (before)
                     ────────────────────
                     for ti in test_intervals:
                         build inputs
                         run GP posterior
                         collect response
                     stack results

                     VMAP (after)
                     ────────────────────

  test_intervals:  [ t_0,   t_1,   t_2,  ...,  t_N  ]
                      │      │      │            │
                      ▼      ▼      ▼            ▼
  vmap(test_fn):  [ fn_0,  fn_1,  fn_2, ...,  fn_N  ]  (parallel)
                      │      │      │            │
                      ▼      ▼      ▼            ▼
  responses:      [ r_0,   r_1,   r_2,  ...,  r_N   ]
```

Each `test_fn` closes over the fixed habituation data and computes
a single GP posterior at one test point. The `vmap` transforms this
into a batched operation — no Python-level iteration, no mutable
accumulator.

This pattern is used in three simulations:

| Simulation               | Mapped over          | N conditions |
|--------------------------|----------------------|--------------|
| Common test (Fig 4)      | Test intervals       | 10           |
| Spontaneous recovery (5) | Recovery delays      | 12           |
| Stimulus specificity (7) | Stimulus distances   | 11           |

### 2.5 The GenJAX Generative Model

In addition to the analytical filter, we provide a GenJAX
probabilistic program that makes the generative model explicit:

```
@gen
habituation_gp_model(z_inputs, alpha, length_scale):
    │
    ├── K = sq_exp_kernel(z_inputs, z_inputs, length_scale)
    │
    ├── x_bar ~ MVN(0, K)                   @ "x_bar"
    │     │
    │     └── latent GP function values
    │
    ├── obs ~ MVN(x_bar, alpha * I)          @ "obs"
    │     │
    │     └── noisy observations
    │
    └── return x_bar
```

This is a two-level hierarchical model: the prior draws latent
function values from a multivariate normal (the finite-dimensional
projection of a GP), and the likelihood generates observations as
noisy readings of those values. The `@` operator tags random
choices with addresses, enabling GenJAX's inference machinery to
constrain observations and compute importance weights.

Inference proceeds by constraining `"obs"` to the observed data and
drawing importance samples:

```
  key ──> seed(model.generate)(key, {"obs": h_obs}, args)
              │
              ├── trace    (contains x_bar sample)
              └── log_w    (importance weight)

  Repeat N times via vmap ──> weighted posterior statistics
```

### 2.6 Data Flow: End-to-End

The complete data flow from parameters to figures, with no mutable
state at any stage:

```
 PARAMETERS                  STIMULI                   OUTPUTS
 ──────────                  ───────                   ───────
 alpha=0.3                z_all: (T, D)
 lambda=1.0    ──────┐    h_all: (T,)
 psi=0.5              │       │
                      ▼       ▼
              ┌───────────────────────┐
              │   sq_exp_kernel       │
              │   K_full = k(z, z)    │──── (T, T) kernel matrix
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   lax.scan            │
              │   ┌─────────────┐     │
              │   │  mask K     │     │
              │   │  cholesky   │     │
              │   │  posterior  │ x T │──── responses: (T,)
              │   │  response   │     │     means:     (T,)
              │   └─────────────┘     │     stds:      (T,)
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   normalize_response  │──── normalized: (T,)
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   matplotlib          │──── figure (side effect,
              │   (I/O boundary)      │     at the boundary only)
              └───────────────────────┘
```

Side effects (file I/O, plotting) exist only at the outermost
boundary. Everything from parameters to normalized responses is a
pure function composition.

---

## 3. Results

### 3.1 Behavioral Phenomena

The purely functional organism reproduces all nine classical
habituation phenomena from Gershman (2024):

**Figure 2 — Simple habituation.** Repeated presentation of a
stimulus at fixed intensity produces a monotonically declining
response. The posterior mean approaches the stimulus value from
below (since the prior mean is zero), while posterior variance
decreases, together driving the response below the CDF threshold.

**Figure 3 — Frequency and intensity effects.** A 2×2 design
(low/high frequency × low/high intensity) shows that high-frequency,
low-intensity stimulation produces the deepest habituation, while
high-intensity stimulation at any frequency produces sensitization
(increasing response). This is because high-intensity signals
($h > \psi$) push the posterior mean above the threshold.

**Figure 4 — Common test procedure.** After habituation at low or
high frequency, a test stimulus at varying intervals shows that
high-frequency habituation produces stronger test responses
(paradoxically), because the tightly-packed stimuli create a
posterior that generalizes less to the test time. The low-frequency
condition spreads observations across a wider temporal range,
creating broader generalization.

**Figure 5 — Spontaneous recovery.** After habituation, responses
recover as the test delay increases. The GP posterior reverts toward
the prior mean (zero) at locations far from the training data, so
the posterior variance increases and the response grows. Longer
habituation series produce slower recovery because more data anchors
the posterior.

**Figure 6 — Potentiation.** A second habituation series, presented
after a recovery delay, habituates faster than the first. The
recovery delay is long enough for the posterior mean to revert
toward baseline, but the accumulated observations from the first
series keep the posterior variance low. Lower variance means the
CDF threshold is crossed sooner.

**Figure 7 — Stimulus specificity.** After habituating to stimulus
$s = 0$, test stimuli at increasing distances $d$ in stimulus space
produce graded recovery. The squared-exponential kernel's
generalization in the stimulus dimension means that nearby stimuli
share posterior information, while distant stimuli are effectively
novel.

**Figure 8 — Dishabituation.** A strong novel stimulus ($s = 1.0$,
$h = 0.7$) inserted after habituation partially restores the
response to the familiar stimulus, more than a weak novel stimulus
does. Repeated interleaving of novel and familiar stimuli eventually
habituates the dishabituation effect, as accumulated familiar-test
data outweighs the novel stimuli.

**Figures 9–10 — Length-scale effects.** A very short length-scale
($\lambda = 0.001$) eliminates temporal generalization: each
stimulus is treated independently, so there is no habituation. A
very long length-scale ($\lambda = 100$) makes the GP nearly
constant, so all observations contribute equally regardless of
temporal distance.

### 3.2 Validation: GenJAX vs Analytical Posterior

Importance sampling with 5000 samples from the GenJAX generative
model produces posterior statistics that match the analytical GP
posterior:

```
  Metric            Value
  ──────            ─────
  Max |mean error|  0.009
  Max |var error|   0.005
  Threshold         0.050
  Result            PASS
```

This confirms that the GenJAX model and the analytical filter
implement the same generative process.

---

## 4. Discussion

### 4.1 The Organism Has No State

The most striking property of the implementation is the absence of
mutable state. The `lax.scan` carry is `None`. There is no
"memory buffer" that gets updated, no "learning rate" that
accumulates, no object whose fields are mutated. The organism's
memory is entirely structural: it is the set of unmasked entries in
the pre-computed kernel matrix.

This means the organism at timestep $t$ can be fully reconstructed
from two things: the fixed kernel matrix $K$ (computed from the
stimulus schedule), and the integer $t$ (which determines the mask).
There is no hidden state that could drift, corrupt, or leak across
experimental conditions.

Compare with the original sklearn implementation:

```
 SKLEARN (imperative)              JAX (functional)
 ────────────────────              ─────────────────

 gp = GaussianProcess()           K = sq_exp_kernel(z, z)
 responses = []                   scan(step, None, range(T))
 for t in range(T):                 │
   gp.fit(z[:t+1], h[:t+1])  ←── mutation    step(_, t):
   mu, s = gp.predict(z[t])        │           mask = where(i<=t,...)
   responses.append(cdf(...)) ←── mutation     L = cholesky(K+mask)
                                    │           r = cdf(posterior)
 return responses                   │           return None, r
                                    └── no mutation anywhere
```

The left column has two kinds of mutation: the GP object's internal
state (training data, cached decomposition) and the response list.
The right column has none.

### 4.2 Why the Functional Form Matters

Beyond aesthetics, the purely functional structure provides concrete
benefits:

**Composability.** Because each layer is a pure function, layers
compose without defensive copying or state management. The
`vmap`-based experiment layer simply calls the scan-based organism
layer on different inputs. No need to "reset" the organism between
conditions — there is nothing to reset.

**Parallelism.** `vmap` over test conditions is only possible
because the test function is pure. If it mutated shared state (as
in the sklearn version), parallelization would require explicit
synchronization. Here, `vmap` is a one-line transformation.

**Reproducibility.** The organism's behavior is a deterministic
function of its inputs. There is no dependence on call order,
global state, or object identity. The same `(K, t)` pair always
produces the same `(r, mu, sigma)` triple.

**Transparency.** The scan body is an 11-line function that maps
directly onto the mathematical equations. There is no framework API
between the user and the linear algebra. The Cholesky decomposition,
the solve, and the CDF are all visible in the body.

### 4.3 What the Organism *Is*

Viewed through the functional lens, the organism is a remarkably
simple object:

```
  organism : (kernel, threshold, noise) -> (stimuli -> responses)
```

It is a *curried function*. Given its "biology" (kernel parameters
$\lambda$, threshold $\psi$, noise $\alpha$), it returns a function
from stimulus sequences to response sequences. The kernel encodes
the organism's inductive bias about temporal and stimulus-space
similarity. The threshold encodes what counts as "worth responding
to." The noise parameter encodes how reliable the organism considers
its own sensory signals.

The nine classical habituation phenomena are not nine mechanisms.
They are nine input-output pairs of a single function:

```
  organism(biology)(repeated_stimulus)     = habituation
  organism(biology)(stimulus + long_pause) = spontaneous_recovery
  organism(biology)(stim_A then stim_B)    = stimulus_specificity
  organism(biology)(two_series)            = potentiation
  ...
```

The explanatory power comes from the *shape of the kernel* and the
*logic of Bayesian inference*, not from any stimulus-specific
mechanism.

### 4.4 The Role of the Generative Model

The GenJAX generative model serves a dual purpose. First, it makes
the probabilistic assumptions explicit in executable code: the GP
prior over latent states and the Gaussian observation model are
written as sampling statements, not implicit in a matrix equation.
Second, it enables inference algorithms beyond the analytical
posterior — importance sampling here, but potentially MCMC or SMC
for extensions with non-Gaussian likelihoods.

The validation (Section 3.2) confirms that the generative model and
the analytical filter agree, establishing that the `@gen` function
is a faithful representation of the same organism.

### 4.5 Limitations and Extensions

The current implementation computes a full $T \times T$ Cholesky
decomposition at every timestep of the scan. While the masking
trick maintains a fixed matrix shape (enabling JIT compilation), it
means the computational cost is $O(T^4)$ rather than the
$O(T^3 \cdot (T+1)/2)$ of the growing-matrix approach. For the
small $T$ values in the paper ($T \leq 60$), this is negligible.
For larger sequences, an incremental Cholesky update within the
scan carry would be more efficient — and would constitute a
meaningful use of the carry parameter.

The model could be extended to non-stationary kernels (modeling
sensitization as a separate process), hierarchical priors over
kernel parameters (learning the length-scale from experience), or
multi-output GPs (modeling multiple response modalities). Each
extension would preserve the functional structure: the scan body
would grow, but the `(prior, stimuli) -> responses` signature
would remain.

---

## 5. Conclusion

We have shown that an organism capable of habituation — the
simplest form of learning — can be implemented as a pure functional
fold over a stimulus sequence, with no mutable state, no
object-oriented scaffolding, and no imperative control flow. The
entire behavioral repertoire (nine classical phenomena) emerges from
composing three pure functions: a kernel, a Bayesian posterior, and
a threshold decision rule. The JAX ecosystem's `lax.scan` and `vmap`
provide the natural vocabulary for expressing this composition,
while GenJAX makes the underlying generative model explicit.

The result is a "minimal organism" in both the biological and
computational senses: biologically, it is the simplest Bayesian
agent that explains habituation; computationally, it is the
simplest functional program that implements that agent.

---

## References

- Gershman, S. J. (2024). Habituation as optimal filtering.
  *iScience*, 27, 110523.
- Thompson, R. F., & Spencer, W. A. (1966). Habituation: A model
  phenomenon for the study of neuronal substrates of behavior.
  *Psychological Review*, 73(1), 16–43.
- Rankin, C. H., et al. (2009). Habituation revisited: An updated
  and revised description of the behavioral characteristics of
  habituation. *Neurobiology of Learning and Memory*, 92(2),
  135–138.

---

## Appendix A: File Structure

```
habituation/
├── core.py           Layer 1-2: kernel, posterior, response, scan, GenJAX model
├── simulations.py    Layer 3: all 9 experiments (vmap + vectorized construction)
├── figs.py           I/O boundary: matplotlib rendering
├── main.py           CLI entry point: --all, --fig N, --validate
└── figs/             Generated figures (PNG)
```

## Appendix B: Running the Code

```bash
# Generate all figures
python -m habituation.main --all

# Generate a specific figure
python -m habituation.main --fig 3

# Validate GenJAX IS vs analytical posterior
python -m habituation.main --validate
```
