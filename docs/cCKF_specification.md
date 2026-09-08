# cCKF Master Specification

**Status:** normative. **Owner:** Matthew Musson. **Last revised:** 2026-07-28.
**Deliverable:** poster, 2026-08-28 (hard).

---

## 0. How to use this document

This is the single source of truth for definitions, conventions, and locked decisions
in the cCKF project. It exists so that agent sessions do not re-derive, re-litigate, or
silently re-invent choices that have already been made.

Rules for agents working in this repo:

- Sections marked **[LOCKED]** MUST NOT be changed without an explicit instruction from
  Matthew and a corresponding entry in §16 (Changelog).
- Sections marked **[OPEN]** are undecided. Do not resolve them unilaterally. Surface the
  question and stop.
- Where this document and `CLAUDE.md` conflict, `CLAUDE.md` governs process (experiment
  discipline, tier system) and this document governs technical content.
- Where this document and `Paper_Experiment_Log.tex` conflict, **this document is correct**
  and the log needs a fix.
- Anything computed from truth information is a *label*, never a *feature*. If an agent
  finds itself adding a truth-derived quantity to a feature vector, stop and flag it.

---

## 1. Project summary

**cCKF** (calibrated Combinatorial Kalman Filter) replaces three hand-tuned heuristics in
the ACTS CKF with calibrated learned components, while leaving the Kalman filter and the
ACTS propagator (adaptive RKN4, material integration, covariance transport) frozen as
exact physics.

| Component | Replaces | Object learned |
|---|---|---|
| Gate $g_\psi$ | the $\chi^2$ hit-acceptance cut | $P(\text{same particle} \mid x)$ |
| Value $V_\phi$ | branch cap / hole cap / sequential-hole cap | $P(\text{branch completes to a matched track})$ |
| Score $q_\omega$ + ILP | greedy ambiguity resolution | weighted set-packing over track candidates |

**Thesis.** Every heuristic threshold in the CKF is a broken Neyman–Pearson test. The
$\chi^2$ cut is NP-optimal *only if* (a) the residual is Gaussian with correctly propagated
$S_k$ and (b) background is uniform in the window. Both fail — (a) through material and
field-map error, (b) because occupancy varies by orders of magnitude between quiet regions
and jet cores. The fix is a calibrated supervised density, not a retuned constant.

**Benchmark.** ColliderML, $t\bar{t}$, $\mu = 200$ pileup, edm4hep, ODD geometry, full
Geant4. Framework: the **cmuchancel fork of ACTS**.

**Headline result target.** Stratified reliability diagrams for $g_\psi$ versus the
$\chi^2$-implied probability under the Gaussian model (figure G3). The calibration gap is
the contribution; end-to-end efficiency improvement is secondary.

---

## 2. Notation conventions [LOCKED]

One collision has already caused confusion; it is resolved as follows.

| Symbol | Meaning | Notes |
|---|---|---|
| $\rho^\pi(s)$, $\rho_{\text{train}}$, $\rho_{\text{deploy}}$ | state-visitation distribution | RL convention. **Reserved.** |
| $\kappa_u, \kappa_v$ | normalized cluster size, $s_u/\langle s_u\rangle$ | **Do not write $\rho_u$.** |
| $\tilde Q$ | normalized cluster charge (MIP units) | §8.2 |
| $x$ | gate feature vector | ~20–25 dim, §8 |
| $z_\psi(x)$ | raw gate logit | pre-calibration |
| $g_\psi = \sigma(z_\psi)$ | gate probability | post-calibration unless stated |
| $\pi$ (scalar) | base rate $P(y{=}1)$ | context disambiguates from policy $\pi$ |
| $\pi^\dagger$ | truth-greedy in-MDP policy | §11.1 |
| $S_k$ | innovation covariance | $H_k P_k^- H_k^\top + R_k$ — **includes $R_k$** |
| $n$ | window multiplier | environment constant, §5.2 |
| $m$ | branch majority particle | frozen at seed formation |

---

## 3. Metric conventions [LOCKED]

Fixed once, implemented in a single metric module, never re-derived per plot.

**Double-majority (DM) matching.** A reconstructed track $T$ matches particle $p$ iff

```
    shared(T, p) / |hits on T|  >= 0.5     (purity)
AND shared(T, p) / |hits of p|  >= 0.5     (completeness)
```

Both conditions. ACTS's default is single-majority; we use DM throughout. Any number
reported without the DM qualifier is a bug.

- $\varepsilon_{\text{DM}}$ — reconstructable particles DM-matched by some track / all reconstructable particles.
- $f_{\text{DM}}$ — reconstructed tracks with **no** DM match / all reconstructed tracks. This is
  `fakeratio_tracks`. **Not** `fakeratio_particles` (fraction of true particles spawning a
  fake), which invalidated the first Pareto front. Assert the definition in code.
- $d_{\text{DM}}$ — extra tracks DM-matched to an already-matched particle / all tracks.
- $t_{\text{seed}\to\text{trk}}$ — ACTS profiler per-event thread time, SpacePointMaker through
  GreedyAmbiguityResolution, summing `time_perevent_s`.

**Efficiency denominator [OPEN — resolve with Rocky/Louie].** Particles with $p_T > 1$ GeV,
$|\eta| < \eta_{\max}^{\text{ODD}}$, $\geq N_{\min}$ measurements. $N_{\min}$ and
$\eta_{\max}^{\text{ODD}}$ per ODD convention — still unresolved. Do not hard-code a guess;
read from a config with an explicit `TODO` and fail loudly if unset.

**Factorized efficiency.** Report $\varepsilon_{\text{total}} = \varepsilon_{\text{seed}} \cdot
\varepsilon_{\text{follow}} \cdot \varepsilon_{\text{select}}$ so losses are attributable.

**Aggregation.** Unweighted mean $\pm$ sample stdev (ddof=1) of *per-event* rates. Do not
pool hits across events and take a global ratio; the per-event spread is the uncertainty.

---

## 4. Baseline and operating points [LOCKED]

Established by joint 10-dimensional MO-TPE optimization (Optuna, `MOTPESampler` — **not**
`NSGAIISampler`), 500 trials, 32 optimization events, evaluated on events $[32, 64)$.

### 4.1 ACTS fair-box baseline

Eval events $[32,64)$, $N=32$, post-ambiguity DM.

| Metric | Mean $\pm \sigma$ |
|---|---|
| $\varepsilon_{\text{DM}}$ | $40.98 \pm 2.26$ % |
| $f_{\text{DM}}$ | $1.273 \pm 0.700$ % |
| wall | $28.7 \pm 5.7$ s/event |

Config: seeds/spm 5, min$p_T$ 0.4 GeV, impactMax 5.0 mm (clipped from the true ACTS default
of 20 mm to match the search box), $\sigma_{\text{scat}}$ 5.0, $\chi^2_{\text{meas}}$ 15,
$\chi^2_{\text{out}}$ 25, branch 2, $n^{\min}_{\text{meas}}$ 7, holes/out 2/2 (separate cuts),
ptMin 1.0 GeV.

> **[OPEN]** The log's parameter section states the ACTS default `numMeasurementsCutOff`
> is 1, but the fair-box table uses 2. Resolve and make the two consistent. Also: the
> `loc0 ±4mm` cut identified as a cause of low baseline efficiency must be confirmed absent
> from all reported runs.

### 4.2 Operating points

| Point | Trial | $\varepsilon_{\text{DM}}$ (%) | $f_{\text{DM}}$ (%) | $t_{\text{seed}\to\text{trk}}$ (s/evt) |
|---|---|---|---|---|
| Tight | 79 | $92.34 \pm 1.03$ | $0.121 \pm 0.108$ | 24.92 |
| Medium | 70 | $96.23 \pm 0.74$ | $0.773 \pm 0.362$ | 48.42 |
| Fast | 331 | $93.14 \pm 0.93$ | $0.163 \pm 0.182$ | 9.81 |

| Point | seeds/spm | min$p_T$ | impactMax | $\sigma_{\text{scat}}$ | $\chi^2_{\text{meas}}$ | $\chi^2_{\text{out}}$ | branch | $n^{\min}_{\text{meas}}$ | holes+out | ptMin |
|---|---|---|---|---|---|---|---|---|---|---|
| Tight | 16 | 0.688 | 2.50 | 2.34 | 16.26 | 20.66 | 3 | 9 | 1 | 0.460 |
| Medium | 46 | 0.587 | 2.86 | 2.95 | 12.04 | 35.75 | 2 | 7 | 1 | 0.594 |
| Fast | 331→25 | 0.671 | 0.58 | 2.15 | 15.40 | 31.87 | 5 | 8 | 1 | 0.622 |

**Role of the three points [LOCKED].** They are **not** three data-collection configs. They
are the **calibration-transfer test set**: train $g_\psi$ once at the envelope config (§6.2),
calibrate on-policy, then report ECE when deployed at each of Tight / Medium / Fast. That
converts an arbitrary choice into a generalization claim.

---

## 5. The label-generating environment [LOCKED]

All labels are defined by this MDP. Its configuration is hashed into `env_config_hash` and
**rows with mismatched hashes MUST NOT be pooled in training.**

### 5.1 MDP

- **State** $\tau_{0:k}$: partial track with filtered parameters and covariance at the current surface.
- **Actions**: every hit inside the window $\mathcal{W}_k(n)$, plus `hole`.
- **Transition** (deterministic): Kalman update on accept; propagate-without-update on
  `hole`, which *widens* the window at subsequent surfaces — so a particle lost after a hard
  scatter can be recaptured later, and targets must allow this.
- **No hole caps or branch caps exist in the label-generating environment.** (Tractability
  caps during collection are a separate, logged concern — §6.4.)

### 5.2 The window

$$
S_k = H_k P_k^- H_k^\top + R_k
$$

**$R_k$ is included.** Omitting it collapses the window below sensor resolution on
well-constrained tracks and loses true hits to measurement noise alone.

$$
\mathcal{W}_k(n) = \left\{ h : |r_0| \le n\sqrt{S_{00}},\ |r_1| \le n\sqrt{S_{11}} \right\},
\qquad r = h - \hat h
$$

The axis-aligned box with half-widths $n\sqrt{S_{ii}}$ is exactly the **bounding box** of the
Mahalanobis ellipse $\{r^\top S_k^{-1} r \le n^2\}$, independent of the off-diagonal
correlation. It is therefore a strict superset, which is the point: training support must
contain every acceptance region the gate could implement. State this in any writeup or a
reader will assume the correlation was ignored.

$n$ is a **normative environment constant**, not a tuned hyperparameter. It defines the
action space, hence the label distribution, hence the base rate. See §6.3 for the collection
value.

---

## 6. Data collection — the rollout [ACTIVE TASK]

This is the immediate work item. Everything in this section is either irreversible or
expensive to redo.

### 6.1 Split discipline [LOCKED — highest-severity trap]

The CKF baseline and all three operating points were evaluated on events **$[32, 64)$**.
If gate training data is drawn from those events, the held-out set is burned and every
cCKF-vs-CKF comparison becomes dishonest.

```
Events [0, 32)   -> gate train / val / calibration    (~24 / 4 / 4)
Events [32, 64)  -> held out. Shared CKF + cCKF test set. Opened ONCE.
```

- Freeze the split by event ID, in code, before the run. Never modify.
- Split **by event**, never by row. Hits within an event share pileup and detector region;
  sibling branches revisit the same hits. Row-level splitting leaks.
- The calibration split is sacred: temperature/Platt fits and calibration audits only.
  Never training, never hyperparameter selection.
- Four calibration events is thin. This is a known limitation and it is the reason §10.3
  specifies a four-parameter continuous calibrator rather than a per-stratum fit.

### 6.2 Envelope collection config

Elementwise-loosest across the three operating points, so that any tighter config's rows are
recoverable by offline filtering.

| Parameter | Value | Source |
|---|---|---|
| `maxSeedsPerSpM` | 46 | max (Medium) |
| seed `minPt` | 0.587 GeV | min (Medium) |
| `impactMax` | 2.86 mm | max (Medium) |
| `sigmaScattering` | 2.95 | max (Medium) |
| `chi2CutOffMeasurement` | 16.26 | max (Tight) |
| `chi2CutOffOutlier` | 35.75 | max (Medium) |
| `numMeasurementsCutOff` | 5 | max (Fast) |
| `nMeasurementMin` | **disabled** | terminal cut, §6.4 |
| `maxHolesAndOutliers` | **disabled** | terminal cut, §6.4 |
| CKF `ptMin` | **disabled** (or = seed minPt) | terminal cut, §6.4 |
| window multiplier $n$ | see §6.3 | environment constant |

**Subset-recoverability assumption — MUST VERIFY IN PILOT.**

The envelope argument rests on tighter configs producing subsets of the envelope's output.
This holds cleanly for the branch cap (top-3 candidates by $\chi^2$ on a surface are a subset
of top-5, and their descendant subtrees follow) but is only *approximate* for seeding:
loosening `minPt` from 0.688 to 0.587 admits additional triplets that compete for the
top-46 slots per middle space point, so a seed in Tight's top-16 is not *guaranteed* to
appear in the envelope's top-46.

Pilot check: run Tight standalone on 1–2 events, run the envelope on the same events, and
measure the fraction of Tight's seeds recovered by offline filtering of the envelope output.
If recovery is below ~98%, the envelope argument needs revision — report and stop.

### 6.3 Window multiplier

Collect at the **largest $n$ you would ever consider**. A tighter window is a strict subset
and can be applied offline; a larger one cannot be recovered without re-running. This is the
single most costly parameter to get wrong.

> **[OPEN]** Concrete value of $n$. Candidate: $n = 5$. Decide from the pilot's window-failure
> curve (§6.6) and row-count/storage measurement, then freeze into `env_config_hash`.

### 6.4 Terminal cuts MUST be disabled

`nMeasurementMin`, CKF `ptMin`, and `maxHolesAndOutliers` are precisely the heuristics
$V_\phi$ exists to replace. If they fire during collection, the downstream rows of exactly
the branches we want to learn to prune are destroyed.

Tractability caps that must remain (branch cap, any global branch-count limit) MUST log
every firing with the reason, so their distortion of $\rho_{\text{train}}$ is measurable
rather than invisible.

### 6.5 Logging schema

One row per (branch, layer, candidate), Parquet. Fields marked **[IRREVERSIBLE]** cannot be
reconstructed after the run.

| Field | Type | Notes |
|---|---|---|
| `event_id, seed_id, branch_id, parent_branch_id, step_k` | int | **[IRREVERSIBLE]** lineage; required for offline config filtering |
| `layer_id, surface_id` | int | geometry |
| `state` | float[6] | $(\ell_0, \ell_1, \phi, \theta, q/p, t)$ predicted |
| `cov_packed` | float[21] | lower-triangular predicted covariance |
| `pred_local` | float[2] | $\hat h$ |
| `cand_hit_id` | int | $-1$ encodes the `hole` action |
| `residual` | float[2] | $h - \hat h$, local coords |
| `chi2_inc` | float | CKF $\chi^2$ increment |
| `cluster_feats` | float[C] | $s_u, s_v, Q_{\text{tot}}, \sigma_{uu}, \sigma_{uv}, \sigma_{vv}$ — **see §6.6 check 1** |
| `incidence` | float[2] | **[IRREVERSIBLE]** local $(\alpha_u, \alpha_v)$ from predicted direction, §8.2 |
| `sensor_props` | float[3–5] | pitch$_u$, pitch$_v$, thickness, is_pixel, is_barrel |
| `occupancy_feats` | float[O] | $n_{\text{window}}$ (Mahalanobis), $n$ in fixed geometric window |
| `context_feats` | float[M] | $X/X_0$ to next surface, dead-module flag, layer embedding index |
| `branch_feats` | float[B] | $n_{\text{hits}}, n_{\text{holes}}, n_{\text{seq\_holes}}$, accumulated log-odds |
| `action_taken` | int8 | accepted / rejected / hole / branch-pruned |
| `prune_reason` | int8 | **[IRREVERSIBLE]** which heuristic or cap fired |
| `contrib_pids` | int64[] | **[IRREVERSIBLE]** full contributing-particle list for the cluster |
| `contrib_charge_frac` | float[] | **[IRREVERSIBLE]** charge fraction per contributor |
| `branch_majority_pid` | int64 | $m$; frozen at seed formation |
| `majority_true_hit_on_surface` | int8 | **[IRREVERSIBLE]** did $m$ leave a measurement here at all? §7.3 |
| `truth_residual` | float[2] | **[IRREVERSIBLE]** $\delta = h - h^{\text{true}}_m$ from SimTrackerHit, §11.3 |
| `vstar_soft` | float | $V^{\pi^\dagger}$, computed offline (§11.1) |
| `env_config_hash` | str | §5 |

**Why contributor lists rather than booleans.** Logging the full `(pid, charge_frac)` list
lets `label_same_particle` under either convention, `cluster_merged`, and
`majority_undefined` all be computed offline in any combination. A pre-computed boolean
locks in a convention still under discussion (§7.2).

**Hole rows.** A row MUST be emitted for the `hole` action on every surface, with its own
features (dead-module flag, layer efficiency prior, $X/X_0$). The gate is a
$(|C|+1)$-way decision implemented as independent logits.

### 6.6 Pilot protocol — run this before the full rollout

Run on 1–2 events. Do not launch the full rollout until all five pass.

1. **Cluster features are populated.** Cluster size, shape moments, and charge are **not**
   stored in `TrackerHitPlane` and must be explicitly logged during digitization. Verify the
   fields are non-null and non-degenerate on real hits. If the digitization hook is missing,
   the entire $\kappa_u, \kappa_v, \tilde Q$ program (§8.2) fails silently — this is the single
   highest-value check in the list.
2. **Seed-recovery fraction** for Tight under offline filtering of the envelope (§6.2).
3. **Scale**: rows/event, positive fraction, storage/event, wall time/event. Extrapolate to
   32 events before committing. Rough prior: $\sim 10^6$ rows/event, $\sim$300 B/row (the
   packed covariance alone is 84 B) $\Rightarrow$ tens of GB. Branch cap 5 plus a wide window
   can inflate this faster than expected.
4. **Window-failure rate** vs $n$: fraction of surfaces where $m$ left a measurement but it
   fell outside $\mathcal{W}_k(n)$. This drives the §6.3 decision and is a reportable figure
   in its own right (§12).
5. **Env hash** is written and stable across events.

### 6.7 Post-collection filtering

The training distribution SHOULD be filtered down to the deployment seed population before
fitting, because seeding changes the $p_T$ spectrum being followed and hence changes
$P(y{=}1 \mid x)$ through the physics, not merely the state marginal. Collect the superset;
subset at training time. This keeps the deployment-seeding decision reversible.

---

## 7. Labels [LOCKED except where marked]

### 7.1 Branch majority

$m$ = the particle contributing the largest number of measurements to the branch, **frozen at
seed-triplet formation and never updated**. Rolling recomputation produces pathological
training signal. Seeds where no particle contributes $\geq 2/3$ of the triplet hits are
excluded as ambiguous.

### 7.2 `label_same_particle`

$$
y^{\text{any}}_i = \mathbb{1}\big[m \in \text{contrib}(h_i)\big]
\qquad\text{vs.}\qquad
y^{\text{maj}}_i = \mathbb{1}\big[m = \arg\max_p Q_p(h_i)\big]
$$

These differ **only** on merged clusters.

> **[OPEN — decide before training, not before collection]** The choice is fixed by the metric
> module, not by taste. If the metric counts shared hits through the hit→particles multimap
> (ACTS default: one measurement increments every contributing particle's count), then
> accepting a merged cluster containing $m$ genuinely increments the DM numerator, and
> $y^{\text{any}}$ is the convention whose expectation the metric rewards. Training on
> $y^{\text{maj}}$ would then teach the gate to reject measurements that would have helped.
> **Action: read the metric module, determine which it does, record the answer here.**

### 7.3 Ambiguity flags — split the old flag

The original `label_ambiguous` conflated two conditions requiring opposite treatment. It is
replaced by two flags:

| Flag | Condition | Treatment |
|---|---|---|
| `majority_undefined` | branch majority not unique | **Exclude.** The target has no value; no feature fixes this. |
| `cluster_merged` | cluster is truth-merged ($\geq 2$ contributors) | **Include at full weight.** These are the hard cases, not the undefined ones. |

Excluding merged clusters deletes exactly the rows the cluster features exist to handle, and
they are concentrated in jet cores — where the stratified reliability diagram is supposed to
show the headline separation. See §13 for the consequent ablation polarity flip.

### 7.4 Holes vs window failures

Two distinct events, currently conflated in the experiment log:

- **Genuine hole**: $m$ left no measurement on this surface (gap, inefficiency).
- **Window failure**: $m$ left a measurement and $\mathcal{W}_k(n)$ missed it.

Only the second is a property of our environment, and $P(\text{window failure} \mid n)$
upper-bounds follow efficiency. `majority_true_hit_on_surface` is what separates them, and it
requires a truth lookup available only at collection time.

---

## 8. Gate features

**No feature may require simulation truth at inference time.** Truth is used only for labels.

### 8.1 Feature vector

$$
x_k = \Big(\underbrace{r_k,\ \text{chol}(S_k),\ \chi^2_k}_{\text{what }\chi^2\text{ sees}},\
\underbrace{s_u, s_v, Q_{\text{tot}}, \sigma_{uu}, \sigma_{uv}, \sigma_{vv}}_{\text{cluster}},\
\underbrace{\kappa_u, \kappa_v, \tilde Q}_{\text{normalized, §8.2}},\
\underbrace{n_{\text{window}}}_{\text{occupancy}},\
\underbrace{\eta, q/p, k, X/X_0}_{\text{context}},\
\underbrace{\text{sensor}}_{3\text{–}5},\
\underbrace{n_{\text{hits}}, n_{\text{holes}}, n_{\text{seq}}}_{\text{branch}}\Big)
$$

Feeding $S_k$ and $\chi^2_k$ as *features* rather than trusting them as *sufficient
statistics* is the design point: the network learns **when the Gaussian model is
trustworthy**.

$n_{\text{window}}$ is doing specific work. Under the Gaussian-uniform-background model,

$$
\log\Lambda(r) = -\tfrac12 r^\top S_k^{-1} r - \tfrac12\log\det(2\pi S_k) - \log\rho_{\text{bkg}}
$$

so $\chi^2$ is NP-optimal only if $\rho_{\text{bkg}}$ is constant. It is not.
$n_{\text{window}}$ is the observable proxy for $-\log\rho_{\text{bkg}}$.

### 8.2 Normalized cluster features [NEW — do not omit]

The raw cluster features alone force the network to rederive sensor geometry from scratch on
very few events. Note that $\eta$ is a *global* track quantity and differs substantially from
*local* incidence between barrel and endcap at the same $\eta$.

Local incidence from the predicted direction $\hat d$ and surface frame $(\hat u, \hat v, \hat n)$:

$$
\alpha_u = \arctan\frac{\hat d \cdot \hat u}{\hat d \cdot \hat n}, \qquad
\alpha_v = \arctan\frac{\hat d \cdot \hat v}{\hat d \cdot \hat n}
$$

Single-particle expectations for a sensor of thickness $t$:

$$
\langle s_u\rangle \approx \frac{t\tan\alpha_u}{\text{pitch}_u} + 1, \qquad
\langle Q_{\text{tot}}\rangle \propto \frac{t}{\cos\alpha}\cdot\frac{dE}{dx}\big(\beta(q/p)\big)
$$

Normalized features:

$$
\kappa_u = \frac{s_u}{\langle s_u\rangle}, \qquad
\kappa_v = \frac{s_v}{\langle s_v\rangle}, \qquad
\tilde Q = \frac{Q_{\text{tot}}\cos\alpha}{t\,\langle dE/dx\rangle_{\text{MIP}}}
$$

A merged cluster is then $\kappa \gg 1$ with $\tilde Q \gtrsim 2$ MIP — a one-dimensional
signal instead of a learned six-way interaction. Same treatment for second moments: pass
$\sigma_{uu}/\langle\sigma_{uu}\rangle$ or the residual after subtracting the predicted
straight-track projection. **Keep the raw features too**; the ratios buy sample efficiency,
which is the binding constraint at 24 training events.

These features also earn their keep off merged rows: a background hit crossing at a different
angle has $\kappa_u$ inconsistent with the branch's prediction even with a small positional
residual — exactly the case where $\chi^2$ is blind.

**Per-cell charge grid** (raw ADC on a padded pixel grid) is the better long-run object but
puts the full trigonometry back on the network. Moments-plus-normalization first; grid as an
ablation only.

### 8.3 Normalization

Per-feature standardization computed on the training split; store $(\mu_j, \sigma_j)$ and
apply identically to val/cal/test. Recompute after each DAgger iteration. Do not standardize
binary or small-integer features.

---

## 9. Gate training

### 9.1 Architecture

MLP, input ~20–25 → 128 → 128 → (128, ablation) → 1, SiLU, $\sim 5\times10^4$ params, raw
logit output. AdamW, lr $10^{-3}$ cosine → $10^{-5}$, weight decay $10^{-2}$, batch 4096–16384,
grad clip 1.0, early stopping patience 5 on val BCE. Trains in minutes; data generation is
the bottleneck. ONNX export for the C++ path.

### 9.2 Loss — **no positive reweighting in the primary** [REVISED]

$$
\mathcal{L} = \frac{1}{N}\sum_i \Big[ y_i\,\zeta(-z_i) + (1-y_i)\,\zeta(z_i) \Big],
\qquad \zeta(u) = \log(1+e^u)
$$

Implement $\zeta$ as $\max(u,0) + \log(1 + e^{-|u|})$ for stability. Never take
`log(sigmoid(·))` directly.

**Why no reweighting.** Weighted BCE with weight $w_+$ on positives has population minimizer

$$
\sigma^\star = \frac{w_+ p}{w_+ p + (1-p)}
\qquad\Longrightarrow\qquad
z^\star(x) = \operatorname{logit} p(x) + \log w_+
$$

a **constant translation** of the logit. With $\sim$10% positives and $w_+ = N_-/N_+$ this is
$\log 9 \approx 2.2$ nats. Three consequences:

1. Temperature scaling ($z \mapsto z/T$, fixed point pinned at $z=0$) is a *dilation* and
   cannot remove a translation. It lands on a compromise $T$ that crushes confident
   predictions toward $\tfrac12$ without fixing ECE — and spuriously trips the
   "if $T > 2$, debug the network" tripwire in the old recipe.
2. Reweighting converts a known constant into a parameter that must be estimated away, at a
   calibration sample size of four events.
3. The imbalance is **conditional structure, not a nuisance**. The marginal positive fraction
   averages over windows with one hit ($p \approx 0.9$) and windows with twenty
   ($p \approx 0.05$). That variation *is* the signal the gate exists to capture. A global
   offset is the wrong instrument for it.

Additionally, weighted BCE has gradient $w_+(1-\sigma_i)$ on positives against $\sigma_i$ on
negatives, de-emphasizing gradations among negatives — but a high-recall gate operates at
*low* $g_{\min}$, putting the decision boundary in exactly that region.

BCE is a proper scoring rule; its minimizer is the posterior. At 10% positives, leave it
alone. Reweighting earns its keep at $10^{-3}$, not $10^{-1}$.

**Merged-cluster oversampling is likewise excluded** from the primary, for the same reason
and with the same correction structure ($z^\star = \operatorname{logit} p + \log w_{s(x)}$).
Handle merged rows by *including* them (§7.3) and *reporting* AUC/ECE on `cluster_merged` as a
named subgroup. If the gate underperforms there specifically, that is the moment to reach for
targeted oversampling — with evidence, and with the $b_s$ correction to remove it.

### 9.3 The reweighting diagnostic [pre-registered]

In the infinite-capacity limit, weighted and unweighted logits differ by a strictly
increasing map, so **AUC is identical by construction**. Any observed difference is pure
misspecification/optimization effect. Free, clean probe — both configs train in minutes.

- $\text{AUC}_w \approx \text{AUC}_1$ **and** fitted Platt intercept $b \approx -\log w_+$
  → effectively well-specified in the relevant region; take unweighted on parsimony.
- Otherwise → capacity allocation matters, and the direction tells you where to spend it.

Report the $(b,\ -\log w_+)$ comparison as the check that this whole account is right.

### 9.4 Structural note: exclusivity

Per-hit BCE yields *marginals*, so $\sum_i g_\psi(x_i)$ over a surface can exceed 1 even
though at most one candidate is the true hit. For accept/reject thresholding this is correct —
it is the NP test per candidate, which is what we are replacing. For MCTS priors,
$\text{softmax}(g)$ silently converts marginals into a distribution. The alternative (masked
softmax over $\{c_1,\dots,c_m,\text{hole}\}$ with categorical CE) respects exclusivity but
destroys the per-candidate calibration story. **Our choice is the marginal + threshold**;
state it as a design point in any writeup, not an oversight.

### 9.5 Expected values

| Metric | Expected | Warning |
|---|---|---|
| Train BCE | 0.02–0.08 | > 0.15 ⇒ label noise / feature bug / normalization error |
| Val BCE | 0.03–0.10 | |
| AUC-ROC | 0.97–0.99 | < 0.95 ⇒ debug before proceeding |
| AUC-PR | 0.85–0.95 | more informative at this imbalance |

**Report AUC-PR beside AUC-ROC.** ROC is base-rate invariant, which hides operational cost
when candidates are abundant: expected spurious branches per surface is
$\approx n_{\text{window}} \cdot \text{FPR}$, so at $n_{\text{window}} \sim 20$ in a jet core,
$\text{FPR} = 1\%$ gives 0.2 spurious branches/surface compounding over ~10 layers. A ROC
that looks excellent can correspond to a branch tree you cannot afford.

---

## 10. Calibration

### 10.1 Why it is the headline

Branch score is $\sum_k \log\frac{g}{1-g}$ and thresholds $g_{\min}, v_{\min}$ are set in
probability units, so uncalibrated outputs make those numbers arbitrary. And the central claim —
that the $\chi^2$-implied probability is badly calibrated in jet cores while $g_\psi$ is not —
*is* an ECE comparison.

Note that all post-hoc calibrators (temperature, Platt, isotonic) are strictly monotone in $z$
and therefore **cannot change AUC**. Calibration relabels the probability axis; it never
improves ranking. If ranking is bad, no calibrator saves you.

### 10.2 The decomposition

$$
z_\psi(x) = \underbrace{\operatorname{logit} p(x)}_{\text{estimand}}
+ \underbrace{\log w_+}_{\text{known, global — set to 0 by §9.2}}
+ \underbrace{\epsilon(x)}_{\text{residual miscalibration}}
$$

$\epsilon$ is *whatever is left*: approximation error (finite width), estimation error (note
the *effective* $N$ is far below the row count — rows within an event share pileup, sibling
branches revisit hits), and optimization error (AdamW + early stopping + weight decay does not
land on the ERM minimizer, and networks trained past zero training error are systematically
overconfident). $\epsilon$ is a **function of $x$**, not zero-mean noise.

Temperature scaling asserts $\epsilon(x) = z_\psi(x)(1 - 1/T)$ — that the bias is a *fixed
fraction of the logit*, with one global constant of proportionality everywhere in feature
space. That is an assumption, not a fact.

### 10.3 Calibrator [LOCKED]

Fit on the calibration split only. **Four parameters, occupancy-conditional:**

$$
\hat p(x) = \sigma\big(a(x)\,z_\psi(x) + b(x)\big), \qquad
a(x) = a_0 + a_1\log n_{\text{window}}, \quad
b(x) = b_0 + b_1\log n_{\text{window}}
$$

Rationale: per-stratum Platt over 5 $\eta$ bins × 5 occupancy quintiles is $2\times25 = 50$
parameters on four correlated events — fitting noise. The continuous form encodes the specific
belief that miscalibration grows with local occupancy, because that is where the uniform-
background assumption behind $\chi^2$ fails. Add an $\eta$ term only if the residual demands it.

This also separates fitting from diagnosis: stratified reliability diagrams become the
**audit**, not the fitting mechanism. A per-stratum fit is guaranteed to look calibrated
in-sample.

Fallbacks: plain Platt $\sigma(az+b)$ if the occupancy term is not significant; temperature
$\sigma(z/T)$ only as an ablation. Isotonic is an ablation only — it needs far more calibration
data than four events.

### 10.4 Audit

$$
\widehat{\text{ECE}} = \sum_b \frac{|B_b|}{N}\left|\frac{1}{|B_b|}\sum_{i\in B_b} y_i - \frac{1}{|B_b|}\sum_{i \in B_b}\hat p_i\right|
$$

- **Use equal-mass (quantile) bins, not equal-width.** Predictions pile up near zero; equal-width
  bins leave most cells nearly empty and the estimate dominated by one or two.
- $\widehat{\text{ECE}}$ is **biased upward at small $N$**. With four calibration events across
  25 strata, some apparent miscalibration is binning noise. Report bin counts alongside ECE.
- Stratify by $\eta$ bin and occupancy quintile; coarsen to 3×3 if 5×5 cells are underpopulated,
  and say so explicitly rather than presenting a noisy 25-cell table.
- Run the identical audit on $\Lambda_{\chi^2}$ (the $\chi^2$-implied probability under the
  Gaussian model). That comparison is figure **G3**.

| Model | Overall ECE | Worst stratum |
|---|---|---|
| $\chi^2$ gate | 0.05–0.15 (expected poor) | 0.10–0.25 (jet cores) |
| $g_\psi$ calibrated | < 0.02 | < 0.05 |

### 10.5 Calibration is a property of a *distribution*

ECE is an expectation over the evaluation population. "Calibrated" is not a property of a
network; it is a property of a network **plus a deployment distribution**. Consequences:

- The calibration split MUST be rolled out under the **deployed policy** at each DAgger
  iteration, not left at iteration-0 CKF states.
- Train loose (coverage), calibrate on-policy (distribution). See §14.

---

## 11. Value function

### 11.1 Target [LOCKED]

$$
V^*_{\text{truth}}(s) \geq V^*_{\text{MDP}}(s) \geq V^{\pi^\dagger}(s)
$$

**Training target is $V^{\pi^\dagger}$**, not $V^*_{\text{truth}}$. $V^*_{\text{truth}}$ ignores
reachability under the propagator and window, and inflates targets systematically for exactly
the hard-scattered tracks where pruning decisions matter most. The gap is bimodal, not uniform,
because hard scatters create correlated downstream misses.

$\pi^\dagger$ (truth-greedy in-MDP): at each surface take $m$'s measurement if it is in the
candidate set, else `hole`. Rollout terminates at the last reachable sensitive surface, no early
stopping.

$$
\texttt{vstar\_soft} := V^{\pi^\dagger}(\tau_{0:k}) =
\frac{\#\{m\text{'s measurements accepted by }\pi^\dagger\text{ from }\tau_{0:k}\}}
{\#\{m\text{'s measurements remaining beyond surface }k\}}
$$

Stored with `env_config_hash`. $\pi^\dagger$ is not provably MDP-optimal — one can construct
states where a *wrong* nearby hit re-centers the filter and recovers more true hits downstream.
That gap is exotic; we accept it and note it.

### 11.2 Training

MLP 12 → 128 → 128 → 1, $\sim 3\times10^4$ params. Soft-target BCE:

$$
\mathcal{L} = -\frac1N\sum_i\big[V_i^{\pi^\dagger}\log\sigma(z_i) + (1-V_i^{\pi^\dagger})\log(1-\sigma(z_i))\big]
$$

One row per (branch, layer), **not** per candidate. Expected train BCE 0.05–0.15, AUC 0.93–0.97.
Diagnostic: 2D histogram of $V_\phi$ vs $V^{\pi^\dagger}$ — should lie on the diagonal.
Ablation: MSE on logit targets.

### 11.3 Deferred: the variance head $\lambda_\psi$

A merged cluster containing $m$ is the right measurement to accept, but its position is a
charge-weighted centroid of two ionization tracks and is biased from $m$'s true crossing point.
Accepting it with nominal $R$ corrupts the filtered state — the gate says "accept, $p=0.9$" and
is right about *association* while saying nothing about *position quality*.

Second head on the same MLP, $\lambda_\psi(x) \geq 1$, with $R \to \lambda_\psi R$:

$$
\mathcal{L}_R = \tfrac12\sum_{i:\,y_i=1}\Big[\delta_i^\top(\lambda_i R_i)^{-1}\delta_i + \log\det(\lambda_i R_i)\Big],
\qquad \delta_i = h_i - h^{\text{true}}_m
$$

**Build only if ahead of schedule after DAgger iteration 1.** But `truth_residual` MUST be
logged now (§6.5) — regenerating it means re-running.

---

## 12. Diagnostics and figures

| ID | Figure | Notes |
|---|---|---|
| G3 | Stratified reliability diagrams, $g_\psi$ vs $\Lambda_{\chi^2}$ | **headline** |
| — | Window-failure rate vs $n$ | quantifies what windowless end-to-end methods win; from pilot |
| — | $\Delta = V^*_{\text{truth}} - V^{\pi^\dagger}$, stratified by $X/X_0$ and $p$ | architectural headroom; reportable regardless of whether $g_\psi$ closes it |
| — | Unit-Gaussian pull test on CKF residuals | ~2 h of work; measures the premise motivating the gate |
| — | Efficiency vs compute-budget curves | greedy / beam / MCTS at matched evaluations (A5) |
| — | AUC/ECE restricted to `cluster_merged` | the cluster-feature claim, isolated |

---

## 13. Ablations

Pre-registered: the list and the hypothesis for each are fixed **before** results are seen.
Same discipline as blinding, same reason.

**Scope-cut order (pre-committed):** A8 → A4c → A9 → A6 → Stage-A v1.
**Never cut:** Stage 0, A1–A3, calibration audits.

**A8b polarity is FLIPPED from the original recipe:**

| | Original | **Current [LOCKED]** |
|---|---|---|
| Primary | exclude `label_ambiguous` rows | include `cluster_merged` at full weight; exclude only `majority_undefined` |
| A8b | include at $w = 0.1$ | exclude merged clusters |

**New pre-registered test.** Train with and without $(\kappa_u, \kappa_v, \tilde Q,
\sigma_{\cdot\cdot})$; report $\Delta$AUC restricted to `cluster_merged` rows. If $\Delta$ is
small, the cluster-feature mechanism is wrong — and that needs to be discovered in week 2,
not week 6.

**Reweighting ablation:** unweighted (primary) vs weighted + affine calibration, comparing AUC
and the fitted $(b, -\log w_+)$ (§9.3).

---

## 14. DAgger

Iteration 0 trains on states visited by the standard CKF; after deployment the gate's own
decisions change which states are visited. The correction rests on separating two requirements:

$$
\underbrace{\text{supp}(\rho_{\text{train}}) \supseteq \text{supp}(\rho_{\text{deploy}})}_{\text{discrimination}}
\qquad\text{vs.}\qquad
\underbrace{\rho_{\text{cal}} \approx \rho_{\text{deploy}}}_{\text{calibration}}
$$

Coverage strictly favors looseness — a looser config's branch tree *contains* a tighter one's.
Calibration cuts the other way: if $g_\psi$ were exactly well-specified, calibration would
transfer to any $\rho$ (covariate-shift robustness of a correct conditional model), but a 50K-
parameter MLP allocates capacity according to $\rho_{\text{train}}$. Hence **train loose,
calibrate on-policy**.

Two limits on looseness: compute, and **label validity** — very loose configs generate long junk
branches with unstable majorities that hit `majority_undefined` and get discarded anyway.

Distinguish **loose** from **worse**. Degrading the *seeding* config changes the particle
population being followed and shifts $P(y{=}1\mid x)$ through the physics, not merely the state
marginal. Loosen the *following* knobs; handle seeding by collecting the superset and filtering
(§6.7).

```
D_0 = decision log from envelope config on train events
Train g_ψ^(0), V_φ^(0); calibrate on the calibration split
For i = 1, 2, ...:
    Roll out gated beam policy using g_ψ^(i-1), V_φ^(i-1) on train events
    Log ALL visited states -> D_i
    Oracle-relabel every state in D_i from truth
    D = D ∪ D_i                     # aggregate, never replace
    Retrain from scratch on D
    RE-ROLL the calibration split under the current policy, then recalibrate
    Audit ECE; check drift
```

2–3 iterations expected; cap at 3. Dynamics are deterministic, labels exact, state space
low-dimensional. Stopping rule on AUC / MAE / ECE thresholds.

---

## 15. Risks and open items

| Risk | Severity | Action |
|---|---|---|
| **64 events total** (24 train / 4 cal after the split) | **Highest.** Design doc targets ~2000; event diversity matters more than event count for occupancy-conditional calibration. Rows are abundant but correlated within an event. | Ask Louie **today**, in parallel with the rollout. Longest latency of anything on the list. If more do not arrive by early August, report a coarser stratification and say so. |
| Digitization hook missing → null cluster features | High | Pilot check 1, §6.6 |
| Seed subset-recoverability fails | Medium | Pilot check 2, §6.6 |
| NERSC maintenance | Medium | Modal fallback; dual backend behind a config flag |
| Storage blowup from envelope config | Medium | Pilot check 3, §6.6 |

**[OPEN] decisions, do not resolve unilaterally:**

1. Window multiplier $n$ (§6.3) — decide from pilot.
2. $y^{\text{any}}$ vs $y^{\text{maj}}$ (§7.2) — read the metric module first.
3. ODD efficiency denominator: $N_{\min}$, $\eta_{\max}$ (§3).
4. `numMeasurementsCutOff` ACTS default: 1 or 2 (§4.1).
5. Pixel-only vs full detector (data-volume dependent).
6. C++ CKF hook vs Python replay environment. Python replay is pre-committed; the C++ hook is a
   week-7 stretch needed only for final timing numbers.
7. Paper vs poster-only — decision point after the first full-pipeline numbers.

---

## 16. Changelog

| Date | Change |
|---|---|
| 2026-07-28 | Document created. Consolidates: split trap ([0,32) vs [32,64)); envelope collection config; $R_k$ in $S_k$; window as bounding box of the ellipse; ambiguity flag split into `majority_undefined` / `cluster_merged`; A8b polarity flip; removal of positive reweighting from the primary; occupancy-conditional 4-parameter calibrator replacing global temperature scaling; normalized cluster features $\kappa_u,\kappa_v,\tilde Q$; hole vs window-failure distinction; irreversible logging fields; $\rho$ notation collision resolved. |

---

## Coding tier system

Classify **before** starting any coding task.

- **Tier 1** — Matthew implements. Core algorithms: loss functions, MDP representation, value
  targets, reward/label definitions, the gate and value network forward passes. Claude's role is
  math-first explanation plus skeleton blocks with `TODO(human)`. **Never write these outright.**
- **Tier 2** — Claude writes integration and wiring, with explanatory comments. Data loaders,
  ACTS config plumbing, DAgger loop scaffolding, Parquet schema handling.
- **Tier 3** — Full delegation. Visualization, logging, W&B integration, plotting, file
  management, infrastructure.

Anything touching §7 (labels), §9.2 (loss), §10.3 (calibrator), or §11.1 (value target) is
Tier 1 by default.