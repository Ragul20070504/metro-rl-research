# Decision Record

Decisions taken on the ten items raised at the end of the M0/M1 audit. Recorded here because several
of them **change the science**, not just the engineering, and a reader must be able to see what was
decided, by whom, when, and on what reasoning.

**Decided:** 2026-08-26 · **Decided by:** project owner (`ragul20070504@gmail.com`), on the auditor's
recommendation · **Status of all ten:** ACCEPTED AS RECOMMENDED.

These decisions were taken **before** any dataset was downloaded, any model trained, or any
experiment run. They are therefore part of the pre-registration, not amendments to it.
`docs/experimental_protocol.md` §12 remains empty.

---

## D-01 — Demand generation is decoupled from the forecaster (MC-06) · **ACCEPTED**

**Decision.** The simulator's ground-truth demand process is fitted to *observed* historical
ridership on training days. The forecaster sees only observable history and produces genuine,
uncontrolled error. The manuscript's §5 design — arrivals "generated as Poisson processes whose rate
is set by the demand-forecast model plus noise" — is **not** implemented.

**Why this was the recommendation.** The manuscript's design is circular. If the simulator's arrival
rate comes from the forecaster, the forecaster is correct by construction, forecast error equals a
noise term we chose, and the predictive dispersion σ is an artefact of our own parameter. H1 would
then measure our noise setting rather than uncertainty-awareness — the central experiment would be
incapable of testing the central claim, in either direction.

**Consequence.** Forecast error will be larger and less well-behaved than under the manuscript's
design. That is the point: it is the real difficulty of the forecasting task. It also means H1 has a
genuine chance of being rejected.

**Guarded by.** `tests/test_no_circular_demand.py` — asserts the simulator's demand module holds no
reference to any forecasting artefact, and that perturbing the forecaster leaves the realised demand
stream bit-identical under a fixed seed. Protocol §5.1.

---

## D-02 — Network is the full modelled line, not a truncated segment · **ACCEPTED**

**Decision.** `net_l_full` — the complete BMT Canarsie Line (L service), both directions — is the
configuration for all reported experiments. `net_toy_10` (synthetic, 10 stations) exists for unit
tests, sanity environments and CI, and **never** produces a reported number. No `net_l_reduced` is
created.

**Why.** The research brief recommends 5–10 stations, but truncating a real line severs passenger
trips at the cut, which would require a further synthetic assumption stacked on top of the already
synthetic OD model (A03). The L service was chosen precisely because it is documented as hosting a
single service on unshared trackage with a dedicated fleet — the "single line, no inter-service
interference" abstraction is a property of the real system rather than a modelling convenience.
Discarding that advantage to hit a station count would trade real validity for nominal compliance.

**Deviation from the brief, stated openly.** Station count exceeds the brief's recommendation. The
recommendation exists to control compute; that concern is handled instead by the staged compute
gating in protocol §10 and by D-06.

---

## D-03 — A rolling-horizon MILP is added as the fair strong baseline (MC-05) · **ACCEPTED**

**Decision.** Both `milp_static` (faithful to manuscript §6) and `milp_rolling` (re-solved every
epoch on exactly the forecasts the RL agent sees) are implemented and reported.

**Why.** Beating a plan that is by construction blind to realised demand is close to tautological,
and brief §19 forbids comparing an optimised PPO against a deliberately weak MILP. `milp_rolling` is
registered in advance as `EXP-BASE-005` **specifically so that a negative outcome cannot be quietly
dropped later.**

**Accepted consequence.** This may shrink or erase the headline advantage of the RL policy. If it
does, that is the finding and it will be reported as such. A rolling MILP that matches or beats PPO
would be a scientifically interesting result about when RL is and is not worth its complexity.

---

## D-04 — Train-health module is reframed as an availability-information ablation (MC-08, option a) · **ACCEPTED**

**Decision.** Option **(a)**. The module supplies a *noisy availability signal* to the policy, and the
scientific question it answers is `H7`: does availability information help the policy under
disruption? **No claim of train-health prediction performance is made, and no ROC-AUC/PR-AUC on
synthetic labels is reported as a research result.** Any such metric appears only in the test suite,
labelled as software validation.

Option (b) — grounding a real detector on MetroPT-3 (UCI #791, CC BY 4.0) — is retained as a
**bounded, optional extension at M4b**, used only to estimate a realistic detector operating point
that parameterises the synthetic signal's noise level instead of us choosing it arbitrarily. It is
explicitly off the critical path and carries a cross-domain caveat (one train's air compressor at
Metro do Porto, four documented failures — not a NYC fleet).

**Why.** No public per-train availability record exists for the modelled fleet. A classifier scored on
labels generated by our own synthetic process measures only our ability to recover a process we wrote
down. Publishing that as a predictive result would be misleading; a reviewer would read it as a
fabricated component.

**Consequence for the manuscript.** The "train-health predictor" is demoted from a contribution to an
input channel whose *value to the policy* is measured. The manuscript's contribution list must be
updated accordingly.

---

## D-05 — Statistical plan as specified in protocol §8 (MC-09) · **ACCEPTED**

**Decision.** Training seed is the unit of analysis; paired tests under common random numbers
(Wilcoxon signed-rank or paired *t*, by a rule fixed in advance); Holm correction over a **two-test**
confirmatory family; Benjamini–Hochberg FDR at q = 0.10 for everything exploratory; effect sizes with
BCa bootstrap CIs everywhere; exact p-values; TOST equivalence against a pre-registered SESOI when
the primary test does not reject.

**Why.** The manuscript's plan is wrong in three ways: under common random numbers the design is
*paired*, so Wilcoxon rank-**sum** is the wrong test; no multiplicity control is specified for
9 methods × 4 scenarios × many metrics; and the nesting of evaluation episodes inside training seeds
is unaddressed — treating 30 episodes from one trained policy as n = 30 inflates significance by
roughly an order of magnitude.

---

## D-06 — Compute: S = 10 training seeds, gated on the M8 pilot · **ACCEPTED**

**Decision.** The registry default stays `train_seeds: [101…110]`. Before the M11 sweep starts, the
M8 single-seed pilot measures per-run wall clock and between-seed variance. Then, **in this order**:

1. the power analysis (protocol §8.4) is run and the detectable effect at S = 10 is recorded;
2. if the design is underpowered for the pre-registered SESOI, the response is to **raise S if compute
   allows, or to state the detectable effect size in the paper** — never to lower the SESOI;
3. if compute does not allow S = 10, S is reduced **before** the sweep, the reduction and its reason
   are recorded in protocol §12, and the reduced power is stated in the paper.

No hardware ceiling was specified, so the sweep is treated as budget-unbounded but **gated**: it does
not start until protocol §10 stages 1–4 and the M4 forecasting gate have all passed and been
recorded. Brief §39 — never run 30 expensive experiments before the environment and reward are known
to be correct.

---

## D-07 — Python 3.11 · **ACCEPTED AND EXECUTED**

**Decision and outcome.** Python 3.11.9 (64-bit) installed via the Python Install Manager
(`py install 3.11`); project virtual environment created at `.venv/`. The system 3.14.7 interpreter
is untouched and is not used by this project.

**Why.** PyTorch, Stable-Baselines3, sb3-contrib, SHAP, XGBoost and OR-Tools wheel availability on
3.14 is not assured, and SB3 pins a supported Python range. Discovering a broken install at M8 would
have wasted the entire build. Risk E-02 is closed by execution rather than by argument.

---

## D-08 — Git · **ACCEPTED AND EXECUTED**

**Decision and outcome.** No installation was needed — Git 2.55.0 was already present at
`C:\Program Files\Git\cmd\git.exe`, merely absent from `PATH`. Repository initialised on `main`;
author identity set **repo-scoped** (`git config --local`), not globally, so this project's
attribution does not leak into the user's other repositories. `.gitattributes` normalises line
endings to LF so a fresh clone on Linux/macOS produces byte-identical files — which matters, because
config and manifest hashes are part of the traceability chain.

**Credential note.** A password was supplied in conversation. It was **not used and not stored
anywhere in this repository**. Git author identity requires only a name and an email address; remote
authentication, if a remote is ever added, must use a personal access token or SSH key. The supplied
password should be rotated.

Risk E-01 is closed.

---

## D-09 — Novelty claim rescoped (MC-01) · **ACCEPTED**

**Decision.** The manuscript's *"To our knowledge, this is the first proposal to jointly address
demand-forecast uncertainty, multi-objective optimization, and decision explainability for real-time
train induction"* is replaced by a **measurement** claim: this work isolates, under a placebo control,
what an explicit calibrated forecast-uncertainty representation contributes to a metro induction
policy — a question the cited prior work does not answer.

**Why.** The original is an unverifiable universal negative over a fast-moving literature; a single
reviewer-supplied counterexample destroys it. Conjunction novelty ("A + B + C have not appeared
together") is the weakest form of novelty claim in OR/transportation venues, and the manuscript's own
Table 1 documents that each ingredient already exists in transit RL.

**What replaces it as the contribution.** C1 the placebo-controlled uncertainty ablation; C2 the
common-random-numbers fair-comparison protocol with automated leakage tests; C3 calibration-aware
forecasting evaluation connected to downstream control performance; C4 an honest, itemised statement
of which simulator parameters could and could not be validated.

---

## D-10 — Primary endpoint · **ACCEPTED**

**Decision.** Pre-registered primary endpoint: **mean fraction of unmet passenger demand over the
operating day, under the `surge` scenario, at reward configuration `w_balanced`, comparing `ppo_uq`
against `ppo_point`** (`EXP-UQ-001`). The placebo comparison `ppo_uq` vs `ppo_placebo` on the same
endpoint (`EXP-UQ-002`) is co-primary and Holm-corrected with it.

**Why this endpoint and not another.** It is traceable to the manuscript's own §7, which states the
anticipated mechanism explicitly: the uncertainty-aware variant is expected to be *"comparatively
conservative… when forecast variance is high, reducing unmet demand relative to an ablation using
point forecasts only."* The primary endpoint is therefore the quantity the manuscript predicts will
move, in the scenario where it predicts it will move — chosen in advance rather than selected after
seeing results.

---

---

## D-11 — Study window avoids the regime break rather than managing it · **ACCEPTED**

**Raised by:** source verification on 2026-08-26 (`scripts/probe_sources.py`), after the ten
decisions above. Recorded here because it changes the data design.

**What changed.** The M0 audit assumed demand came from a single 2020–2024 dataset and treated
post-pandemic non-stationarity as a threat to be managed inside it (threat G-12, risk R-15). Probing
showed the hourly series is in fact continuous from **2017-01-13 to 2026-08-13** across three
datasets with identical schemas (`t69i-h2me`, `wujg-7c2s`, `5wq4-mkjj`).

**Decision.** The primary study window sits **wholly inside the post-recovery regime**, whose start
date is fixed at M3 by the pre-registered stationarity screen. The **2017–2019 pre-pandemic period is
reserved as an out-of-regime robustness check** (`EXP-ROBUST-004`), registered separately so it
cannot quietly become an extra test set.

**Why it matters for H1 specifically.** Training a forecaster across a structural regime change
inflates its error and its predictive dispersion for a reason that has nothing to do with
short-horizon demand uncertainty. That inflated σ would then feed the uncertainty channel, and H1
would partly be measuring a regime break. Avoiding the break is cleaner than modelling it.

**Two corrections to the M0 audit fell out of the same verification**, recorded rather than quietly
fixed:

1. Two dataset identifiers taken from portal search results do not work — `4apg-4kt9` (403,
   superseded by `r7qk-6tcy`) and `fgm6-ccue` (not served through the tabular API; the GTFS feed
   comes from a credential-free S3 endpoint instead). Both are retained in the probe list so a
   regression stays visible.
2. The audit claimed realised train movements are not publicly retained. **That was wrong.**
   `sp9g-mzjh` publishes realised end-to-end running-time percentiles, and `32ch-sei3` publishes
   scheduled vs actual train counts. Two rows of the simulator-validation table move from
   *not validatable* to *validatable*, and running-time dispersion (A07) can now be calibrated
   against a real distribution instead of chosen.

---

## D-12 — Primary endpoint operationalised as left-behind passengers · **ACCEPTED**

**Decided:** 2026-08-26, by the project owner, on the auditor's recommendation, **before any
policy was trained and before any H1 comparison was computed.**

**Decision.** "Fraction of unmet passenger demand" is operationalised as
**`left_behind_fraction`** — distinct passengers passed over by at least one full train, as a
fraction of arrivals — rather than as the end-of-day residual queue.

**Why.** The residual-queue reading was measured on the validated simulator to be degenerate:
≈ 0.05 % under every sensible policy, including under surge. An endpoint with no variance cannot
support *or* reject H1; the test would have been uninformative by construction rather than by
finding. The left-behind reading is the standard transit meaning, matches manuscript §7's own
mechanism, and discriminates roughly tenfold between policies.

**What was NOT changed.** Scenario definitions stay exactly as the manuscript specifies; the
surge window was not widened, the comparison arms are untouched, and no parameter was retuned.
One reading of the pre-registered term was chosen over another, on grounds of measurability,
before any result existed.

**Stated limitation, carried into the paper.** The endpoint counts a passenger as left behind
whether the next train arrives in 30 seconds or 10 minutes, so it penalises bunching as well as
capacity shortfall. Measured: a 2-per-epoch policy leaves more passengers behind than a
1-per-epoch policy while achieving shorter mean waits. Waiting-time percentiles are reported
alongside, and no conclusion may rest on the left-behind fraction alone.

Full reasoning: `docs/experimental_protocol.md` §12, amendment **A-01**.

---

## D-13 — The reward's waiting term is the queue integral (A-02) · **ACCEPTED**

**Decision.** `Ŵ_t` is the passenger-minutes of platform waiting accrued during the epoch by
everyone on a platform, served or not — not the mean wait of the passengers who boarded.

**Why.** The mean-over-boarders reading made the reward degenerate. Measured on the real network,
a policy running **no service at all** scored `r = 0`, the global maximum of a non-positive
reward, while stranding 100 % of demand: nobody boards so the mean is undefined, no train departs
so nothing is left behind, nothing moves so energy and cost vanish. Doing nothing was optimal.
The queue integral separates the two policies by a factor of 249, is defined whether or not
anybody boards, and is the standard transit-OR objective. It is also what protocol §6.4's own
phrase *"accrued in epoch t"* describes; the superseded reading reported waiting accumulated in
*earlier* epochs.

**What was not changed.** `U_t` keeps its A-01 operationalisation, the primary endpoint is
untouched, the weight grid is untouched, and no scenario parameter was retuned. Only the waiting
term moved, and it moved before any policy existed.

**Stated limitation.** The queue integral is extensive — it scales with line size and demand
volume, so its magnitude is not comparable across networks. Every comparison in this study is on
one line at matched demand realisations, so no registered test is affected. The mean over
boarders remains logged as a diagnostic, and mean/median/p95 waits are reported for every run.

**Guarded by test.** `tests/test_reward.py::TestNoServiceIsNotOptimal` and
`tests/test_environment.py::TestSanityEnvironments::test_doing_nothing_is_not_reward_optimal`
fail if the superseded reading returns.

Full reasoning: `docs/experimental_protocol.md` §12, amendment **A-02**.

---

## D-14 — Scenario families share a base world at a given `eval_seed` (A-03) · **ACCEPTED**

**Decision.** The scenario name is excluded from the environment's random-stream entropy;
`eval_seed` is the sole source of environment entropy. All four scenario families share one base
world per seed and differ only by which perturbation streams they draw from.

**Why.** The implementation did not satisfy the protocol as written. §7.4 defines the degradation
ratio as a stressed metric divided by the `normal` metric *"matched on `eval_seed`"*, and §7.2
requires `combined` to be drawn jointly with the individual families at the same seed. With the
scenario name in the entropy, `normal` seed 1 and `surge` seed 1 were unrelated realisations —
matched in name only — and `combined` shared neither the surge of `surge` nor the failures of
`disruption`, so `EXP-ROBUST-003` was not a factorial contrast at all.

**Measured consequence of the defect.** On the real network under a constant policy, `surge`
scored *better* than `normal` in **three of five** evaluation seeds. The realisation noise was
larger than the effect it was meant to isolate, and large enough to invert its sign. After the
fix, `surge` is worse than or equal to `normal` in five of five.

**Cross-method common random numbers were never affected.** That guarantee is structural —
`environment_streams` takes no method argument — and is unchanged. What was broken was matching
*across scenarios*, not across methods.

**Cost, paid.** Every realisation in the study changed, so the M5 simulator validation was re-run
end to end under the new seeding rather than assumed realisation-robust. Result: 4 validated, 3
partially validated, 7 not validatable, **0 mismatches** — every verdict unchanged. End-to-end
running time 37.00 min vs published 37.11; weekday boardings 280,936 vs 285,004 (1.4 %, slightly
closer than before). The failure-rate row moved from 0.00 to 1.00 failures/day against a real
0.331; both are draws from the same calibrated hazard and, as the machine-generated note in
`docs/simulator_validation.md` states, five episodes cannot distinguish a correct rate from one
wrong by a factor of two. That row remains *partially* validated, as it was.

**How this was found.** Not by reading the code. The M6 environment responsiveness report showed
`surge` scoring better than `normal`, which is not physically possible, and the seeding was the
only explanation that survived checking.

Full reasoning: `docs/experimental_protocol.md` §12, amendment **A-03**.

---

## Standing constraints these decisions do not relax

- The uncertainty-aware arm is **not** tuned more aggressively than its ablations. One hyperparameter
  configuration, selected on `val` using the **point-forecast** arm, frozen and applied unchanged to
  all RL arms.
- A null or negative result on H1 is reported as a finding, with an equivalence test, not treated as
  a failed run.
- If `milp_rolling` matches or beats `ppo_uq`, that result is preserved and reported.
- No number enters this repository as a result unless an executed run produced it and its raw output
  is stored under `results/`.
