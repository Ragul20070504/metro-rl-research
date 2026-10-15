# Peer-review resilience pass

*Generated at 2026-08-28T09:50:46Z from the same stored artefacts as the manuscript. Three hostile reviewers; for each, the strongest criticism, whether it is fatal, the evidence already in the paper, the exact location that addresses it, the remaining weakness, and the revision required.*

## R1 — Transportation / Operations Research

**Stance.** The experimental discipline is unusual for this literature and the negative result is credible. My concerns are about what the paper can claim for practice and whether the operational model supports any of it.

### R1.1 A threshold rule beats the proposed controller, so what is the operational contribution?

Table 4 shows a tuned threshold heuristic (0.01184) and a static MILP (0.01187) outperforming every learned arm, with the proposed controller (0.03168) worse even than a fixed timetable (0.02550). What does an operator gain from this paper?

- **Fatal?** **No** to the paper as a controlled evaluation; **fatal** to any framing of it as a method contribution.
- **Evidence already in the paper:** Reported as the first results subsection rather than an appendix, with the practical implication drawn explicitly.
- **Addressed at:** §11.1, Table 4, Figure 10, §15
- **Remaining weakness:** The paper offers no operational improvement, and does not claim one.
- **Required revision:** None outstanding. Resist any editorial pressure to move this comparison later in the paper.

### R1.2 The endpoint depends on unvalidated simulator components

7 of 14 quantities are not externally validated, and four of them — capacity and peak occupancy, dwell-time distribution, passenger balking and patience, and the OD distribution — are direct inputs to the primary endpoint. The validated quantities constrain supply, not crowding.

- **Fatal?** **No** for the ordering of methods, which is internally valid under common random numbers; **fatal** for any magnitude claim.
- **Evidence already in the paper:** Stated with the four quantities named, and every magnitude claim scoped to within-environment ordering.
- **Addressed at:** §5, Table 3, §16 item 5
- **Remaining weakness:** Absolute effect sizes cannot be transferred to a physical system.
- **Required revision:** None outstanding. Verify at proof stage that no abstract or conclusion sentence quotes an absolute passenger figure as an operational prediction.

### R1.3 Synthetic OD structure could manufacture the result

The left-behind fraction depends directly on origin–destination structure, which is assumed rather than measured.

- **Fatal?** **No.** Addressed by design.
- **Evidence already in the paper:** The comparison was re-scored under three OD variants; `ppo_uq` is worse in 12 of 12 sensitivity conditions and none reverses it.
- **Addressed at:** §13, Table 8, Figure 11
- **Remaining weakness:** The variants re-score policies *trained* under the registered OD model, so they test robustness of evaluation, not of training.
- **Required revision:** None. The manuscript already states that limitation.

## R2 — Reinforcement Learning / Machine Learning

**Stance.** The placebo control is genuinely good and rare in this literature. But I am not convinced the paper is about uncertainty; I think it may be about an undertrained agent.

### R2.1 An agent that cannot beat a threshold rule is undertrained, so this is a result about the optimiser

One frozen configuration at one budget, and no learned arm beats a heuristic. The parsimonious explanation is that none of the agents learned the task and the uncertainty arm simply faced a harder optimisation problem.

- **Fatal?** **Not fatal** to the reported claim, which is explicitly scoped to this configuration and budget. It *is* fatal to any general claim about uncertainty-aware RL — which the paper does not make.
- **Evidence already in the paper:** Conceded as the principal limitation and given a full reading in the discussion. The placebo comparison answers it partially: a shuffled signal of identical dimensionality does not reproduce the effect, so the gap is not pure input-capacity cost.
- **Addressed at:** §14 Reading 2, §16 items 2–3, §17 item 1
- **Remaining weakness:** Unresolved. The registered follow-up that would settle it is incomplete (11 of 80 policies, no evaluation cells).
- **Required revision:** **Requires a new experiment**, which is specified and pre-registered but not run. Do not resume it partially and report whichever budgets finish.

### R2.2 No learning curves, so convergence cannot be assessed

Only final-policy performance is reported. A converged policy that uses uncertainty badly cannot be distinguished from one that never converged.

- **Fatal?** **No**, but it is a genuine gap in the evidence.
- **Evidence already in the paper:** The mechanism artefact records this as unanswerable from stored data; the limitation is stated.
- **Addressed at:** §16 item 2, §12.3 limitation row
- **Remaining weakness:** Convergence cannot be compared across arms from the stored artefacts.
- **Required revision:** State explicitly in §16 that no learning curves were stored, and require curve logging in the follow-up protocol.

### R2.3 The convex-loss argument is about the environment, not the agent's objective

Service loss is convex in arrivals, but the agent optimises a scalarised reward including energy and cost. The optimal hedging direction under *that* reward is not obviously upward.

- **Fatal?** **No**, but it weakens the interpretation if handled loosely.
- **Evidence already in the paper:** The paper raises this objection against its own interpretation, and separates observation from interpretation explicitly.
- **Addressed at:** §12.3 interpretation table, §14 Reading 3
- **Remaining weakness:** The argument motivates a direction of interest; it does not derive the optimal action under the full multi-objective reward.
- **Required revision:** None. Preserve the OBSERVED / MECHANISM / INTERPRETATION / LIMITATION separation through copy-editing; it must not collapse into a stronger claim.

### R2.4 SHAP adds nothing and invites causal misreading

Attributions come from one policy at one seed (`ppo_uq`, seed 103). Feature-importance orderings are seed-unstable.

- **Fatal?** **No.**
- **Evidence already in the paper:** No causal claim is made; the material supports no hypothesis.
- **Addressed at:** §13 final paragraph, S9
- **Remaining weakness:** It is presentationally weak.
- **Required revision:** Already demoted to supplementary material; keep it there.

## R3 — Statistics / Experimental Methodology

**Stance.** Pre-registration and the seed-level unit of analysis are handled better than most submissions I see. My concerns are power, what the design can test, and one class of claim that is not licensed.

### R3.1 The pilot variance estimate was wrong by a factor of two

The gate used a four-seed pilot estimating SD = 0.005298; the realised SD was 0.011269, 2.13× larger. Every power figure derived from the pilot is optimistic.

- **Fatal?** **No.** The rejection is computed from realised data; the planning estimate does not enter the p-value or the interval.
- **Evidence already in the paper:** Both standard deviations and the ratio are reported, the registered MDE is described as optimistic, and **no post-hoc power argument is made anywhere in the paper**.
- **Addressed at:** §9 final two paragraphs, §16 item 1
- **Remaining weakness:** None beyond the stated underpowering.
- **Required revision:** None. Confirm at proof stage that no sentence of the form 'the effect is N× the MDE, therefore conclusive' has reappeared; that is a fallacy independent of the arithmetic.

### R3.2 You cannot test RL against your baselines, which is the comparison that matters operationally

Baselines are deterministic and seedless, so under the seed-level unit they contribute n = 1 and no paired test is defined. Given the descriptive comparison is unfavourable, a sceptical reader may read this as a convenient gap.

- **Fatal?** **No**, but it limits the paper to descriptive statements about its most operationally relevant comparison.
- **Evidence already in the paper:** Stated explicitly wherever the comparison appears, and labelled as differences of means rather than tests.
- **Addressed at:** §11.1 final paragraph, Table 4 caption
- **Remaining weakness:** No inferential RL-versus-baseline statement is available.
- **Required revision:** Consider adding an evaluation-seed-paired secondary analysis (pairing on the 30 evaluation days with the learned arm averaged over training seeds), clearly labelled secondary to the registered primary. It is computable from stored data.

### R3.3 Underpowered for your own SESOI, so nulls say nothing

Detecting the SESOI of 0.002 needs n = 69 at the pilot SD and more at the realised SD.

- **Fatal?** **No.** Correctly handled.
- **Evidence already in the paper:** Every null is reported as inconclusive, never as equivalence, and TOST is run only on non-rejection.
- **Addressed at:** §9, §10, §16 item 1
- **Remaining weakness:** None.
- **Required revision:** Confirm no null anywhere is phrased as 'no difference' rather than 'no difference detected'.

### R3.4 Dependence across the sensitivity family

12 sensitivity conditions re-score the *same* policies. Consistent signs across correlated conditions are not independent evidence.

- **Fatal?** **No**, but the strength of the sign argument should not be overstated.
- **Evidence already in the paper:** The paper states that individual conditions mostly do not clear the BH threshold, and that the conditions are strongly dependent because they re-score identical policies.
- **Addressed at:** §13 second paragraph
- **Remaining weakness:** The sensitivity evidence is weaker than independent replication.
- **Required revision:** None; the qualification is already present.

### R3.5 Provenance of the stored results

Most stored cells were produced from an uncommitted working tree, so the recorded commit does not determine the code that produced them.

- **Fatal?** **No**, but it undermines a commit-level reproducibility claim.
- **Evidence already in the paper:** Disclosed as a provenance caveat rather than left in the manifests.
- **Addressed at:** §18 provenance caveat
- **Remaining weakness:** Commit-anchored results require a re-run from the committed state.
- **Required revision:** None outstanding.

## Summary

| ID | Reviewer | Fatal to the reported claim? | Needs a new experiment? | Status |
|---|---|---|---|---|
| R1.1 | Transport/OR | No (fatal only to a method framing) | No | Reported prominently |
| R1.2 | Transport/OR | No (fatal to magnitude claims) | No | Disclosed and scoped |
| R1.3 | Transport/OR | No | No | Addressed by OD sensitivity |
| R2.1 | RL/ML | No (claim is scoped) | **Yes** | Follow-up specified, incomplete |
| R2.2 | RL/ML | No | Yes, to resolve | Gap; stated |
| R2.3 | RL/ML | No | No | Handled by claim separation |
| R2.4 | RL/ML | No | No | Moved to supplementary |
| R3.1 | Statistics | No | No | Corrected and reported |
| R3.2 | Statistics | No | No | Secondary analysis recommended |
| R3.3 | Statistics | No | No | Correctly handled |
| R3.4 | Statistics | No | No | Qualification present |
| R3.5 | Statistics | No | No | Disclosed |

**No reviewer raises an objection fatal to the reported claim.** The claim survives because it is narrow: a pre-registered, placebo-controlled result about one controller, one configuration, one budget and one environment. Every fatality identified attaches to a *broader* claim the manuscript deliberately does not make.

**The single most damaging criticism is R2.1** — that the result may be about an undertrained optimiser rather than about uncertainty. It is the only criticism requiring a new experiment; that experiment is specified and pre-registered, and it was stopped incomplete. The manuscript reports the confound prominently rather than claiming it away.

**Recommendation.** Major revision, on R3.2 (add or justify omitting the evaluation-seed-paired baseline comparison) and R2.2 (state the learning-curve gap explicitly). Neither requires new computation.
