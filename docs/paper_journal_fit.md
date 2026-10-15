# Journal fit assessment

*Generated at 2026-08-28T09:50:46Z.*

**No claim is made that any journal will accept this manuscript.** Acceptance depends on editorial judgement and on reviewers who may not value a negative result. What follows is a scope and methodology assessment only.

Format requirements below were **not** verified against current author guidelines and are marked accordingly; confirm each against the publisher's guide for authors before submission.

## Transportation Research Part C: Emerging Technologies

| Dimension | Assessment |
|---|---|
| Scope fit | Strong. The journal's stated focus is scholarly research on the development, application and implications of emerging transportation technologies, with interest in implications for planning, operation and control rather than in technologies themselves. A controlled evaluation of a learned controller's operational implications sits squarely inside that framing. |
| Methodological fit | Strong. Simulation-based evaluation of learning-based control is core subject matter. |
| Transportation relevance | Strong. Metro dispatch, real ridership data, operational baselines. |
| Negative-result compatibility | Moderate. The journal publishes methodological evaluation; a negative result with an identified mechanism and a placebo control is a plausible fit, but editors may still prefer a positive contribution. |
| Likely reviewer concerns | That no learned arm beats a heuristic; that the environment's crowding components are unvalidated; that a single training configuration confounds the interpretation. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED — confirm word limit, structure and reference style against the current guide for authors]` |

## Transportation Research Part B: Methodological

| Dimension | Assessment |
|---|---|
| Scope fit | Moderate. Part B is the natural home for the metro-scheduling literature this paper builds on, including demand-uncertain metro rescheduling and MILP scheduling formulations. However Part B favours methodological and mathematical contribution, and this paper's contribution is experimental rather than a new formulation. |
| Methodological fit | Moderate. The statistical design is rigorous, but no new model or algorithm is proposed. |
| Transportation relevance | Strong. |
| Negative-result compatibility | Lower. A methodological journal expecting a formulation or theorem may regard a controlled empirical falsification as out of scope. |
| Likely reviewer concerns | Absence of a methodological contribution in the Part B sense. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED]` |

## Journal of Rail Transport Planning & Management

| Dimension | Assessment |
|---|---|
| Scope fit | Strong and specific. Rail operations planning and management is the exact domain, and the journal publishes applied rail studies including learning-based dispatching. |
| Methodological fit | Good. Applied simulation studies are standard. |
| Transportation relevance | Very strong — rail-specific. |
| Negative-result compatibility | Higher than the Part B/C pair. A smaller, domain-focused venue is more likely to value a careful negative operational result. |
| Likely reviewer concerns | Lower profile than Part B/C; may request more operational grounding. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED]` |

## European Journal of Operational Research

| Dimension | Assessment |
|---|---|
| Scope fit | Moderate. The convex-loss/newsvendor framing connects to OR theory, and EJOR publishes applied OR in transport. |
| Methodological fit | Moderate. EJOR would likely want the OR contribution foregrounded — the convex service loss and the MILP baselines — rather than the RL comparison. |
| Transportation relevance | Moderate; EJOR is domain-general. |
| Negative-result compatibility | Moderate. |
| Likely reviewer concerns | The paper's centre of gravity is experimental design, not optimisation. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED]` |

## Computers & Industrial Engineering

| Dimension | Assessment |
|---|---|
| Scope fit | Moderate. Broad enough to accept a simulation-based operational study. |
| Methodological fit | Adequate. |
| Transportation relevance | Weaker fit; less transport-specialised readership. |
| Negative-result compatibility | Moderate. |
| Likely reviewer concerns | The transport-specific contribution may be under-appreciated. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED]` |

## Transportation Science

| Dimension | Assessment |
|---|---|
| Scope fit | Lower. Strongly methodological and theory-oriented; an empirical falsification without a new model is an unlikely fit. |
| Methodological fit | Lower for this paper. |
| Transportation relevance | Strong. |
| Negative-result compatibility | Low. |
| Likely reviewer concerns | Absence of theoretical contribution. |
| Article type | Research Paper |
| Format requirements | `[FORMAT REQUIREMENTS NOT VERIFIED]` |

## Recommendation

| Role | Venue | Reasoning |
|---|---|---|
| **PRIMARY** | *Transportation Research Part C: Emerging Technologies* | The stated scope — implications of emerging technologies for the operation and control of transportation systems — matches a controlled evaluation of a learned dispatch controller more closely than any alternative. The readership is the one most likely to act on the finding that a tuned heuristic outperformed the learned arms. |
| **BACKUP** | *Transportation Research Part B: Methodological* | Home to the closest prior work on metro scheduling under uncertain demand, so the contribution is legible to its reviewers, though the absence of a new formulation is a real risk. |
| **SAFE** | *Journal of Rail Transport Planning & Management* | Rail-specific and applied, and the venue most likely to value a careful negative operational result on its own terms. |

Venues are ranked by scope fit and by the likelihood that reviewers will engage with a controlled negative result — **not** by impact factor or quartile. A journal chosen solely for its ranking would be the wrong home for this paper.

## Framing advice, whichever venue

1. Lead with the **design**, not the algorithm. The contribution is the control structure that makes the comparison interpretable.
2. Report the heuristic comparison **early and plainly**. A reviewer who finds it in a figure after the text has passed over it will distrust the rest.
3. State the training-budget confound in the abstract, not only in the limitations.
4. Do not describe the environment as a validated digital twin.
5. Keep the primary result exactly as measured: +0.013742, Holm-adjusted p = 0.007741, H1 = NOT SUPPORTED.
