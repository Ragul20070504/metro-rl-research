"""Render the three-reviewer resilience pass and the journal-fit assessment.

Numbers come from the same `Study` object the manuscript is built from, so a reviewer response
cannot cite a figure the paper does not contain.

Writes ``docs/paper_peer_review.md`` and ``docs/paper_journal_fit.md``.

    python scripts/build_paper_review.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

from datetime import datetime, timezone

from scripts.build_manuscript import DOCS, Study


def peer_review(s: Study) -> str:
    p, pw = s.primary, s.power
    L: list[str] = []
    add = L.append
    add("# Peer-review resilience pass")
    add("")
    add(f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} from the same stored "
        "artefacts as the manuscript. Three hostile reviewers; for each, the strongest "
        "criticism, whether it is fatal, the evidence already in the paper, the exact location "
        "that addresses it, the remaining weakness, and the revision required.*")
    add("")

    def review(tag: str, discipline: str, stance: str, items: list[dict]) -> None:
        add(f"## {tag} — {discipline}")
        add("")
        add(f"**Stance.** {stance}")
        add("")
        for index, item in enumerate(items, start=1):
            add(f"### {tag}.{index} {item['title']}")
            add("")
            add(item["criticism"])
            add("")
            add(f"- **Fatal?** {item['fatal']}")
            add(f"- **Evidence already in the paper:** {item['evidence']}")
            add(f"- **Addressed at:** {item['location']}")
            add(f"- **Remaining weakness:** {item['remaining']}")
            add(f"- **Required revision:** {item['revision']}")
            add("")

    review("R1", "Transportation / Operations Research",
           "The experimental discipline is unusual for this literature and the negative result is "
           "credible. My concerns are about what the paper can claim for practice and whether the "
           "operational model supports any of it.",
           [
               {
                   "title": "A threshold rule beats the proposed controller, so what is the "
                            "operational contribution?",
                   "criticism": f"Table 4 shows a tuned threshold heuristic "
                                f"({s.rank.loc['heuristic_threshold', 'surge']:.5f}) and a static "
                                f"MILP ({s.rank.loc['milp_static', 'surge']:.5f}) outperforming "
                                f"every learned arm, with the proposed controller "
                                f"({s.rank.loc['ppo_uq', 'surge']:.5f}) worse even than a fixed "
                                f"timetable ({s.rank.loc['fixed_timetable', 'surge']:.5f}). What "
                                "does an operator gain from this paper?",
                   "fatal": "**No** to the paper as a controlled evaluation; **fatal** to any "
                            "framing of it as a method contribution.",
                   "evidence": "Reported as the first results subsection rather than an "
                               "appendix, with the practical implication drawn explicitly.",
                   "location": "§11.1, Table 4, Figure 10, §15",
                   "remaining": "The paper offers no operational improvement, and does not claim "
                                "one.",
                   "revision": "None outstanding. Resist any editorial pressure to move this "
                               "comparison later in the paper.",
               },
               {
                   "title": "The endpoint depends on unvalidated simulator components",
                   "criticism": f"{len(s.unvalidated)} of {len(s.sim['rows'])} quantities are not "
                                "externally validated, and four of them — capacity and peak "
                                "occupancy, dwell-time distribution, passenger balking and "
                                "patience, and the OD distribution — are direct inputs to the "
                                "primary endpoint. The validated quantities constrain supply, not "
                                "crowding.",
                   "fatal": "**No** for the ordering of methods, which is internally valid under "
                            "common random numbers; **fatal** for any magnitude claim.",
                   "evidence": "Stated with the four quantities named, and every magnitude claim "
                               "scoped to within-environment ordering.",
                   "location": "§5, Table 3, §16 item 5",
                   "remaining": "Absolute effect sizes cannot be transferred to a physical system.",
                   "revision": "None outstanding. Verify at proof stage that no abstract or "
                               "conclusion sentence quotes an absolute passenger figure as an "
                               "operational prediction.",
               },
               {
                   "title": "Synthetic OD structure could manufacture the result",
                   "criticism": "The left-behind fraction depends directly on origin–destination "
                                "structure, which is assumed rather than measured.",
                   "fatal": "**No.** Addressed by design.",
                   "evidence": f"The comparison was re-scored under three OD variants; `ppo_uq` "
                               f"is worse in "
                               f"{s.sens['direction_agreement']['conditions_where_ppo_uq_is_worse']}"
                               f" of {s.sens['direction_agreement']['conditions_total']} "
                               "sensitivity conditions and none reverses it.",
                   "location": "§13, Table 8, Figure 11",
                   "remaining": "The variants re-score policies *trained* under the registered OD "
                                "model, so they test robustness of evaluation, not of training.",
                   "revision": "None. The manuscript already states that limitation.",
               },
           ])

    review("R2", "Reinforcement Learning / Machine Learning",
           "The placebo control is genuinely good and rare in this literature. But I am not "
           "convinced the paper is about uncertainty; I think it may be about an undertrained "
           "agent.",
           [
               {
                   "title": "An agent that cannot beat a threshold rule is undertrained, so this "
                            "is a result about the optimiser",
                   "criticism": "One frozen configuration at one budget, and no learned arm beats "
                                "a heuristic. The parsimonious explanation is that none of the "
                                "agents learned the task and the uncertainty arm simply faced a "
                                "harder optimisation problem.",
                   "fatal": "**Not fatal** to the reported claim, which is explicitly scoped to "
                            "this configuration and budget. It *is* fatal to any general claim "
                            "about uncertainty-aware RL — which the paper does not make.",
                   "evidence": "Conceded as the principal limitation and given a full reading in "
                               "the discussion. The placebo comparison answers it partially: a "
                               "shuffled signal of identical dimensionality does not reproduce the "
                               "effect, so the gap is not pure input-capacity cost.",
                   "location": "§14 Reading 2, §16 items 2–3, §17 item 1",
                   "remaining": "Unresolved. The registered follow-up that would settle it is "
                                "incomplete (11 of 80 policies, no evaluation cells).",
                   "revision": "**Requires a new experiment**, which is specified and "
                               "pre-registered but not run. Do not resume it partially and report "
                               "whichever budgets finish.",
               },
               {
                   "title": "No learning curves, so convergence cannot be assessed",
                   "criticism": "Only final-policy performance is reported. A converged policy "
                                "that uses uncertainty badly cannot be distinguished from one "
                                "that never converged.",
                   "fatal": "**No**, but it is a genuine gap in the evidence.",
                   "evidence": "The mechanism artefact records this as unanswerable from stored "
                               "data; the limitation is stated.",
                   "location": "§16 item 2, §12.3 limitation row",
                   "remaining": "Convergence cannot be compared across arms from the stored "
                                "artefacts.",
                   "revision": "State explicitly in §16 that no learning curves were stored, and "
                               "require curve logging in the follow-up protocol.",
               },
               {
                   "title": "The convex-loss argument is about the environment, not the agent's "
                            "objective",
                   "criticism": "Service loss is convex in arrivals, but the agent optimises a "
                                "scalarised reward including energy and cost. The optimal hedging "
                                "direction under *that* reward is not obviously upward.",
                   "fatal": "**No**, but it weakens the interpretation if handled loosely.",
                   "evidence": "The paper raises this objection against its own interpretation, "
                               "and separates observation from interpretation explicitly.",
                   "location": "§12.3 interpretation table, §14 Reading 3",
                   "remaining": "The argument motivates a direction of interest; it does not "
                                "derive the optimal action under the full multi-objective reward.",
                   "revision": "None. Preserve the OBSERVED / MECHANISM / INTERPRETATION / "
                               "LIMITATION separation through copy-editing; it must not collapse "
                               "into a stronger claim.",
               },
               {
                   "title": "SHAP adds nothing and invites causal misreading",
                   "criticism": f"Attributions come from one policy at one seed "
                                f"(`{s.shap['method']}`, seed {s.shap['train_seed']}). "
                                "Feature-importance orderings are seed-unstable.",
                   "fatal": "**No.**",
                   "evidence": "No causal claim is made; the material supports no hypothesis.",
                   "location": "§13 final paragraph, S9",
                   "remaining": "It is presentationally weak.",
                   "revision": "Already demoted to supplementary material; keep it there.",
               },
           ])

    review("R3", "Statistics / Experimental Methodology",
           "Pre-registration and the seed-level unit of analysis are handled better than most "
           "submissions I see. My concerns are power, what the design can test, and one class of "
           "claim that is not licensed.",
           [
               {
                   "title": "The pilot variance estimate was wrong by a factor of two",
                   "criticism": f"The gate used a four-seed pilot estimating SD = "
                                f"{pw['between_seed_sd']:.6f}; the realised SD was "
                                f"{s.sd_realised:.6f}, "
                                f"{s.sd_realised / pw['between_seed_sd']:.2f}× larger. Every power "
                                "figure derived from the pilot is optimistic.",
                   "fatal": "**No.** The rejection is computed from realised data; the planning "
                            "estimate does not enter the p-value or the interval.",
                   "evidence": "Both standard deviations and the ratio are reported, the "
                               "registered MDE is described as optimistic, and **no post-hoc "
                               "power argument is made anywhere in the paper**.",
                   "location": "§9 final two paragraphs, §16 item 1",
                   "remaining": "None beyond the stated underpowering.",
                   "revision": "None. Confirm at proof stage that no sentence of the form 'the "
                               "effect is N× the MDE, therefore conclusive' has reappeared; that "
                               "is a fallacy independent of the arithmetic.",
               },
               {
                   "title": "You cannot test RL against your baselines, which is the comparison "
                            "that matters operationally",
                   "criticism": "Baselines are deterministic and seedless, so under the "
                                "seed-level unit they contribute n = 1 and no paired test is "
                                "defined. Given the descriptive comparison is unfavourable, a "
                                "sceptical reader may read this as a convenient gap.",
                   "fatal": "**No**, but it limits the paper to descriptive statements about its "
                            "most operationally relevant comparison.",
                   "evidence": "Stated explicitly wherever the comparison appears, and labelled "
                               "as differences of means rather than tests.",
                   "location": "§11.1 final paragraph, Table 4 caption",
                   "remaining": "No inferential RL-versus-baseline statement is available.",
                   "revision": "Consider adding an evaluation-seed-paired secondary analysis "
                               "(pairing on the 30 evaluation days with the learned arm averaged "
                               "over training seeds), clearly labelled secondary to the "
                               "registered primary. It is computable from stored data.",
               },
               {
                   "title": "Underpowered for your own SESOI, so nulls say nothing",
                   "criticism": f"Detecting the SESOI of {pw['sesoi']} needs "
                                f"n = {pw['required_n_for_sesoi']} at the pilot SD and more at the "
                                "realised SD.",
                   "fatal": "**No.** Correctly handled.",
                   "evidence": "Every null is reported as inconclusive, never as equivalence, and "
                               "TOST is run only on non-rejection.",
                   "location": "§9, §10, §16 item 1",
                   "remaining": "None.",
                   "revision": "Confirm no null anywhere is phrased as 'no difference' rather "
                               "than 'no difference detected'.",
               },
               {
                   "title": "Dependence across the sensitivity family",
                   "criticism": f"{s.sens['direction_agreement']['conditions_total']} sensitivity "
                                "conditions re-score the *same* policies. Consistent signs across "
                                "correlated conditions are not independent evidence.",
                   "fatal": "**No**, but the strength of the sign argument should not be "
                            "overstated.",
                   "evidence": "The paper states that individual conditions mostly do not clear "
                               "the BH threshold, and that the conditions are strongly dependent "
                               "because they re-score identical policies.",
                   "location": "§13 second paragraph",
                   "remaining": "The sensitivity evidence is weaker than independent replication.",
                   "revision": "None; the qualification is already present.",
               },
               {
                   "title": "Provenance of the stored results",
                   "criticism": "Most stored cells were produced from an uncommitted working "
                                "tree, so the recorded commit does not determine the code that "
                                "produced them.",
                   "fatal": "**No**, but it undermines a commit-level reproducibility claim.",
                   "evidence": "Disclosed as a provenance caveat rather than left in the "
                               "manifests.",
                   "location": "§18 provenance caveat",
                   "remaining": "Commit-anchored results require a re-run from the committed "
                                "state.",
                   "revision": "None outstanding.",
               },
           ])

    add("## Summary")
    add("")
    add("| ID | Reviewer | Fatal to the reported claim? | Needs a new experiment? | Status |")
    add("|---|---|---|---|---|")
    for row in (
        ("R1.1", "Transport/OR", "No (fatal only to a method framing)", "No", "Reported prominently"),
        ("R1.2", "Transport/OR", "No (fatal to magnitude claims)", "No", "Disclosed and scoped"),
        ("R1.3", "Transport/OR", "No", "No", "Addressed by OD sensitivity"),
        ("R2.1", "RL/ML", "No (claim is scoped)", "**Yes**", "Follow-up specified, incomplete"),
        ("R2.2", "RL/ML", "No", "Yes, to resolve", "Gap; stated"),
        ("R2.3", "RL/ML", "No", "No", "Handled by claim separation"),
        ("R2.4", "RL/ML", "No", "No", "Moved to supplementary"),
        ("R3.1", "Statistics", "No", "No", "Corrected and reported"),
        ("R3.2", "Statistics", "No", "No", "Secondary analysis recommended"),
        ("R3.3", "Statistics", "No", "No", "Correctly handled"),
        ("R3.4", "Statistics", "No", "No", "Qualification present"),
        ("R3.5", "Statistics", "No", "No", "Disclosed"),
    ):
        add("| " + " | ".join(row) + " |")
    add("")
    add("**No reviewer raises an objection fatal to the reported claim.** The claim survives "
        "because it is narrow: a pre-registered, placebo-controlled result about one controller, "
        "one configuration, one budget and one environment. Every fatality identified attaches to "
        "a *broader* claim the manuscript deliberately does not make.")
    add("")
    add("**The single most damaging criticism is R2.1** — that the result may be about an "
        "undertrained optimiser rather than about uncertainty. It is the only criticism requiring "
        "a new experiment; that experiment is specified and pre-registered, and it was stopped "
        "incomplete. The manuscript reports the confound prominently rather than claiming it "
        "away.")
    add("")
    add("**Recommendation.** Major revision, on R3.2 (add or justify omitting the "
        "evaluation-seed-paired baseline comparison) and R2.2 (state the learning-curve gap "
        "explicitly). Neither requires new computation.")
    add("")
    return "\n".join(L)


def journal_fit(s: Study) -> str:
    L: list[str] = []
    add = L.append
    add("# Journal fit assessment")
    add("")
    add(f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}.*")
    add("")
    add("**No claim is made that any journal will accept this manuscript.** Acceptance depends on "
        "editorial judgement and on reviewers who may not value a negative result. What follows "
        "is a scope and methodology assessment only.")
    add("")
    add("Format requirements below were **not** verified against current author guidelines and "
        "are marked accordingly; confirm each against the publisher's guide for authors before "
        "submission.")
    add("")

    venues = [
        {
            "name": "Transportation Research Part C: Emerging Technologies",
            "scope": "Strong. The journal's stated focus is scholarly research on the development, "
                     "application and implications of emerging transportation technologies, with "
                     "interest in implications for planning, operation and control rather than in "
                     "technologies themselves. A controlled evaluation of a learned controller's "
                     "operational implications sits squarely inside that framing.",
            "method": "Strong. Simulation-based evaluation of learning-based control is core "
                      "subject matter.",
            "transport": "Strong. Metro dispatch, real ridership data, operational baselines.",
            "negative": "Moderate. The journal publishes methodological evaluation; a negative "
                        "result with an identified mechanism and a placebo control is a plausible "
                        "fit, but editors may still prefer a positive contribution.",
            "concerns": "That no learned arm beats a heuristic; that the environment's crowding "
                        "components are unvalidated; that a single training configuration "
                        "confounds the interpretation.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED — confirm word limit, structure and "
                      "reference style against the current guide for authors]`",
        },
        {
            "name": "Transportation Research Part B: Methodological",
            "scope": "Moderate. Part B is the natural home for the metro-scheduling literature "
                     "this paper builds on, including demand-uncertain metro rescheduling and "
                     "MILP scheduling formulations. However Part B favours methodological and "
                     "mathematical contribution, and this paper's contribution is experimental "
                     "rather than a new formulation.",
            "method": "Moderate. The statistical design is rigorous, but no new model or "
                      "algorithm is proposed.",
            "transport": "Strong.",
            "negative": "Lower. A methodological journal expecting a formulation or theorem may "
                        "regard a controlled empirical falsification as out of scope.",
            "concerns": "Absence of a methodological contribution in the Part B sense.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED]`",
        },
        {
            "name": "Journal of Rail Transport Planning & Management",
            "scope": "Strong and specific. Rail operations planning and management is the exact "
                     "domain, and the journal publishes applied rail studies including "
                     "learning-based dispatching.",
            "method": "Good. Applied simulation studies are standard.",
            "transport": "Very strong — rail-specific.",
            "negative": "Higher than the Part B/C pair. A smaller, domain-focused venue is more "
                        "likely to value a careful negative operational result.",
            "concerns": "Lower profile than Part B/C; may request more operational grounding.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED]`",
        },
        {
            "name": "European Journal of Operational Research",
            "scope": "Moderate. The convex-loss/newsvendor framing connects to OR theory, and EJOR "
                     "publishes applied OR in transport.",
            "method": "Moderate. EJOR would likely want the OR contribution foregrounded — the "
                      "convex service loss and the MILP baselines — rather than the RL comparison.",
            "transport": "Moderate; EJOR is domain-general.",
            "negative": "Moderate.",
            "concerns": "The paper's centre of gravity is experimental design, not optimisation.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED]`",
        },
        {
            "name": "Computers & Industrial Engineering",
            "scope": "Moderate. Broad enough to accept a simulation-based operational study.",
            "method": "Adequate.",
            "transport": "Weaker fit; less transport-specialised readership.",
            "negative": "Moderate.",
            "concerns": "The transport-specific contribution may be under-appreciated.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED]`",
        },
        {
            "name": "Transportation Science",
            "scope": "Lower. Strongly methodological and theory-oriented; an empirical "
                     "falsification without a new model is an unlikely fit.",
            "method": "Lower for this paper.",
            "transport": "Strong.",
            "negative": "Low.",
            "concerns": "Absence of theoretical contribution.",
            "type": "Research Paper",
            "format": "`[FORMAT REQUIREMENTS NOT VERIFIED]`",
        },
    ]

    for venue in venues:
        add(f"## {venue['name']}")
        add("")
        add("| Dimension | Assessment |")
        add("|---|---|")
        add(f"| Scope fit | {venue['scope']} |")
        add(f"| Methodological fit | {venue['method']} |")
        add(f"| Transportation relevance | {venue['transport']} |")
        add(f"| Negative-result compatibility | {venue['negative']} |")
        add(f"| Likely reviewer concerns | {venue['concerns']} |")
        add(f"| Article type | {venue['type']} |")
        add(f"| Format requirements | {venue['format']} |")
        add("")

    add("## Recommendation")
    add("")
    add("| Role | Venue | Reasoning |")
    add("|---|---|---|")
    add("| **PRIMARY** | *Transportation Research Part C: Emerging Technologies* | The stated "
        "scope — implications of emerging technologies for the operation and control of "
        "transportation systems — matches a controlled evaluation of a learned dispatch "
        "controller more closely than any alternative. The readership is the one most likely to "
        "act on the finding that a tuned heuristic outperformed the learned arms. |")
    add("| **BACKUP** | *Transportation Research Part B: Methodological* | Home to the closest "
        "prior work on metro scheduling under uncertain demand, so the contribution is legible to "
        "its reviewers, though the absence of a new formulation is a real risk. |")
    add("| **SAFE** | *Journal of Rail Transport Planning & Management* | Rail-specific and "
        "applied, and the venue most likely to value a careful negative operational result on its "
        "own terms. |")
    add("")
    add("Venues are ranked by scope fit and by the likelihood that reviewers will engage with a "
        "controlled negative result — **not** by impact factor or quartile. A journal chosen "
        "solely for its ranking would be the wrong home for this paper.")
    add("")
    add("## Framing advice, whichever venue")
    add("")
    add("1. Lead with the **design**, not the algorithm. The contribution is the control "
        "structure that makes the comparison interpretable.")
    add("2. Report the heuristic comparison **early and plainly**. A reviewer who finds it in a "
        "figure after the text has passed over it will distrust the rest.")
    add("3. State the training-budget confound in the abstract, not only in the limitations.")
    add("4. Do not describe the environment as a validated digital twin.")
    add(f"5. Keep the primary result exactly as measured: {s.primary['mean_difference']:+.6f}, "
        f"Holm-adjusted p = {s.primary['p_adjusted']:.6f}, H1 = {s.verdict()}.")
    add("")
    return "\n".join(L)


def main() -> int:
    s = Study()
    (DOCS / "paper_peer_review.md").write_text(peer_review(s), encoding="utf-8")
    (DOCS / "paper_journal_fit.md").write_text(journal_fit(s), encoding="utf-8")
    print("wrote docs/paper_peer_review.md and docs/paper_journal_fit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
