# Cover letter (draft)

[JOURNAL NAME TO BE CONFIRMED]  
[EDITOR NAME TO BE CONFIRMED]

Dear Editor,

We submit for your consideration our manuscript, *Informative Demand Uncertainty Changes but Does Not Improve a Reinforcement-Learning Metro Dispatch Controller: A Pre-Registered, Placebo-Controlled Evaluation*.

**Problem.** Metro operators must repeatedly decide how many trainsets to release into service under demand that is known only through a forecast. A widely held assumption is that supplying a learned controller with calibrated forecast uncertainty should improve those decisions. That assumption is rarely tested against a control able to separate the information in the uncertainty signal from the mere presence of an additional input.

**What we did.** We conducted a pre-registered, placebo-controlled evaluation in a demand-driven metro simulation environment. Three reinforcement-learning arms share identical architecture, parameter count, observation dimensionality, reward, training budget, seeds and evaluation days, and differ only in what a two-value uncertainty channel carries: zero, a shuffled placebo of identical marginal distribution, or the calibrated forecast dispersion. Fixed-timetable, tuned-heuristic and mixed-integer baselines are evaluated under common random numbers.

**Main finding, stated plainly.** The registered hypothesis was **not supported**. The uncertainty-aware controller left more passengers behind than the point-forecast controller (+0.013742; 95% BCa CI [0.006570, 0.018523]; Holm-adjusted p = 0.007741), and also performed worse than the placebo arm. Mechanism analysis shows the calibrated signal is associated with approximately 11% fewer dispatches while the placebo signal is not, and that measured service loss is strongly convex once capacity binds — so uncertainty is decision-relevant here and the controller's conservative response ran opposite to what convex loss motivates.

**Why it matters.** We report a controlled negative result with an identified mechanism. We also report, prominently rather than in an appendix, that no learned arm outperformed a tuned threshold heuristic. Together these bound what can currently be claimed for learned metro dispatch and identify a specific, testable failure mode. We believe controlled negative evidence of this kind is useful to the field precisely because the design distinguishes competing explanations.

We make no claim that uncertainty-aware reinforcement learning is generally ineffective, and we state the study's principal confound — a single frozen training configuration — explicitly, together with the registered follow-up that would resolve it and which remains incomplete.

**Declarations.** This manuscript is original work, has not been published previously, and is not under consideration elsewhere. `[AUTHORS TO CONFIRM]`  
Funding: `[TO BE CONFIRMED — no funding information is recorded in the project]`  
Competing interests: `[TO BE CONFIRMED BY THE AUTHORS]`  
All code and computational artefacts are version-controlled and are available for editorial and reviewer inspection; see the data and code availability statement.

Thank you for your consideration.

Yours sincerely,

Ragul A (corresponding author `[TO BE CONFIRMED]`)  
on behalf of Ragul A, Raksith N P, Pedakolimi Chandra Arjun  
Supervisor: Dr M N Kavitha  
[AUTHOR AFFILIATION TO BE CONFIRMED]
