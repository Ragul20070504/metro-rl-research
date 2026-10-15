# Panel demonstration script — 8 minutes

Turn on **Panel demo mode** in the sidebar. It presents the nine steps below in order with a
progress bar, so you never hunt for a page in front of the panel.

**Before you start**

```bash
streamlit run app.py --server.port 8501            # research dashboard
streamlit run product/app.py --server.port 8502    # this product
```

Open **http://localhost:8502**. Set the sidebar to strategy `ppo_point`, scenario **Demand
surge**, simulation day **7**. Press **Run** once so the line has trains moving.

> If a port refuses to bind, an earlier Streamlit is still running. Stop it first — Streamlit
> exits rather than sharing a port.

---

## Step 1 — Home · the problem *(60 s)*

**Click:** Home (step 1).

**Panel sees:** metric cards — passengers waiting, forecast, uncertainty, trains available,
capacity pressure — and the ten-step operator workflow.

**Say:** "Metro dispatchers decide repeatedly, under time pressure, how many trainsets to release.
Release too few and passengers are left on platforms; too many and energy and crew cost rise.
Demand is known only as a forecast. This prototype presents that decision. Note the banner: it
does not control trains, and every recommendation needs operator approval."

**Expected:** live values for the selected day; capacity pressure LOW/MODERATE/HIGH.

---

## Step 2 — Network · where the pressure is *(45 s)*

**Click:** Network.

**Panel sees:** per-station queue bars, station table, trains in service.

**Say:** "The queue is not uniform. Dispatch decisions have to account for where passengers are
waiting, not just how many. This is a schematic — station order is real, the geometry is not."

---

## Step 3 — Forecast · what is coming *(60 s)*

**Click:** Forecast.

**Panel sees:** forecast mean, dispersion, 80% bounds; forecast-model quality below.

**Say:** "The forecast comes from a quantile model frozen before any policy was trained, so no
controller can influence its own uncertainty signal. The interval is the dispersion the
controller can see."

**Point out:** switch the strategy to `ppo_point` and the uncertainty reads **NOT AVAILABLE** —
that arm's uncertainty channel is zeroed by design. That is the experimental contrast, live.

---

## Step 4 — Dispatch · the recommendation *(90 s)*

**Click:** Dispatch.

**Panel sees:** current operating state, a large recommendation card, "Why this recommendation?",
and expected effect.

**Say:** "The action comes from the trained policy itself, clipped to the feasible set exactly as
the research environment would execute it. The reasons describe the observable state — queue,
forecast, availability, capacity. They are *not* a claim about what the network computed
internally; we are careful about that distinction."

**Expected:** an action inside the feasible set, and the operator-approval warning.

---

## Step 5 — Simulator · what happens if we act *(75 s)*

**Click:** Scenario simulator → set 60 epochs → **RUN SIMULATION**.

**Panel sees:** before/after table — left-behind fraction, waiting time, inductions, energy, cost.

**Say:** "This is a product demonstration scenario, not a confirmatory experiment. Anything the
simulator cannot compute reads NOT AVAILABLE rather than being filled in."

---

## Step 6 — Strategy comparison · how the options rank *(75 s)*

**Click:** Strategy comparison. Metric `left_behind_fraction`, scenario `surge`.

**Panel sees:** all nine strategies ranked, badged **CONFIRMATORY**.

**Say:** "These are measured research results, not this session's simulation — the badge marks the
difference, and the two are never mixed. The honest headline is here: **no learned strategy beat
the tuned threshold heuristic in any scenario.** We report that rather than hiding it."

**Expected:** heuristic and static MILP at the top; `ppo_uq` near the bottom.

---

## Step 7 — Research results · what the study concluded *(75 s)*

**Click:** Research results.

**Panel sees:** a large **H1 = NOT SUPPORTED** banner, effect size, CI, adjusted p, placebo.

**Say:** "The registered hypothesis was that calibrated forecast uncertainty would reduce the
left-behind fraction. It did the opposite, significantly: +0.0137, confidence interval excluding
zero, Holm-adjusted p = 0.0077. Critically, the uncertainty arm was also worse than a **placebo**
carrying a fake uncertainty signal of identical distribution — so this is not just the cost of
extra inputs."

---

## Step 8 — Mechanism · why *(90 s)*

**Click:** Mechanism.

**Panel sees:** dispatch response per scenario, matched-volume comparison, convex-loss figure.

**Say:** "Given genuine uncertainty the controller dispatched about 11% fewer trains. Given the
placebo it did not. So it was responding to real forecast dispersion — by withholding service.
Now the important part: measured service loss is *convex* — flat while capacity is slack, steep
once it binds. Under a convex loss, uncertainty should motivate hedging **upward**. The controller
went the other way. That points to a policy-learning limitation, not to uncertainty being
worthless."

---

## Step 9 — About · limits and next steps *(45 s)*

**Click:** About.

**Say:** "To be clear about what this is not: not connected to real infrastructure, not safety
certified, not autonomous, and simulation-based — several simulator components that drive the
endpoint are not externally validated. The next experiment is specified and pre-registered: a
training-budget study to separate an information effect from an optimisation one. It was stopped
incomplete, and the Follow-up page reports it as incomplete rather than as evidence."

---

## Likely questions

**"Why present a negative result?"** It is pre-registered and placebo-controlled. A less
controlled comparison would very likely have produced a favourable, unreplicable result. The
design is the contribution.

**"Is the RL useless then?"** In this setting it did not beat a tuned heuristic, and we say so.
The convex-loss argument says uncertainty *is* decision-relevant here, so the more likely reading
is that the controller was undertrained rather than that the information is worthless. The
follow-up would settle it.

**"Could you make it work by tuning?"** Possibly — and doing so after seeing the result would
convert a pre-registered study into a post-hoc one. That is why the follow-up is registered
instead.

**"Can this be deployed?"** No. It is a decision-support prototype requiring operator approval,
built on a simulator whose crowding components are not externally validated.

## Timing

| Step | Minutes |
|---|---|
| 1–3 Problem, network, forecast | 2:45 |
| 4–5 Dispatch and simulation | 2:45 |
| 6–8 Comparison, result, mechanism | 4:00 |
| 9 Limits | 0:45 |
| **Total** | **~8 min** (trim step 2 and step 5 for a 5-minute version) |
