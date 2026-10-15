# Panel demonstration guide

Operational checklist for demonstrating the prototype. The word-by-word script is
`docs/panel_demo_script.md`.

## Before the session

```bash
cd metro-rl-research
.venv/Scripts/python.exe scripts/check_both_apps.py     # verifies both apps and integrity
```

Expect **8/8 checks pass**. If it reports a port already occupied, stop the earlier Streamlit —
Streamlit exits rather than sharing a port.

Then launch both:

```bash
streamlit run app.py --server.port 8501            # research dashboard
streamlit run product/app.py --server.port 8502    # product
```

Open **http://localhost:8502**, enable **Panel demo mode**, and press **Run** once so the line has
trains moving before the panel sees it.

## Recommended settings

| Setting | Value | Why |
|---|---|---|
| Strategy | `ppo_point` | The strongest learned arm; the honest comparison point |
| Scenario | Demand surge | Where capacity pressure is visible |
| Policy instance | 101 | Any seed works; fix one for a repeatable demonstration |
| Simulation day | 7 | Reproduces identically every time |

## The route

Panel demo mode walks these in order with a progress bar:

1. Home — the problem and live state
2. Network — where passengers are waiting
3. Forecast — what demand is expected
4. Dispatch — the recommendation and why
5. Scenario simulator — what happens if we act
6. Strategy comparison — how strategies compare
7. Research results — what the study concluded
8. Mechanism — why the controller behaved that way
9. About — limitations and next steps

## Points to make

- **The prototype does not control trains.** Say it at step 1; the banner is on screen.
- **Research and demonstration values are separated by badge**, and never mixed.
- **The headline finding is negative and we present it**: H1 = NOT SUPPORTED, and no learned
  strategy beat a tuned heuristic.
- **The mechanism is the interesting part**: the controller withheld service under uncertainty,
  while measured service loss is convex — which argues it should have done the opposite.

## Things to avoid

- Do not claim the uncertainty-aware method improves scheduling. It did not.
- Do not present the Follow-up page as a result. It is incomplete by design of what happened.
- Do not describe simulated magnitudes as predictions about a real metro.
- Do not switch strategy mid-page while a simulation is mid-run; press Reset first.

## Recovery

| Problem | Fix |
|---|---|
| Port will not bind | Stop the earlier Streamlit process |
| A page shows an error | Move on; other pages are unaffected |
| A panel says NOT AVAILABLE | Expected when an artefact is absent; say so, do not improvise |
| Simulation seems stuck | Press Reset, then Run |
| Slow first load | The first policy load is the slow one; press Run before the panel arrives |

## After the session

Both applications can be stopped with Ctrl-C. Stopping the product has no effect on the research
dashboard or on any research artefact — `scripts/check_both_apps.py` verifies this.
