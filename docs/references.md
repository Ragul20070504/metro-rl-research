# References

**Provenance note.** Every entry below was located by web search during finalisation and its
title and URL verified to exist. Where an entry lacks authors, year or full bibliographic detail,
that is because the search result did not carry it — **not** because it was reconstructed from
memory. No citation here is invented. Before submission each entry must be completed against the
publisher record and, where the argument depends on a specific claim, read in full rather than
cited from its abstract.

Entries marked **[verified-existence]** have had title and venue confirmed; entries marked
**[read]** have additionally been read closely enough to support the specific claim attached to
them. At the time of writing, all entries are **[verified-existence]** only.

---

## RL for railway and metro operations

- *Reinforcement Learning for Scalable Train Timetable Rescheduling with Graph Representation.*
  arXiv:2401.06952. https://arxiv.org/pdf/2401.06952 **[verified-existence]**
- *Solving the train dispatching problem via deep reinforcement learning.* Journal of Rail
  Transport Planning & Management.
  https://www.sciencedirect.com/science/article/pii/S2210970623000264 **[verified-existence]**
- *Adaptive rescheduling of rail transit services with short-turnings under disruptions via a
  multi-agent deep reinforcement learning approach.* Transportation Research Part B.
  https://www.sciencedirect.com/science/article/abs/pii/S0191261524001917 **[verified-existence]**
- *Integrated train timetabling and rolling stock rescheduling for a disturbed metro system: A
  hybrid deep reinforcement learning and adaptive large neighborhood search approach.*
  https://www.sciencedirect.com/science/article/abs/pii/S0360835223007660 **[verified-existence]**
- *A multi-task deep reinforcement learning approach to real-time railway train rescheduling.*
  Transportation Research Part E.
  https://www.sciencedirect.com/science/article/abs/pii/S1366554524004915 **[verified-existence]**
- *A reinforcement learning approach to solving very-short term train rescheduling problem for a
  single-track rail corridor.*
  https://www.sciencedirect.com/science/article/pii/S2210970624000532 **[verified-existence]**

## Optimisation baselines for train scheduling

- *A Survey of Optimization Models for Train Routing and Scheduling.* Transportation Science
  32(4):380. https://pubsonline.informs.org/doi/10.1287/trsc.32.4.380 **[verified-existence]**
- *Real-time train rescheduling optimization with combined cross-line strategies for urban rail
  network.* Transportation Research Part E.
  https://www.sciencedirect.com/science/article/abs/pii/S1366554525002510 **[verified-existence]**

## Uncertainty-aware and risk-sensitive reinforcement learning

- *A Review of Uncertainty for Deep Reinforcement Learning.*
  https://www.researchgate.net/publication/364395789 **[verified-existence]**
- *Estimating Risk and Uncertainty in Deep Reinforcement Learning.* arXiv:1905.09638.
  https://arxiv.org/pdf/1905.09638 **[verified-existence]**
- *Uncertainty-Aware Reinforcement Learning for Risk-Sensitive Player Evaluation.* NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/7f6e51d8298aa01b084b700ab91aff94-Paper-Conference.pdf
  **[verified-existence]**
- *UAMDP: Uncertainty-Aware Markov Decision Process for Risk-Constrained Reinforcement Learning
  from Probabilistic Forecasts.* arXiv:2510.08226. https://arxiv.org/pdf/2510.08226
  **[verified-existence]**
- *Adaptive risk tendency in uncertainty-aware motion planning using risk-sensitive Reinforcement
  Learning.* https://www.sciencedirect.com/science/article/abs/pii/S1474034624005937
  **[verified-existence]**

## Probabilistic demand forecasting for transit

- *Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic Graph Neural
  Networks.* arXiv:2303.04040. https://arxiv.org/pdf/2303.04040 **[verified-existence]**
- *Uncertainty quantification for joint demand prediction of multi-modal ride-sourcing services
  using spatiotemporal Mixture-of-Expert neural network.* Transportation Research Part C.
  https://www.sciencedirect.com/science/article/abs/pii/S0968090X2500511X **[verified-existence]**
- *CONTINA: Confidence Interval for Traffic Demand Prediction with Coverage Guarantee.*
  arXiv:2504.13961. https://arxiv.org/pdf/2504.13961 **[verified-existence]**

## Convex loss under demand uncertainty (the newsvendor connection)

- *The Newsvendor Problem.* Columbia University lecture notes.
  http://www.columbia.edu/~gmg2/4000/pdf/lect_07.pdf **[verified-existence]**
- Van Mieghem & Rudi. *Newsvendor Networks: Inventory Management and Capacity Investment with
  Discretionary Activities.* M&SOM (2002).
  https://www.kellogg.northwestern.edu/faculty/vanmieghem/htm/pubs/vanmieghem_rudi_2002_newsnet_msom.pdf
  **[verified-existence]**
- *Risk-Adaptive Approaches to Stochastic Optimization: A Survey.* arXiv:2212.00856.
  https://arxiv.org/pdf/2212.00856 **[verified-existence]**

## Reproducibility, seed variance and evaluation methodology in deep RL

- Henderson et al. *Deep Reinforcement Learning that Matters.* arXiv:1709.06560.
  https://www.emergentmind.com/papers/1709.06560 **[verified-existence]**
- Colas et al. *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning
  Experiments.* arXiv:1806.08295. https://arxiv.org/pdf/1806.08295 **[verified-existence]**
- *Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control.*
  arXiv:1708.04133. https://arxiv.org/pdf/1708.04133 **[verified-existence]**
- *Deterministic Implementations for Reproducibility in Deep Reinforcement Learning.*
  arXiv:1809.05676. https://arxiv.org/pdf/1809.05676 **[verified-existence]**

## Digital twins in rail

- *Digital twin in railway industry: a bibliometric analysis and systematic review.*
  https://www.tandfonline.com/doi/full/10.1080/27525783.2025.2533858 **[verified-existence]**
- *Digital twin in transportation infrastructure management: a systematic review.* Intelligent
  Transportation Infrastructure. https://academic.oup.com/iti/article/doi/10.1093/iti/liad024/7370943
  **[verified-existence]**
- *A WebGIS-based digital twin platform for intelligent operation and maintenance of rail transit
  infrastructure.* Expert Systems with Applications.
  https://www.sciencedirect.com/science/article/abs/pii/S0957417425027964 **[verified-existence]**

---

## How these are used in the manuscript

Two of these sources carry load in the argument and must be **read**, not merely cited, before
submission:

1. **Colas et al., *How Many Random Seeds?*** — the paper's justification for a seed-level unit of
   analysis and for reporting an underpowered design honestly rests on the same statistical point
   this work makes. If its recommendations differ from what was done here, that difference must be
   discussed rather than glossed.
2. **The newsvendor literature** — the convex-loss argument in section 14 is a restatement of the
   standard newsvendor observation that a convex cost makes expected cost under stochastic demand
   exceed cost at expected demand. The paper should cite this as established rather than present
   it as a novel derivation.

The remaining entries position the work; none of them is load-bearing for a result.
