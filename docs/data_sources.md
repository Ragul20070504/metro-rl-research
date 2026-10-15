# Data-Source Plan

**Principle (brief §3):** authoritative public sources only; no silent Kaggle substitution when an
official source exists; every substitution documented.

## Verification status

Every identifier below was **probed directly on 2026-08-26** by `scripts/probe_sources.py`, which
checks that each source is readable, captures its field list and temporal coverage, and confirms the
modelled line actually appears in it. Machine-readable evidence: `results/source_probe.json`.
Re-run it before M3 ingestion and whenever a source is added.

> **The M0 audit recorded two identifiers that do not work.** Both were taken from portal search
> results, which is *not* evidence a resource can be read. They are corrected below and retained in
> the probe list so the error stays visible rather than being quietly overwritten:
>
> | Recorded in M0 | Reality | Replacement |
> |---|---|---|
> | `4apg-4kt9` — Customer Journey-Focused Metrics | **HTTP 403** from the tabular API | **`r7qk-6tcy`** — and it covers 2015→2026, far more than the 2020–2024 originally assumed |
> | `fgm6-ccue` — GTFS Static | **HTTP 403**; the portal entry's only attachment is an overview PDF, the feed is not served through the tabular API | **`https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip`** — 5.6 MB, no API key |

`download_date`, row counts and file hashes are written by the loader into
`data/raw/<dataset_id>/MANIFEST.json` at ingestion. They are never typed by hand.

---

## 1. Demand — three datasets, one continuous series

**Observed fact.** The three MTA hourly-ridership datasets share an **identical schema** and their
coverage is contiguous, giving continuous hourly, station-complex-resolved demand from
**2017-01-13 to 2026-08-13** — about 9.5 years:

| dataset_id | Title | Verified coverage |
|---|---|---|
| `t69i-h2me` | MTA Subway Hourly Ridership: 2017–2019 | 2017-01-13T00:00 → 2019-12-31T23:00 |
| `wujg-7c2s` | MTA Subway Hourly Ridership: 2020–2024 | 2020-01-01T00:00 → 2024-12-31T23:00 |
| `5wq4-mkjj` | MTA Subway Hourly Ridership: Beginning 2025 | 2025-01-01T00:00 → 2026-08-13T23:00 |

**Fields (all three, confirmed):** `transit_timestamp, transit_mode, station_complex_id,
station_complex, borough, payment_method, fare_class_category, ridership, transfers, latitude,
longitude, georeference`.

**Why this matters more than it looks.** The M0 audit treated post-2020 non-stationarity as a
threat to be managed (`research_audit.md` §G-12, risk R-15). With `t69i-h2me` and `5wq4-mkjj`
available, it becomes something that can be *designed around*: the study window can sit wholly
inside one regime, and the pre-pandemic period is available as an out-of-regime robustness check
rather than as contamination. **Amendment to protocol §3.1 recorded — see `docs/decisions.md` D-11.**

**Volume is tractable.** Filtered to the modelled line's 24 station complexes, `wujg-7c2s` holds
**6,893,715 rows** — a filtered Socrata pull, not a bulk download. (An unfiltered `count(*)` over the
whole table times out, which is itself a useful fact about how this source must be queried.)

**Attribution:** Metropolitan Transportation Authority — <https://www.mta.info/open-data>.

**Preprocessing:** filter `transit_mode = subway`; restrict to the line's complexes; sum `ridership`
(and separately `transfers`) over `payment_method` and `fare_class_category` to one value per
(complex, hour); cast the numeric fields, which arrive as **strings**; localise timestamps and
resolve DST; run the §3.2 quality suite.

**Known limitations — load-bearing, unchanged by the new sources:**

- **Hourly only.** The 5-minute decision interval needs a within-hour disaggregation assumption (A02).
- **Entries and transfers only. No exits, no destinations, no OD matrix.** Alighting stays synthetic
  (A03) and remains the dominant simulator sensitivity.
- **Station-complex granularity**, not platform or direction. Direction split is an assumption.
- Values are MTA *estimates* derived from fare transactions, not raw turnstile counts.

---

## 2. Network and timetable

### 2.1 `mta_subway_stations` — `39hk-dx4f`

<https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f>

**Verified:** the modelled line is served by exactly **24 station complexes**, identified by a
standalone `L` token in `daytime_routes`. (A naive `like '%L%'` filter is unsafe on route labels;
`scripts/probe_sources.py` requires a standalone token and re-derives the set on every run, flagging
any change.)

Complex ids: 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 131, 133, 134, 135, 136, 137,
138, 601, 602, 618, 621, 629, 630.

Fields: `complex_id, station_id, gtfs_stop_id, division, line, stop_name, borough, cbd,
daytime_routes, structure, gtfs_latitude, gtfs_longitude, north_direction_label,
south_direction_label, ada*, georeference`.

### 2.2 `mta_gtfs_static` — `gtfs_subway.zip`

<https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip> · **verified 5,610,175 bytes, no API key.**

**Role:** inter-station running times (median scheduled segment durations by direction and time
band), scheduled headways, trip counts (which set the fleet-size configuration), and the
`fixed_timetable` baseline.

**Note on access.** The `mta.info/developers` distribution requires registering for an API key. The
S3 endpoint above does not, which keeps a fresh clone reproducible without credentials — the reason
it is preferred. The feed is a **snapshot**, so its hash and retrieval date must be pinned in the
manifest; older editions are not archived on this endpoint.

**Limitations:** scheduled, not realised; the base feed excludes short-term service changes.

---

## 3. Simulator validation and calibration

This section is materially stronger than the M0 audit assumed. **One M0 claim was simply wrong:**
the audit stated that realised train movements are "not retained as an open historical archive". They
are — see §3.3.

### 3.1 `mta_mdbf` — `e2qc-xgxs` · failure-rate calibration

Coverage **2015-01 → 2026-06**. Fields: `month, division, car_class, total_miles,
number_of_failures, number_of_cars, mdbf, _12_month_average_mdbf`.

**Verified for the modelled fleet:** car class **R143** appears under **division B**, e.g. 2026-06 —
212 cars, 934,940 total miles, 13 failures, MDBF 71,918, 12-month average MDBF 139,048.

**Role:** this is what turns the train-failure process from *invented* into *rate-calibrated*. The
synthetic per-train hazard is scaled so simulated failures per 1000 train-km match the published
figures for the relevant car class and period. Because `number_of_failures`, `total_miles` and
`number_of_cars` are all published, the calibration target is a directly computed rate rather than a
derived index.

**Limitations:** monthly, by car class — **not per train, not per day, no sensor traces.** It
constrains the aggregate rate only; within-day timing structure stays synthetic (A13).

### 3.2 `mta_cjfm` — `r7qk-6tcy` · excess passenger time

Coverage **2015-01 → 2026-06**, line `L` present. Fields: `month, division, line, period,
num_passengers, additional_platform_time, additional_train_time, total_apt, total_att,
over_five_mins, over_five_mins_perc, customer_journey_time`.

**Role:** the real quantity against which simulated passenger waiting time is checked. **Additional
Platform Time** (excess platform wait vs schedule) is split by `peak` / `offpeak`, which matches the
simulator's time bands directly.

### 3.3 `mta_running_times` — `sp9g-mzjh` · realised running times **(new; corrects the M0 audit)**

Coverage **2019-01 → 2026-06**, line `L` present. Fields include `average_actual_runtime`,
`average_scheduled_runtime`, `_25th/_50th/_75th_percentile_runtime`, `actual_trains`,
`scheduled_trains`, `distance`, `average_speed`, `number_of_stops`, `origin_station_id`,
`destination_station_id`, `direction`, `time_period`, `schedule_day_type`.

**Why this matters.** The M0 audit classified running-time noise (A07) as a simulation assumption
that could not be validated. It can: this dataset publishes the **realised end-to-end runtime
distribution** — not just a mean, but 25th/50th/75th percentiles by direction, day type and time
period. The simulator's running-time noise can be calibrated against a real dispersion instead of
being chosen. Protocol §5.3 is updated accordingly.

It also independently corroborates the network: end-to-end paths report **23 stops** over
**9.84–10.15 miles**, consistent with the 24 complexes derived in §2.1 from a different dataset.

### 3.4 `mta_wait_assessment` — `s666-h6b7` · headway regularity

Coverage **2015-01 → 2026-06**, line `L` present. Fields: `month, division, line, day_type, period,
num_timepoints_passing_wait_assessment, num_sched_timepoints, wait_assessment`.

### 3.5 `mta_service_delivered` — `32ch-sei3` · scheduled vs actual trains **(new)**

Coverage **2015-01 → 2026-06**, line `L` present. Fields: `month, division, line, day_type,
num_sched_trains, num_actual_trains, service_delivered`.

**Role:** validates the `fixed_timetable` baseline's dispatch counts and supplies a real reference
for "dispatch success rate", a metric the protocol requires (§7.4) but for which the M0 audit had no
real comparator.

> **Data-quality finding, recorded now so the pipeline catches it.** The published
> `service_delivered` field appears to be scaled by 1/100: a 2026-06 row reports
> `num_actual_trains = 2028`, `num_sched_trains = 2220` and `service_delivered = 0.00913`, whereas
> 2028 / 2220 = 0.9135. **The pipeline recomputes this ratio from the two counts and does not trust
> the published field**, and `tests/test_data_quality.py` asserts the recomputed value lies in
> [0, 1.05]. Flagged rather than silently corrected.

### 3.6 `mta_terminal_otp` — `f6rf-2a3t` · punctuality **(new)**

Coverage **2015-01 → 2026-06**, line `L` present. Fields: `month, division, line, day_type,
num_on_time_trips, num_sched_trips, terminal_on_time_performance`. Secondary validation target.

---

## 4. Conditional source — real rolling-stock sensor data

### `metropt3` — used only under decision D-04's optional extension

- **Source:** UCI ML Repository #791 (MetroPT-3); also Zenodo; documented in *Scientific Data* (2022).
- **URL:** <https://archive.ics.uci.edu/dataset/791/metropt+3+dataset> · **DOI:** 10.24432/C5VW3R
- **Licence:** **CC BY 4.0** — redistribution and adaptation permitted with attribution.
- **Content:** 1,516,948 rows at 0.1 Hz, Feb–Aug 2020; 15 channels (analogue: TP2, TP3, H1,
  DV_pressure, Reservoirs, Oil_temperature, Motor_current; digital: COMP, DV_electric, TOWERS, MPG,
  LPS, Pressure_Switch, Oil_Level, Caudal_Impulse). Shipped **unlabelled**, with company failure
  reports covering **four air-leak failures**.
- **What it can honestly support:** an estimate of the achievable operating point (precision/recall)
  of a real failure detector on real metro rolling-stock telemetry, which then parameterises the
  noise level of the synthetic availability signal instead of us choosing that level arbitrarily.
- **What it cannot support:** a per-train availability predictor for the modelled fleet. It is **one
  train's Air Production Unit, at Metro do Porto, a different system, with four events.** Using it is
  a **cross-domain transfer assumption** and is labelled as such wherever it appears.

---

## 5. Required data that does NOT exist publicly

Per brief §2F — do not pretend these exist.

| Need | Status | Consequence |
|---|---|---|
| Per-train daily availability / maintenance-hold status for the modelled fleet | **Not public** | The train-health module cannot be validated against reality. Reframed as an availability-information ablation (D-04); no prediction-performance claim is made. |
| Origin–destination matrix for the modelled line | **Not public at required resolution** | Alighting is synthetic (A03) and drives occupancy and unmet demand, i.e. the primary endpoint. Three OD variants are swept; the primary conclusion must survive all three or be reported as OD-dependent. |
| Trip-level realised train movements (individual train timestamps) | **Not public.** GTFS-RT is a live feed with no historical archive. | Only *aggregate* realised statistics are available — but §3.3 shows those include runtime percentiles, which is more than the M0 audit assumed. Per-train trajectories remain unavailable. |
| Traction energy per train-km | **Not sourced** | Assumption A15 stays `unresolved`. Energy is reported but **embargoed from headline claims** until a citable figure with provenance exists. |
| Induction / operating cost per train | **Not sourced** | Assumption A16, same treatment. |
| Train capacity (passengers per trainset) for the modelled fleet | **Not yet sourced with a citable reference** | Assumption A09, swept until resolved. Do not invent a number. |
| Beijing Metro AFC open data (named in manuscript §5) | **Not confirmed public**; access appears case-by-case under agreement | Dropped, not substituted — the NYC sources suffice. MC-02. |
| Live asset telemetry for the modelled line | **Not obtainable** | The system is a calibrated simulator, not an asset-coupled digital twin. MC-04. |
| Platform-level or direction-resolved boardings | **Not public** | Direction split is an assumption, layered on top of A03. |

---

## 6. Ingestion, immutability, and licensing hygiene

- `src/data/download.py` writes only into `data/raw/<dataset_id>/`, then writes `MANIFEST.json`
  (`source_url`, `download_utc`, per-file `sha256`, `row_count`, `column_list`, `socrata_view_id`,
  `dataset_last_updated`, `license`, `attribution`, tool version) and `LICENSE.txt` with the licence
  text as published at download time.
- `data/raw/` is **never** written to again. `tests/test_raw_immutable.py` re-verifies the hashes.
- Raw and processed data files are **git-ignored**; manifests and hashes are **committed**, so a
  fresh clone can verify it re-downloaded identical bytes and fails loudly if upstream has changed.
- **No credentials are required for any source in §§1–3.** `.env.example` documents an optional
  `MTA_API_KEY` used only if the `mta.info` GTFS route is chosen over the credential-free S3 feed.
- Attribution requirements from NY Open Data / MTA are reproduced in the README and in every
  generated report.

---

## 7. Sources

- [MTA Subway Hourly Ridership 2020–2024 (`wujg-7c2s`)](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s)
- [MTA Subway Hourly Ridership Beginning 2025 (`5wq4-mkjj`)](https://data.ny.gov/d/5wq4-mkjj)
- [MTA Subway Hourly Ridership 2017–2019 (`t69i-h2me`)](https://data.ny.gov/d/t69i-h2me)
- [MTA Subway Stations (`39hk-dx4f`)](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f)
- [MTA subway GTFS static feed](https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip) · [MTA Developer Resources](https://www.mta.info/developers)
- [MTA Subway Mean Distance Between Failures (`e2qc-xgxs`)](https://data.ny.gov/Transportation/MTA-Subway-Mean-Distance-Between-Failures-Beginnin/e2qc-xgxs)
- [MTA Subway Customer Journey-Focused Metrics Beginning 2015 (`r7qk-6tcy`)](https://data.ny.gov/d/r7qk-6tcy)
- [MTA Subway Wait Assessment Beginning 2015 (`s666-h6b7`)](https://data.ny.gov/d/s666-h6b7)
- [MTA Subway Service Delivered Beginning 2015 (`32ch-sei3`)](https://data.ny.gov/d/32ch-sei3)
- [MTA Subway End-to-End Running Times Beginning 2019 (`sp9g-mzjh`)](https://data.ny.gov/d/sp9g-mzjh)
- [MTA Subway Terminal On-Time Performance Beginning 2015 (`f6rf-2a3t`)](https://data.ny.gov/d/f6rf-2a3t)
- [MetroPT-3 Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) · [Scientific Data paper](https://www.nature.com/articles/s41597-022-01877-3)
