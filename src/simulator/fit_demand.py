"""Fit the simulator's ground-truth demand process to observed ridership (M5).

Fitted on the **training split only**. Using validation or test days would leak
evaluation-period information into the environment the policy trains in -- a subtler leak than
showing the agent future demand, but just as disqualifying.

Run::

    python -m src.simulator.fit_demand
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.common.logging import get_logger, setup_logging
from src.common.paths import PROCESSED_DIR
from src.simulator.demand import DAY_TYPES, fit_demand_process
from src.simulator.network import Network

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="split to fit on (never test)")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="fit_demand.log")

    if args.split == "test":
        raise SystemExit(
            "refusing to fit the ground-truth demand process on the test split: that would "
            "leak evaluation-period information into the training environment"
        )

    network = Network.load()
    demand = pd.read_parquet(PROCESSED_DIR / "demand_hourly.parquet")
    process = fit_demand_process(demand, network.station_ids, split=args.split)
    path = process.save()

    weekday_mean = process.mean["weekday"]
    print(f"stations            : {process.n_stations}")
    print(f"fitted on split     : {args.split}")
    print(f"days                : {process.fitted_on['days']}")
    print(f"date range          : {process.fitted_on['first_date']} .. {process.fitted_on['last_date']}")
    for day_type in DAY_TYPES:
        block = process.mean[day_type]
        share = process.fitted_on["over_dispersed_cell_fraction"][day_type]
        print(f"{day_type:<8} daily total : {block.sum():>10,.0f} boardings/day   "
              f"over-dispersed cells {share:.1%}")
    peak_hour = int(np.argmax(weekday_mean.sum(axis=1)))
    print(f"weekday peak hour   : {peak_hour:02d}:00 "
          f"({weekday_mean[peak_hour].sum():,.0f} boardings/h line-wide)")
    print(f"written             : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
