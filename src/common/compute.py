"""Thread configuration for parallel workers.

Two reasons this exists, and the second is the one that matters.

**Oversubscription.** The sweep runs one process per core, and PyTorch defaults to using most of
the machine's cores *inside each process*. On a 12-core machine that is 11 workers each claiming 8
threads: 88 threads contending for 12 cores. The networks here are two 64-unit layers, far too
small for intra-op parallelism to pay for its own synchronisation, so the contention is pure loss.

**Reproducibility.** Torch's CPU intra-op parallelism splits reductions across threads, and
floating-point addition is not associative, so the number of threads can change results in the low
bits. Thread count is a property of the machine, not of the experiment. Pinning it to one makes a
run reproducible on a machine with a different core count -- which is what protocol section 9 asks
for, and what a reader trying to rerun this would otherwise be unable to obtain.

Call `single_threaded()` at the top of every worker, before importing or touching torch tensors.
"""

from __future__ import annotations

import os

__all__ = ["single_threaded"]

#: Environment variables the numeric stacks read at import time. Setting them after the library is
#: imported has no effect, which is why this runs first thing in a worker process.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def single_threaded() -> None:
    """Pin this process to one compute thread across torch, BLAS and OpenMP.

    Safe to call more than once. Never call it in the parent process of a pool that also does
    heavy single-process numeric work -- it is for workers.
    """
    for variable in _THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # RuntimeError: interop threads can only be set once, before any parallel work starts.
        # Both cases are non-fatal -- the environment variables above still bound the BLAS pools.
        pass
