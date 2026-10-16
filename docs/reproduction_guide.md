# Reproduction and resumption

Use the current [README](../README.md) for the tested Linux installation and stored-results
checks, and the [script guide](../scripts/README.md) for supported commands.

The [replication execution report](replication_execution.md) explains the active experiment,
frozen inputs, logs, and how to resume after interruption. Inspect live processes before
starting a runner. Keep its ignored data/models/results directory; Git alone is insufficient.

Stored-cell reanalysis is available. Exact historical simulation replay still requires the
missing original models, inputs and source history; see [submission readiness](SUBMISSION_READINESS.md).
A passing dependency check does not establish complete experiment reproduction.

The superseded Windows-era guide is preserved in [the historical archive](archive/reproduction_guide.md).
