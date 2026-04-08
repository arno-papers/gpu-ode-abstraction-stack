#!/bin/bash
# Query SLURM job status and map to snakemake's expected values:
#   running, success, or failed
jobid="${1%%;*}"
state=$(sacct -M genius -j "$jobid" --format=State --noheader | head -1 | tr -d ' ')

case "$state" in
    PENDING|RUNNING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED)
        echo running ;;
    COMPLETED)
        echo success ;;
    FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
        echo failed ;;
    "")
        echo running ;;  # job not yet visible in accounting
    *)
        echo failed ;;
esac
