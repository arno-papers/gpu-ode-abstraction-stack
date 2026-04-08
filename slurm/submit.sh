#!/bin/bash
# Wrapper: run sbatch and strip ";cluster" suffix from --parsable output
sbatch "$@" 2>&1 | sed 's/;.*//'
