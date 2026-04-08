#!/bin/bash
# Wrapper: cancel jobs on genius cluster (receives jobids as arguments)
scancel -M genius "$@"
