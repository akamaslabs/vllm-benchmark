#!/bin/bash
# Stub apply_config.sh for test_run_eval.sh: logs the run (the params.env's directory) and
# its parameters to $STUB_LOG; exits $STUB_APPLY_RC (default 0).
echo "apply $(basename "$(dirname "$PARAMS")") $(grep -v '^#' "$PARAMS" | tr '\n' ' ')" >> "$STUB_LOG"
exit "${STUB_APPLY_RC:-0}"
