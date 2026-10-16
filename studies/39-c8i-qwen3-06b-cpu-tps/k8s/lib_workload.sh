# The AIPerf 0.11.0 workload of study 39, shared by render_job.sh (the study's ramp) and
# ../probe/render_sweep_job.sh (the calibration sweep), so the two always send the same
# requests. Sourced; defines variables only. Flag names checked against the aiperf 0.11.0
# wheel (aiperf/config/flags/cli_config.py).
# shellcheck shell=bash disable=SC2034  # variables read by the scripts that source this file
#
# Synthetic multi-turn chat sessions with a pool of shared prefixes:
#   --prefix-prompt-pool-size 16 --prefix-prompt-length 1536
#       16 distinct ~1.5k-token prefixes; each session's FIRST user message starts with one
#       of them, drawn at random (dataset/composer/synthetic.py). Mutually exclusive with
#       --shared-system-prompt-length (one prefix for everyone), which would make the
#       prefix-cache axis trivial.
#   --synthetic-input-tokens-mean 384 (stddev 96), --output-tokens-mean 192 (stddev 48)
#       per turn. --extra-inputs ignore_eos:true pins the output length to the requested
#       max_completion_tokens: with thinking off, Qwen3-0.6B would otherwise stop at its own
#       EOS, and TPOT / token throughput would mix configurations with different reply
#       lengths.
#   --conversation-turn-mean 4 (stddev 1): 3-6 turns. Every turn resends the whole history,
#       so turns 2+ hit the prefix cache with the session's own earlier context: the main
#       driver of the KV-budget / prefix-caching axes, together with the prefix pool.
#   --conversation-turn-delay-mean 5000 (stddev 2000) ms of user think time: a session's
#       blocks sit idle between turns and survive only if the KV budget keeps them.
#   --num-dataset-entries 8000 unique sessions. NOT --conversation-num: in 0.11.0 that
#       also sets the profiling phase's session count, a stop condition, while this run is
#       duration-bounded. With 100 entries (the default) AIPerf would recycle the same
#       sessions and the prefix cache would hit whole conversations, which no real
#       workload does. 8000 sessions x ~4 turns = ~32k requests before any repeat; a
#       0 -> R ramp over D s sends R x D / 2 (the placeholder 0 -> 20 req/s over 3600 s =
#       36k, more than 32k: sessions repeat only if the ramp runs to its end, which
#       run_test.sh logs as an anomaly; the watchdog normally ends the run well before).
#   Context per running sequence: ~1.5k + up to 6 x (384 + 192) ~ 5k tokens, worst case
#   ~10k, below --max-model-len 16384.
# Rate semantics (timing/strategies/request_rate.py): --request-rate counts requests
# (turns), and a queued continuation turn takes priority over a new session at each
# arrival, so the new-session rate is below R once sessions are in flight.
WORKLOAD_ARGS="--endpoint-type chat --streaming --prefix-prompt-pool-size 16 --prefix-prompt-length 1536 --synthetic-input-tokens-mean 384 --synthetic-input-tokens-stddev 96 --output-tokens-mean 192 --output-tokens-stddev 48 --conversation-turn-mean 4 --conversation-turn-stddev 1 --conversation-turn-delay-mean 5000 --conversation-turn-delay-stddev 2000 --extra-inputs ignore_eos:true"
# Unique sessions (--num-dataset-entries) and seed (arrivals and dataset) of the measured
# run; the 60 s warm-up uses few sessions (dataset generation time) and another seed, so
# it does not pre-load the measured sessions into the prefix cache.
MEASURED_ENTRIES=8000
WORKLOAD_SEED=39
WARMUP_ENTRIES=100
WARMUP_SEED=7
