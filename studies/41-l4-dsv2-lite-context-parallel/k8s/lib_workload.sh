# The AIPerf 0.11.0 workload of study 41, shared by render_job.sh (the study's users ramp)
# and tests/dry_run_aiperf.sh, so both always send the same requests. Sourced; defines
# variables only. Flag names checked against the aiperf 0.11.0 wheel
# (aiperf/config/flags/cli_config.py) on 2026-10-22.
# shellcheck shell=bash disable=SC2034  # variables read by the scripts that source this file
#
# Synthetic multi-turn LONG-CONTEXT chat sessions with a pool of document-like shared prefixes:
#   --prefix-prompt-pool-size 32 --prefix-prompt-length 8192
#       32 distinct ~8k-token prefixes; each session's FIRST user message starts with one of
#       them, drawn at random (dataset/composer/synthetic.py). Not
#       --shared-system-prompt-length (one prefix for everyone, trivial for the prefix cache).
#   --synthetic-input-tokens-mean 2048 (stddev 512), --output-tokens-mean 256 (stddev 64)
#       per turn. --extra-inputs ignore_eos:true pins the output length to the requested
#       max_completion_tokens, so TPOT and token throughput compare configurations at equal
#       reply lengths.
#   --conversation-turn-mean 5 (stddev 1): AIPerf takes the ceiling of a normal sample
#       (common/random_generator.py sample_positive_normal_integer), so ~5.5 turns on
#       average, 8 at +2.5 sigma. Every turn resends the whole history.
#   --conversation-turn-delay-mean 10000 (stddev 3000) ms of user think time.
#   Context per sequence at turn k ~ 8192 + k x 2048 + (k - 1) x 256: ~19.7k at turn 5,
#   ~26.4k at turn 8, always below --max-model-len 32768 (10+ turns, > 4 sigma, would reach
#   ~31k). This is what makes the KV cache the bottleneck: TP4 holds 374,192 tokens, i.e.
#   ~19-26 in-flight sequences of 14-20k, so a 0 -> 96 users ramp saturates it mid-ramp;
#   DCP4 holds 1,524,864.
#   --tokenizer-trust-remote-code: as vLLM's --trust-remote-code for this checkpoint.
# Closed loop (render_job.sh): --concurrency N --concurrency-ramp-duration RAMP. In 0.11.0
# --concurrency is a SESSION limit (timing/concurrency.py: "A slot is acquired on first turn
# and released on final turn"), held during the think time too: N = concurrent users.
#   --num-dataset-entries 4000 unique sessions. NOT --conversation-num (in 0.11.0 also a stop
#   condition). Sessions used by a ramp 0 -> N over RAMP s ~ (N / 2) x RAMP / session length;
#   a session lasts ~5.5 x (10 s think + 5-15 s service) ~ 80-140 s, so 0 -> 96 over 3600 s
#   uses ~1200-2200 sessions: 4000 leaves margin, no session repeats. The calibration run
#   measures the dataset-generation time (4000 x ~5.5 x 2k tokens ~ 45M synthetic tokens).
WORKLOAD_ARGS="--endpoint-type chat --streaming --tokenizer-trust-remote-code --prefix-prompt-pool-size 32 --prefix-prompt-length 8192 --synthetic-input-tokens-mean 2048 --synthetic-input-tokens-stddev 512 --output-tokens-mean 256 --output-tokens-stddev 64 --conversation-turn-mean 5 --conversation-turn-stddev 1 --conversation-turn-delay-mean 10000 --conversation-turn-delay-stddev 3000 --extra-inputs ignore_eos:true"
# Unique sessions (--num-dataset-entries) and seed (dataset and timing) of the measured
# run; the 60 s warm-up uses few sessions and another seed, so it does not pre-load the
# measured sessions (or, with another seed, the measured prefix pool) into the prefix cache.
MEASURED_ENTRIES=4000
WORKLOAD_SEED=40
WARMUP_ENTRIES=50
WARMUP_SEED=7
