#!/bin/bash
# Output-length calibration for 34-l40s-gemma4-26b-awq-customer-replay: run ONCE, by hand, on the
# toolbox, before the study (and again whenever the template, the corpus, the mix, the seed or the
# users change: caps.py apply refuses a dataset that is not the calibrated one).
#
#   bash calibrate_caps.sh            # ~1.5-3 h; CC_* below override the defaults
#
# 1. Starts vLLM with calibration.params.env (the compose's numerics, more sequences and batched
#    tokens for speed) through apply_config.sh.
# 2. Runs the calibration pod (07-calibration_pod_template.yaml): the dataset rendered as the AIPerf
#    Job renders it (same users: USERS + 20 %, at least + 10, as render_job.sh), then caps.py measure
#    on every request a ramp of CC_USERS over CC_RAMP_S s can reach, uncapped.
# 3. Copies caps.json.gz to CC_OUT (next to the template on the toolbox, where run_test.sh reads it)
#    and deletes the pod. The pod's log is printed as it runs.
# Exit codes: 0 caps written; 1 calibration failed; 2 invalid input; 3 vLLM did not start.
set -euo pipefail
K8S=${K8S:-/work/vllm-benchmark/studies/34-l40s-gemma4-26b-awq-customer-replay/k8s}
D=/work/customer-data/34-l40s-gemma4-26b-awq-customer-replay
NS=llm-l40s
KUBE_CONTEXT=${KUBE_CONTEXT:-default}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }
USERS=${CC_USERS:-200}          # the workflow's RT_USERS
RAMP_S=${CC_RAMP_S:-6000}       # the workflow's RT_RAMP_S
CONC=${CC_CONCURRENCY:-128}     # calibration.params.env's MAX_NUM_SEQS
TEMPLATE=${CC_TEMPLATE:-$D/dataset_u000.template.jsonl}
OUT=${CC_OUT:-$D/caps.json.gz}
POD=aiperf-calibration
t() { echo "$(date -u +%T) calibrate_caps: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

[ -s "$TEMPLATE" ] || die "dataset template $TEMPLATE missing on the toolbox (../load/README.md)"
EXTRA=$((USERS / 5)); [ "$EXTRA" -ge 10 ] || EXTRA=10
RENDERED=$((USERS + EXTRA))
# The AIPerf Job must not be running: it shares vLLM and the results volume.
[ -z "$(kubectl -n $NS get jobs -l app=aiperf-replay -o name 2>/dev/null)" ] || die "an AIPerf Job is in $NS: stop the study first"

t "1/3 vLLM with calibration.params.env"
PARAMS=$K8S/calibration.params.env RENDERED=$(mktemp /tmp/calibration-sts-XXXXXX.yaml) bash "$K8S/apply_config.sh" || die "apply_config.sh failed" 3

t "2/3 calibration pod: $RENDERED users rendered, ramp $USERS users over $RAMP_S s, concurrency $CONC"
TMP=$(mktemp -d /tmp/calibrate-34-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
gzip -c "$TEMPLATE" > "$TMP/template.jsonl.gz"
kubectl -n $NS delete pod $POD --ignore-not-found --wait=true
kubectl -n $NS create configmap aiperf-calibration --from-file=template.jsonl.gz="$TMP/template.jsonl.gz" \
  --from-file=render_dataset_it.py="$K8S/render_dataset_it.py" --from-file=build_corpus.py="$K8S/build_corpus.py" \
  --from-file=mix.json="$K8S/mix.json" --from-file=caps.py="$K8S/caps.py" \
  --dry-run=client -o yaml | kubectl apply -f -
sed -e "s|@USERS_RENDERED@|$RENDERED|g" -e "s|@USERS@|$USERS|g" -e "s|@RAMP_S@|$RAMP_S|g" \
  -e "s|@CONCURRENCY@|$CONC|g" "$K8S/07-calibration_pod_template.yaml" > "$TMP/pod.yaml"
! grep -q '@[A-Z_]*@' "$TMP/pod.yaml" || die "unrendered token in the calibration pod"
kubectl apply -f "$TMP/pod.yaml"
kubectl -n $NS wait --for=condition=Ready pod/$POD --timeout=600s >/dev/null || die "calibration pod not ready" 1

# Follow the log until the pod says it is done, or stops.
kubectl -n $NS logs -f $POD &
LOGS=$!
while true; do
  PHASE=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null || echo Gone)
  if kubectl -n $NS logs $POD --tail=5 2>/dev/null | grep -q 'CALIBRATION DONE'; then break; fi
  if [ "$PHASE" != Running ]; then
    kill $LOGS 2>/dev/null || true
    die "calibration pod $PHASE before CALIBRATION DONE (log above)" 1
  fi
  sleep 30
done
kill $LOGS 2>/dev/null || true

t "3/3 copying the caps to $OUT"
mkdir -p "$(dirname "$OUT")"
kubectl -n $NS cp "$POD:/benchmarks/caps/caps.json.gz" "$OUT.tmp" -c calibration
python3 -c "import gzip, json, sys; c = json.load(gzip.open(sys.argv[1], 'rt')); print('caps: %d measured of %d reachable, %d errors, sha256 %s' % (c['measured'], c['reachable'], sum(c['errors'].values()), c['dataset_sha256'][:12]))" "$OUT.tmp" \
  || die "copied caps file unreadable" 1
mv "$OUT.tmp" "$OUT"
kubectl -n $NS delete pod $POD --wait=false
kubectl -n $NS delete configmap aiperf-calibration --ignore-not-found
t "done: $OUT (the study's RunTest reads it; vLLM is left on the calibration configuration)"
