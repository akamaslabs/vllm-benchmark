"""Kernel microbenchmark driver (study 24 prep). Runs on the toolbox pod with nohup.

1. Waits for the fp8-tune pod to print TUNING-DONE, copies /out to $K8S/tuned-configs,
   deletes the pod.
2. For each config: renders study 24's serving template (1P1D) with fixed values, applies
   ConfigMaps + Deployment, waits for the rollout, runs bench_in_pod.py in the engine
   container, saves results and key log lines.
3. Scales vllm-pd to 0.
Results: $OUT/<name>.json, $OUT/<name>.log, $OUT/summary.txt, progress in $OUT/driver.log.
"""
import json, os, re, shutil, subprocess, sys, tarfile, time, uuid, glob

K8S = os.environ.get('KB_K8S', '/work/vllm-benchmark/studies/24-l4-pd-kernels/k8s')
OUT = os.environ.get('KB_OUT', os.path.expanduser('~/study24-kbench'))
NS = 'llm-serving'
BENCH = os.path.join(OUT, 'bench_in_pod.py')
os.makedirs(OUT, exist_ok=True)


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(os.path.join(OUT, 'driver.log'), 'a') as f: f.write(line + '\n')


TXT = dict(text=True, encoding='utf-8', errors='replace', capture_output=True)  # tqdm output has broken UTF-8


def sh(cmd, check=True, input=None, timeout=None):
    r = subprocess.run(cmd, shell=True, input=input, timeout=timeout, **TXT)
    if check and r.returncode != 0:
        raise RuntimeError(f'{cmd}\n{r.stdout}\n{r.stderr}')
    return r.stdout


BASE = {
    'pd_topology.pd_prefill_instances': 1, 'pd_topology.pd_decode_instances': 1,
    'pd_topology.pd_kv_connector': 'NixlConnector', 'pd_topology.pd_kv_buffer_device': 'cpu',
    'vllm_prefill.gpu_memory_utilization': 0.9, 'vllm_prefill.max_num_seqs': 128,
    'vllm_prefill.max_num_batched_tokens': 8192,
    'vllm_decode.gpu_memory_utilization': 0.9, 'vllm_decode.max_num_seqs': 128,
    'vllm_decode.max_num_batched_tokens': 8192,
}
CONFIGS = [
    # linear backend, same on both roles; attention FLASHINFER, kv auto
    ('L1_linear_auto',           dict(pl='auto',    dl='auto',    attn='FLASHINFER', kv='auto', tuned='false')),
    ('L2_linear_humming',        dict(pl='humming', dl='humming', attn='FLASHINFER', kv='auto', tuned='false')),
    ('L3_linear_triton_default', dict(pl='triton',  dl='triton',  attn='FLASHINFER', kv='auto', tuned='false')),
    ('L4_linear_triton_tuned',   dict(pl='triton',  dl='triton',  attn='FLASHINFER', kv='auto', tuned='true')),
    # attention backend, with triton tuned on prefill and auto (Marlin) on decode
    ('A1_attn_flashinfer',       dict(pl='triton',  dl='auto',    attn='FLASHINFER', kv='auto', tuned='true')),
    ('A2_attn_triton',           dict(pl='triton',  dl='auto',    attn='TRITON_ATTN', kv='auto', tuned='true')),
    ('A3_attn_flash',            dict(pl='triton',  dl='auto',    attn='FLASH_ATTN', kv='auto', tuned='true')),
    ('A4_attn_auto',             dict(pl='triton',  dl='auto',    attn='auto', kv='auto', tuned='true')),
    ('A5_attn_flashinfer_fp8',   dict(pl='triton',  dl='auto',    attn='FLASHINFER', kv='fp8', tuned='true')),
    ('A6_attn_triton_fp8',       dict(pl='triton',  dl='auto',    attn='TRITON_ATTN', kv='fp8', tuned='true')),
]


def render(c):
    v = dict(BASE)
    v.update({'vllm_prefill.linear_backend': c['pl'], 'vllm_decode.linear_backend': c['dl'],
              'vllm_decode.attention_backend': c['attn'], 'vllm_decode.kv_cache_dtype': c['kv'],
              'vllm_decode.tuned_kernel_configs': c['tuned']})
    text = open(os.path.join(K8S, '01-deployment_template.yaml')).read()
    for k, val in v.items():
        text = text.replace('${' + k + '}', str(val))
    lines = [l for l in text.splitlines() if '${' not in l]  # as apply_config.sh: drop unrendered flags
    return '\n'.join(lines).replace('__PD_CONFIG_SHA__', uuid.uuid4().hex[:16]) + '\n'


def wait_tuning():
    log('waiting for fp8-tune TUNING-DONE')
    while True:
        r = subprocess.run(f'kubectl -n {NS} logs fp8-tune', shell=True, **TXT)
        if 'TUNING-DONE' in r.stdout:
            break
        if 'Traceback' in r.stdout:
            log('tuning FAILED:\n' + r.stdout[-3000:]); return False
        if r.returncode != 0 and 'NotFound' in r.stderr:
            log('fp8-tune pod not found'); return os.path.isdir(os.path.join(K8S, 'tuned-configs'))
        time.sleep(30)
    dst = os.path.join(K8S, 'tuned-configs')
    tmp = os.path.join(OUT, 'tune-out-' + uuid.uuid4().hex[:6])
    sh(f'kubectl -n {NS} cp fp8-tune:/out {tmp}')
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, '**', '*.json'), recursive=True):
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))  # /home and /work are different filesystems
    with open(os.path.join(OUT, 'tuning.log'), 'w') as f: f.write(r.stdout)
    files = sorted(glob.glob(os.path.join(dst, '*.json')))
    log(f'tuned configs copied: {[os.path.basename(x) for x in files]}')
    sh(f'kubectl -n {NS} delete pod fp8-tune --wait=true')
    return len(files) == 4


def apply_config(name, c):
    sh(f'kubectl create configmap pd-scripts -n {NS} --from-file=launcher.sh={K8S}/launcher.sh '
       f'--from-file=pd_router.py={K8S}/pd_router.py --dry-run=client -o yaml | kubectl apply -f -')
    files = sorted(glob.glob(os.path.join(K8S, 'tuned-configs', '*.json')))
    if files:
        tgz = os.path.join(OUT, 'tuned-configs.tgz')
        with tarfile.open(tgz, 'w:gz') as t:
            for f in files: t.add(f, arcname=os.path.basename(f))
        sh(f'kubectl create configmap pd-tuned-configs -n {NS} --from-file=tuned-configs.tgz={tgz} '
           f'--dry-run=client -o yaml | kubectl apply -f -')
    dep = os.path.join(OUT, f'{name}.deployment.yaml')
    open(dep, 'w').write(render(c))
    t0 = time.time()
    sh(f'kubectl apply -f {dep}')
    deadline = time.time() + 1800
    while time.time() < deadline:
        r = subprocess.run(f'kubectl rollout status deployment/vllm-pd -n {NS} --timeout=30s', shell=True, **TXT)
        if r.returncode == 0:
            return True, time.time() - t0
        rs = sh(f"kubectl get pod -n {NS} -l app=vllm-pd -o jsonpath='{{range .items[*]}}{{.metadata.creationTimestamp}} "
                f"{{.status.containerStatuses[?(@.name==\"engine\")].restartCount}}{{\"\\n\"}}{{end}}'", check=False)
        restarts = [int(x.split()[1]) for x in rs.splitlines() if len(x.split()) == 2 and x.split()[1].isdigit()]
        if restarts and max(restarts) >= 1:
            return False, time.time() - t0
    return False, time.time() - t0


def pod():
    return sh(f"kubectl get pod -n {NS} -l app=vllm-pd -o jsonpath='{{.items[0].metadata.name}}'").strip()


def run(name, c):
    log(f'=== {name}: {c}')
    ok, startup = apply_config(name, c)
    logs = sh(f'kubectl logs deployment/vllm-pd -n {NS} -c engine --tail=-1', check=False)
    if not ok:
        logs += sh(f'kubectl logs deployment/vllm-pd -n {NS} -c engine --previous --tail=300', check=False)
    keyre = re.compile(r'launcher:|Selected .* for|Using configuration from|Using default W8A8|linear-backend|'
                       r'attention backend|Using AttentionBackend|FlashInfer resolved|KV cache size|Error|error:|OutOfMemory|Traceback')
    key = '\n'.join(l for l in logs.splitlines() if keyre.search(l) and 'GET /' not in l)
    open(os.path.join(OUT, f'{name}.log'), 'w').write(key + '\n\n--- full tail ---\n' + '\n'.join(logs.splitlines()[-400:]))
    result = {'name': name, 'config': c, 'started': ok, 'startup_s': round(startup)}
    if ok:
        p = pod()
        sh(f'kubectl cp {BENCH} {NS}/{p}:/tmp/bench_in_pod.py -c engine')
        try:
            o = sh(f'kubectl exec -n {NS} {p} -c engine -- python3 /tmp/bench_in_pod.py', timeout=1500)
            m = re.search(r'BENCH_RESULT (.*)', o)
            result['bench'] = json.loads(m.group(1)) if m else {'raw': o[-2000:]}
        except Exception as e:
            result['bench_error'] = str(e)[-2000:]
        rs = sh(f"kubectl get pod -n {NS} {p} -o jsonpath='{{.status.containerStatuses[?(@.name==\"engine\")].restartCount}}'", check=False)
        result['engine_restarts_after_bench'] = rs
    json.dump(result, open(os.path.join(OUT, f'{name}.json'), 'w'), indent=2)
    s = (result.get('bench') or {}).get('summary')
    log(f'{name}: started={ok} startup={round(startup)}s summary={s} err={result.get("bench_error", "")[:200]}')
    with open(os.path.join(OUT, 'summary.txt'), 'a') as f: f.write(json.dumps({'name': name, 'config': c, 'started': ok, 'summary': s}) + '\n')


def main():
    if not wait_tuning():
        log('WARNING: tuned configs incomplete; configs with tuned=true would fail — running anyway')
    for name, c in CONFIGS:
        try:
            run(name, c)
        except Exception as e:
            log(f'{name}: driver exception {e}')
    sh(f'kubectl -n {NS} scale deployment/vllm-pd --replicas=0', check=False)
    log('ALL DONE, vllm-pd scaled to 0')


if __name__ == '__main__':
    main()
