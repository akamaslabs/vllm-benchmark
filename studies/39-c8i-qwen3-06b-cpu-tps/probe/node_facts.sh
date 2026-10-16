#!/bin/bash
# Runs INSIDE vllm-0 (study 39 probe): `kubectl -n llm-cpu exec -i vllm-0 -c vllm -- bash -s < node_facts.sh`.
# Prints what the container actually sees: CPU model and ISA (AMX), SMT sibling layout, NUMA
# nodes, the exclusive cpuset the static CPU manager gave it, the cgroup memory limit and
# usage (vLLM sizes the KV cache from memory.max), and the vLLM CPU env. Read-only.
set -u
section() { echo; echo "### $*"; }
section "lscpu"
lscpu 2>&1
section "lscpu -e (CPU, CORE, NODE, SOCKET)"
lscpu -e=CPU,CORE,NODE,SOCKET,ONLINE 2>&1 | head -60
section "ISA flags of interest"
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(amx|avx512)' | sort -u | tr '\n' ' '; echo
section "NUMA"
if command -v numactl >/dev/null 2>&1; then numactl -H 2>&1; else
  echo "numactl not in the image; /sys/devices/system/node:"
  for n in /sys/devices/system/node/node*; do
    [ -d "$n" ] || continue
    echo "$(basename "$n"): cpus $(cat "$n/cpulist") ; $(grep -E 'MemTotal|MemFree' "$n/meminfo" | awk '{print $3, $4, $5}' | tr '\n' ' ')"
  done
fi
section "SMT siblings"
FIRST=$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status | cut -d, -f1 | cut -d- -f1)
echo "cpu$FIRST (first CPU of this cpuset): siblings $(cat /sys/devices/system/cpu/cpu"$FIRST"/topology/thread_siblings_list 2>/dev/null)"
echo "cpu0: siblings $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list 2>/dev/null) (reservedSystemCPUs should be this pair)"
section "cpuset / affinity of this container"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
echo "cpuset.cpus.effective: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo n/a)"
echo "cpuset.mems.effective: $(cat /sys/fs/cgroup/cpuset.mems.effective 2>/dev/null || echo n/a)"
echo "nproc: $(nproc)"
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
section "cgroup memory (bytes)"
for f in memory.max memory.current memory.peak; do
  echo "$f: $(cat /sys/fs/cgroup/$f 2>/dev/null || echo n/a)"
done
grep -E '^(anon|file|shmem|kernel) ' /sys/fs/cgroup/memory.stat 2>/dev/null
section "processes (RSS KiB)"
# No procps in the image: read /proc directly.
python3 - <<'PY'
import os
rows = []
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        st = dict(l.split(':', 1) for l in open('/proc/%s/status' % pid) if ':' in l)
        rows.append((int(st.get('VmRSS', '0 kB').split()[0]), pid, st['Threads'].strip(), st['Name'].strip(),
                     st.get('Cpus_allowed_list', '').strip()))
    except (OSError, KeyError, ValueError):
        pass
for r in sorted(rows, reverse=True)[:8]:
    print('pid %s rss %d KiB threads %s %s cpus %s' % (r[1], r[0], r[2], r[3], r[4]))
PY
section "env"
env | grep -E '^(VLLM_CPU_|VLLM_ENGINE_READY|LD_PRELOAD|OMP_|KMP_)' | sort
