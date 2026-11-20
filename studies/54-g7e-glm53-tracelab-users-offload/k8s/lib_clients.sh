# shellcheck shell=bash
# Helpers for the two long-lived AIPerf client pods (run_test.sh, stop_aiperf.sh). Sourced after
# workload.env (NS). The pods have python3 but no pgrep/pkill: processes are found through /proc.
CLIENTS="aiperf-client-0 aiperf-client-1"
kx() { local c=$1; shift; kubectl -n "$NS" exec "$c" -c aiperf -- "$@"; }

# A task killed by Akamas (finish, abort, timeout) drops its kubectl exec but the replay keeps running
# in the pod (study 47: a survivor doubled the next load). Kill everything except PID 1 (sleep
# infinity) and this command, then check nothing is left (exit 1 if something is).
REAPER='
import os, signal, time
me = {os.getpid(), os.getppid()}
def others():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != 1 and int(p) not in me:
            try:
                c = open("/proc/%s/cmdline" % p, "rb").read().replace(b"\0", b" ").decode().strip()
            except OSError:
                continue
            if c: out.append((int(p), c))
    return out
found = others()
for pid, c in found:
    print("killing stale process %d: %s" % (pid, c[:100]))
    try: os.kill(pid, signal.SIGKILL)
    except OSError: pass
time.sleep(2)
left = others()
print("stale processes killed: %d, left: %d" % (len(found), len(left)))
raise SystemExit(1 if left else 0)
'

# SIGINT to the top aiperf process ("aiperf system_controller", child of client_run.sh's sh): on its
# first SIGINT AIPerf stops sending, cancels the requests in flight and writes its files (study 38).
# Its workers ignore SIGINT. Prints the PIDs signalled.
STOPPER='
import os, signal
procs = {}
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        s = open("/proc/%s/stat" % p).read()
    except OSError:
        continue
    procs[p] = (s[s.index("(") + 1:s.rindex(")")], s[s.rindex(")") + 2:].split()[1])
for p, (comm, ppid) in procs.items():
    if comm.startswith("aiperf") and not procs.get(ppid, ("",))[0].startswith("aiperf"):
        os.kill(int(p), signal.SIGINT)
        print(p)
'
