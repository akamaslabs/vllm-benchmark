import sys
from perfetto.trace_processor import TraceProcessor
tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    try:
        for r in tp.query(s):
            print("  ", {k: getattr(r, k) for k in r.__dict__ if not k.startswith("_")})
    except Exception as e:
        print("  ERR", str(e)[:200])
print("trace bounds"); q("select (end_ts-start_ts)/1e9 as secs from trace_bounds")
print("slices per category"); q("select category, count(*) n from slice group by 1 order by n desc limit 20")
print("processes"); q("select pid, name from process order by pid limit 30")
print("tracks (top by slices)"); q("select t.name, t.type, count(*) n from slice s join track t on s.track_id=t.id group by 1,2 order by n desc limit 30")
print("top slice names"); q("select name, count(*) n, sum(dur)/1e6 as ms from slice group by 1 order by ms desc limit 40")
print("counter tracks"); q("select name, count(*) n from counter_track ct join counter c on c.track_id=ct.id group by 1 order by n desc limit 40")
