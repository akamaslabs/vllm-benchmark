with k as (select ts, ts + dur as te, name like 'ncclDevKernel%' as nccl
           from slice where category = 'cuda_gpu_kernel_launch'),
ev as (select ts, 1 - nccl as dc, nccl as dn, null as metric, null as value from k
       union all select te, nccl - 1, -nccl, null, null from k
       union all select c.ts + 500000, 0, 0, substr(t.name, 1, instr(t.name, ' (') - 1), c.value
                 from counter c join counter_track t on t.id = c.track_id
                 where t.name like '%(Utilization) (GPU Metric)'),
run as (select ts, metric, value, lead(ts) over (order by ts) - ts as span,
               sum(dc) over (order by ts rows unbounded preceding) as nc,
               sum(dn) over (order by ts rows unbounded preceding) as nn
        from ev),
st as (select case when nc > 0 then '1 compute kernel running'
                   when nn > 0 then '2 only NCCL kernel (waiting for the peer)'
                   else '3 no kernel (idle)' end as gpu_state, metric, value, span
       from run)
select gpu_state,
       round(100.0 * sum(span) / (select sum(span) from st), 1) as pct_of_time,
       round(avg(case when metric = 'GR Active' then value end), 1) as gr_active,
       round(avg(case when metric = 'SM Active' then value end), 1) as sm_active,
       round(avg(case when metric = 'Tensor Active' then value end), 1) as tensor_active,
       round(avg(case when metric = 'DRAM Read' then value end), 1) as dram_read
from st group by gpu_state
union all
select '= whole trace', 100.0,
       round(avg(case when metric = 'GR Active' then value end), 1),
       round(avg(case when metric = 'SM Active' then value end), 1),
       round(avg(case when metric = 'Tensor Active' then value end), 1),
       round(avg(case when metric = 'DRAM Read' then value end), 1)
from st
order by 1
