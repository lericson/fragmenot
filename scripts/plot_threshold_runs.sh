#!/bin/sh
plot_ncol=3 plot_orientation=portrait plot_sort_groups=false \
exec ./scripts/plot_runs.py \
  -- --group='Baseline' --d=100.0 rsync/var/runs_uni/run_* \
  -- --group='$c_T=6.00$' var/runs_t6/run_* \
  -- --group='$c_T=2.00$' var/runs_t2/run_* \
  -- --group='$c_T=0.25$' var/runs_t025/run_* \
  -- --group='$c_T=0.10$' var/runs_t010/run_* \
  -- --group='$c_T=0.05$' var/runs_t005/run_* \
#  -- --group='$c_T=12.0$' var/runs_t12/run_* \
#  -- --group='$c_T=4.00$' var/runs_t4/run_* \
#  -- --group='$c_T=1.00$' var/runs_t1/run_* \
#  -- --group='$c_T=0.50$' var/runs_t05/run_* \
#  -- --group='$c_T=0.00$' --d=100.0 rsync/var/runs_ours/run_* \
