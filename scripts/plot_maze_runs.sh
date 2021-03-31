#!/bin/sh
plot_sort_groups=false plot_xmax=1200 \
plot_colors=[C4,C3,C2,C0,C5,C6,C1] plot_ncol=4 \
exec ./scripts/plot_runs.py \
  -- --group='$d=2,      c_T=0.00$' var/runs_maze_bt_d2_ct00/run_* \
  -- --group='$d=5,      c_T=0.00$' var/runs_maze_bt_d5_ct00/run_* \
  -- --group='$d=\infty, c_T=0.00$' var/runs_maze_bt_di_ct00/run_* \
  -- --group='$d=\infty, \rho_f=1$' --d=100.0 var/runs_maze_tree_uni/run_*   \
  -- --group='$d=2,      c_T=0.25$' var/runs_maze_bt_d2_ct25/run_* \
  -- --group='$d=5,      c_T=0.25$' var/runs_maze_bt_d5_ct25/run_* \
  -- --group='$d=\infty, c_T=0.25$' var/runs_maze_bt_di_ct25/run_* \
