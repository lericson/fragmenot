#!/bin/sh
plot_orientation=portrait \
plot_ncol=3 \
plot_colors=[C0,C3,C2,C4,C5,C1] \
exec ./scripts/plot_runs.py --filter='(desc.d != 0.5) and not (5.0 < desc.d < 100.0)' rsync/var/runs_ours/run_*
