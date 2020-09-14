#!/bin/bash
for d in ./var/runs_*; do
  f=( "$d"/run_* )
  printf "%-20s %d %s\n" "$d:" "${#f[@]}"
done
