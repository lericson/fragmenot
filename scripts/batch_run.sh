#!/bin/bash

# run njobs simultaneously ntimes

python=${python:-./env/bin/python}
script=${script:-./src/__main__.py}
jobs=(
    "GUI_MINIMIZED=true PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
    "GUI_MINIMIZED=true TREE_SEARCH_SCORE_FUNCTION=gn PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
    "GUI_MINIMIZED=true TREE_SEARCH_PATH_SELECTION=bfs PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
    "GUI_MINIMIZED=true TREE_SEARCH_SCORE_FUNCTION=gn TREE_SEARCH_PATH_SELECTION=bfs PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
    "GUI_MINIMIZED=true TREE_SEARCH_WEIGHT_FUNCTION=uniform PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
    "GUI_MINIMIZED=true TREE_SEARCH_PATH_SELECTION=bfs PRM_MAX_DEGREE_JUMP=0 PARAME_PROFILE=[shortrange,pv"{000,050,100,200,300,500,inf}"] $python $script"
)
njobsmax=${njobsmax:-4}
njobsnow=0
ntimes=${ntimes:-100}

set -me

log() {
    printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$@"
}

int() {
  ntimes=0
  for i in {1..6}; do
    (( i > 1 )) && sleep 1 || true
    sig=INT
    (( i > 3 )) && sig=TERM
    (( i > 5 )) && sig=KILL
    readarray -td $'\n' running < <(jobs -p -r)
    njobsnow=${#running[@]}
    (( njobsnow == 0 )) && return
    log "interrupting $njobsnow jobs (attempt $i, SIG$sig, pids: ${running[*]})"
    kill -$sig "${running[@]}" || true
  done
  log "processes seem immortal, giving up. bye."
}

trap int INT

for (( ; 0 < ntimes; )); do
  for (( ; njobsnow < njobsmax; ntimes--, njobsnow++ )); do
    job="${jobs[ntimes % ${#jobs[@]}]} &"
    eval $job
    readarray -td $'\n' running < <(jobs -p -r)
    log "started job \`${job}\` (pid=$!, $ntimes remaining, ${#running[@]} running)"
  done
  wait -n
  readarray -td $'\n' running < <(jobs -p -r)
  njobsnow=${#running[@]}
done
log "finished, waiting for remaining jobs"
wait
