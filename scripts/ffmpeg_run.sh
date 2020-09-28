#!/bin/bash

# create a video of the screenshots produced by a run
#   ./scripts/ffmpeg_run.sh runs/run_12345

fps=2
crf=12
fn=run.mp4

IFS=$'\n' 

while getopts "r:c:o:" opt; do
  case "$opt" in
    r) fps="$OPTARG";;
    c) crf="$OPTARG";;
    o) fn="$OPTARG";;
    ?) exit 1
  esac
done

shift $((OPTIND-1))

for A; do
  cd "$A"
  ffmpeg -y -framerate "$fps" -i step%05d.png -pix_fmt yuv420p -crf "$crf" "$fn"
done
