#!/bin/bash

# create a video of the screenshots produced by a run
#   ./scripts/mp4_run.sh runs/run_12345

fps=18
crf=12
fn=run.mp4
ffmpeg=ffmpeg
inpat='step%05d.png'

IFS=$'\n' 

while getopts "r:c:o:p:h" opt; do
  case "$opt" in
    r) fps="$OPTARG";;
    c) crf="$OPTARG";;
    o) fn="$OPTARG";;
    p) inpat="$OPTARG";;
    [h?]) echo "usage: $0 [-r fps] [-c crf] [-o out.mp4]" >&2; exit 1
  esac
done

shift $((OPTIND-1))

for A; do
  pushd "$A"
  "$ffmpeg" -y -framerate "$fps" -i "$inpat" -pix_fmt yuv420p -crf "$crf" "$fn" || exit
  popd
done
