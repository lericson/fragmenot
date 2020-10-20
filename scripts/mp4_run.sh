#!/bin/bash

# create a video of the screenshots produced by a run
#   ./scripts/mp4_run.sh runs/run_12345

fps=18
crf=12
fn=run.mp4
ffmpeg=ffmpeg

IFS=$'\n' 

while getopts "r:c:o:h" opt; do
  case "$opt" in
    r) fps="$OPTARG";;
    c) crf="$OPTARG";;
    o) fn="$OPTARG";;
    h|?) echo "usage: $0 [-r fps] [-c crf] [-o out.mp4]" >&2; exit 1
  esac
done

shift $((OPTIND-1))

for A; do
  pushd "$A"
  "$ffmpeg" -y -framerate "$fps" -i step%05d.png -pix_fmt yuv420p -crf "$crf" "$fn" || exit
  popd
done
