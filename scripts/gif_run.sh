#!/bin/bash

# create a gif of the screenshots produced by a run
#   ./scripts/gifrun.sh runs/run_12345

fps=18
scale=w=256:h=-1
fn=run.gif
ffmpeg=ffmpeg

IFS=$'\n' 

while getopts "r:s:o:h" opt; do
  case "$opt" in
    r) fps="$OPTARG";;
    s) scale="$OPTARG";;
    o) fn="$OPTARG";;
    h|?) echo "usage: $0 [-r fps=$fps] [-s scale=$scale] [-o $fn]" >&2; exit 1
  esac
done

shift $((OPTIND-1))

for A; do
  pushd "$A"
  "$ffmpeg" -y -framerate "$fps" -i step%05d.png -filter_complex '[0:v] scale='"$scale"',split [a][b];[a] palettegen [p];[b][p] paletteuse' "$fn" || exit
  popd
done
