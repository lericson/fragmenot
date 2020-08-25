#!/bin/bash

# create a video of the screenshots produced by a run
#   ./scripts/ffmpeg_run.sh runs/run_12345

fps=2
crf=12
fn=run.mp4

for A; do
  cd "$A"
  ffmpeg -y -framerate $fps -i step%05d.png -pix_fmt yuv420p -crf $crf $fn
done
