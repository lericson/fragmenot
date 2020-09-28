# Experiments in Exploration

Clone this repository somewhere.

Change into that directory.

Do this bit of black magic for dependencies:

    git fetch origin +refs/heads/deps/*:refs/heads/deps/*

Create a virtualenv:

    virtualenv --python=/usr/bin/python3 env

Install dependencies

    Ubuntu:
    apt-get install libspatialindex-dev liboctomap-dev libdynamicedt3d-dev build-essential python3-dev python3-numpy
    ./scripts/install-embree.sh

    macOS:
    brew install spatialindex octomap embree
    [You need to manually install dynamicEDT3D?]

    All:
    ./env/bin/pip install -r requirements.txt
    ./env/bin/python3 setup.py develop

Run the demo:

    ./env/bin/python3 src/__main__.py

## Batch running

Most of what we do with this is to collect statistics. To make running batches
easier, there is a system with the very funny name "batsh".

How to use: start a tmux, then the batsh-queue program.

    ./scripts/batsh-cmds | ./scripts/batsh-queue

You may run batsh-cmds to inspect which commands will be run, one per line of output.

Once the queue program is running, open new tmux tabs and start a worker in each:

    ./scripts/batsh-worker

This has the advantage that you can cycle through the current runs in tmux,
with full terminal interactivity. Batsh is the future.

## Plotting

Use
    
    ./scripts/plot_runs.py ./var/runs/runs_foo/*

