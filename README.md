# Experiments in Exploration

Clone this repository somewhere.

Change into that directory.

Do this bit of black magic for dependencies:

    git fetch origin +refs/heads/deps/*:refs/heads/deps/*

Create a virtualenv:

    virtualenv --python=/usr/bin/python3 env

Install dependencies (Ubuntu)

    apt-get install libspatialindex-dev
    [You need to download, compile and install embree v2.x.]
    ./env/bin/pip install -r requirements.txt -e .

Run the demo:

    ./env/bin/python3 src/__main__.py
