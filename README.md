# Experiments in Exploration

Clone this repository somewhere.

Change into that directory.

Do this bit of black magic for dependencies:

    git fetch origin +refs/heads/deps/*:refs/heads/deps/*

Create a virtualenv:

    virtualenv --python=/usr/bin/python3 env

Install dependencies (Ubuntu)

    apt-get install libspatialindex-dev
    [You need embree 3.x.]
    ./env/bin/pip install -r requirements.txt

Run the demo:

    ./env/bin/python3 setup.py develop
    ./env/bin/python3 src/__main__.py
