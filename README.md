# Experiments in Exploration

Clone this repository somewhere.

Change into that directory.

Do this bit of black magic for dependencies:

    git fetch origin +refs/heads/deps/*:refs/heads/deps/*

Create a virtualenv:

    virtualenv --python=/usr/bin/python3 env

Install dependencies:

    ./env/bin/pip install -r requirements.txt

Run the demo:

    ./env/bin/python3 src/__main__.py
