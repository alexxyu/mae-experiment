# MAE Experiment

The code in this repository was used in a study on the effects of uni- and bi-directional motion
adaptation on direction discrimination sensitivity. Link TBD.

## Setup Instructions

The demo.py and exp.py scripts are relevant to the experimental procedure. The following setup
steps have been tested on an M1 ARM MacBook. Requires Python@3.10.

Run these the first time to create a virtual environment and install dependencies.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run demo.py to see an example of the stimulus:

```sh
source .venv/bin/activate
python demo.py
```

Or, run exp.py to run the experiment:

```sh
source .venv/bin/activate
python exp.py
```
