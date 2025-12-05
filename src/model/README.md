# Model Code
Baseline and analog simulation helpers for the AIHWKIT project.

## Setup
1) Create a virtualenv (optional):  
```bash
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:  
```bash
pip install -r requirements.txt
```

## Scripts
- `src/model/baseline_mnist.py` — trains a digital MNIST CNN baseline and saves `data/baseline.pt` plus `data/baseline_metrics.csv`.
- `src/model/analog_mnist.py` — loads the baseline, converts to AIHWKIT analog layers, and logs accuracy/latency/energy estimates to `data/analog_metrics.csv`.
- `src/model/run_sweeps.py` — runs a few analog configs (different noise levels) and writes `data/analog_runs.csv` plus a quick plot in `data/figures/acc_energy.png`.
- `src/model/experiment_suite.py` — one-shot runner: trains (or reuses) baseline, runs an analog sweep, and writes a summary for the proposal targets.
- `src/model/analog_demo.py` — minimal AIHWKIT example (single analog layer) to sanity-check the install.

## Quickstart
Train baseline (default: 3 epochs, batch 128):  
```bash
python src/model/baseline_mnist.py --epochs 3 --batch-size 128 --lr 0.01 --device auto
```

Run a single analog eval (requires `data/baseline.pt`):  
```bash
python src/model/analog_mnist.py --config-name noisy0.02 --out-noise 0.02 --batch-size 256
```

Run the built-in sweep set:  
```bash
python src/model/run_sweeps.py
```

End-to-end (baseline + sweep + summary):  
```bash
python src/model/experiment_suite.py --epochs 3 --out-noise-list "0,0.02,0.05"
```

AIHWKIT sanity demo (quick analog layer example):  
```bash
python src/model/analog_demo.py
```

Outputs land in `data/` so they can be committed or plotted later.
