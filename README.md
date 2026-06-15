# GNSS/RAIM Snapshot Detector Review: Replication Package

This repository contains the code and processed data needed to reproduce the illustrative validation results used in the paper:

> A Tutorial Review Of Statistical Snapshot Detectors for GNSS/RAIM Fault Detection: Unified Derivations and Detector Relationships

The package is intentionally limited to the paper-supporting Section 6 validation workflow. It is not a full GNSS processing library and it does not include the manuscript drafting repository, raw source project files, or internal research notes.

## What This Repository Reproduces

The script `src/validation/simulate_snapshot_validation.py` reproduces the pilot validation run `section6_medium_urban_epoch3648_gps`.

It generates:

- threshold and relationship sanity checks;
- detection probability curves under synthetic Gaussian noise and a synthetic single-measurement bias;
- theoretical minimum detectable bias (MDB) summaries;
- type I / type II error tradeoff data;
- the PDF figures used by the manuscript validation section.

The validation uses a real GPS-only satellite geometry from one medium-urban epoch and generates all measurement noise and fault biases synthetically.

## Repository Layout

```text
src/validation/
  detectors.py
  simulate_snapshot_validation.py

data/processed/medium_urban_epoch3648_gps/
  geometry.csv
  geometry_matrix.npz
  metadata.json

experiments/manifests/
  section6_medium_urban_epoch3648_gps.json

results/section6_medium_urban_epoch3648_gps/
  detection_probability.csv
  geometry_summary.json
  mdb_summary.csv
  sanity_summary.csv
  theoretical_mdb_by_measurement.csv
  type_tradeoff.csv

figures/
  section6_detection_probability.pdf
  section6_type_tradeoff_*.pdf

docs/
  data_provenance.md
  result_mapping.md

replication/
  reproduce_section6.sh
```

## Quick Start

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the reproduction workflow:

```bash
bash replication/reproduce_section6.sh
```

Alternatively, run the Python script directly from the repository root:

```bash
python src/validation/simulate_snapshot_validation.py
```

The script writes regenerated numerical outputs to `results/section6_medium_urban_epoch3648_gps/`, regenerated figures to `figures/`, and an updated manifest to `experiments/manifests/section6_medium_urban_epoch3648_gps.json`.

## Reproducibility Notes

The default run uses:

- random seed: `20260509`;
- Monte Carlo trials: `10000`;
- bias grid: `0` to `10` in `0.5` whitened-sigma increments;
- false alarm settings: `alpha=0.05`, `alpha0=0.05`, `tau=0.05`;
- geometry: `data/processed/medium_urban_epoch3648_gps/geometry_matrix.npz`.

The output files included in this repository were generated with these settings. Small numerical differences may occur across Python, NumPy, SciPy, or Matplotlib versions.

## Scope Limitations

This package supports the illustrative validation in the accompanying tutorial-review paper only. The results should not be interpreted as:

- a full RAIM benchmark;
- a real urban fault-detection performance evaluation;
- a comparison under non-Gaussian measurement noise;
- a validation of fault exclusion or protection-level algorithms.

Only the satellite/receiver geometry is reused from the source GNSS processing context. Synthetic Gaussian noise and synthetic single-measurement biases are generated inside this repository.

## Citation

If you use this code, please cite the accompanying paper. A formal citation entry should be added after the paper metadata is finalized.

