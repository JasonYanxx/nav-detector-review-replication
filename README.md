# GNSS/RAIM Snapshot Detector Review: Replication Package

This repository contains the public code and sanitized processed geometry for the published epoch-3549 GPS validation accompanying the paper:

> A Tutorial Review Of Statistical Snapshot Detectors for GNSS/RAIM Fault Detection: Unified Derivations and Detector Relationships

The package is intentionally limited to the main GPS-only validation workflow. It is not a full GNSS processing library and it does not include the manuscript drafting repository, raw source project files, or internal research notes. The default run reproduces the published epoch 3549 Tables 5–8 / Figures 3–5 evidence. The earlier epoch-3648 pilot remains available with `--legacy-epoch3648`.

## What This Repository Provides

The script `src/validation/simulate_snapshot_validation.py` runs the published GPS case `published_epoch3549_gps` by default.

It generates:

- threshold and relationship sanity checks;
- detection probability curves under synthetic Gaussian noise and a synthetic single-measurement bias;
- theoretical minimum detectable bias (MDB) summaries;
- type I / type II error tradeoff data;
- PDF figures generated for the published workflow.

The validation uses a real GPS-only satellite geometry from one medium-urban epoch and generates all measurement noise and fault biases synthetically.

## Repository Layout

```text
src/validation/
  detectors.py
  simulate_snapshot_validation.py

data/processed/medium_urban_epoch3549_gps/
  geometry.csv
  geometry_matrix.npz
  metadata.json

results/published_epoch3549_gps/
  geometry_summary.json
  matched_local_mdb_by_measurement.csv
  main_epoch3549_local_detection_probability.csv
  main_epoch3549_workflow_detection_probability.csv
  main_epoch3549_type_tradeoff.csv
  run_summary.json  # generated after running the script

figures/
  section6_*.pdf  # generated after running the script

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

The script writes regenerated numerical outputs to `results/published_epoch3549_gps/` and regenerated Figures 3–5 to `figures/`. Use `python src/validation/simulate_snapshot_validation.py --legacy-epoch3648` to rerun the retained pilot.

## Published Outputs

After running `bash replication/reproduce_section6.sh`, use the following mapping to inspect the generated published outputs.

| Published output | What it reports | Generated file to inspect |
|---|---|---|
| Geometry summary | Satellite-geometry size, rank, condition number, and selected G09 direction | `results/published_epoch3549_gps/geometry_summary.json` |
| Matched-local MDB and identity audit | Matched local-test MDB values and squared-statistic identity residuals | `results/published_epoch3549_gps/matched_local_mdb_by_measurement.csv` |
| Matched-local empirical power (Figure 3) | Baarda, standardized Jackknife, normalized solution-separation, and theoretical local detection probability | `figures/section6_baarda_local_detection_probability.pdf`; source data in `results/published_epoch3549_gps/main_epoch3549_local_detection_probability.csv` |
| Workflow detection probability (Table 7 / Figure 4) | Published workflow thresholds and direction-conditioned detection probabilities | `figures/section6_detection_probability.pdf`; source data in `results/published_epoch3549_gps/main_epoch3549_workflow_detection_probability.csv` |
| Type I / Type II error tradeoff (Table 8 / Figure 5) | Type I and Type II error curves under varying family-wise false-alarm budgets | `figures/section6_type_tradeoff_*.pdf`; source data in `results/published_epoch3549_gps/main_epoch3549_type_tradeoff.csv` |
| Run summary and provenance | Seed, geometry identity, output inventory, and nominal tradeoff values | `results/published_epoch3549_gps/run_summary.json` |

Additional diagnostic outputs:

The default run intentionally keeps the public output surface small: the
matched-local CSV contains all 11 directions, while the three main CSV files
contain the selected G09 direction and the frozen 21-point or 40-point grids.
The epoch-3648 files and their historical names are legacy-only outputs from
`--legacy-epoch3648` and are not the published Table 5–8 evidence. Legacy
figures are isolated under `figures/legacy_epoch3648/` so they cannot replace
the published PDFs in `figures/`.

## Reproducibility Notes

The default run uses:

- random seed: `20260509`;
- Monte Carlo trials: `10000`;
- bias grid: `0` to `10` in `0.5` whitened-sigma increments;
- false alarm settings: `alpha=0.05`, `alpha0=0.05`, `tau=0.05`;
- geometry: `data/processed/medium_urban_epoch3549_gps/geometry_matrix.npz`.

The output files are generated locally by the reproduction script and are intentionally not committed to this repository. Small numerical differences may occur across Python, NumPy, SciPy, or Matplotlib versions.

## Scope Limitations

This package supports the illustrative validation in the accompanying tutorial-review paper only. The results should not be interpreted as:

- a full RAIM benchmark;
- a real urban fault-detection performance evaluation;
- a comparison under non-Gaussian measurement noise;
- a validation of fault exclusion or protection-level algorithms.

Only the satellite/receiver geometry is reused from the source GNSS processing context. Synthetic Gaussian noise and synthetic single-measurement biases are generated inside this repository.

## Citation

If you use this code, please cite the accompanying paper:

Yan, P., Song, B., Li, Y., & Hsu, L.-T. (2026). *A Tutorial Review of Statistical Snapshot Detectors for GNSS/RAIM Fault Detection: Unified Derivations and Detector Relationships*. **Sensors, 26**(15), 4938. https://doi.org/10.3390/s26154938
