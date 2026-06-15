# Result Mapping

This document maps manuscript-supporting validation artifacts to the release package files that reproduce them.

## Run ID

```text
section6_medium_urban_epoch3648_gps
```

## Command

```bash
python src/validation/simulate_snapshot_validation.py
```

or:

```bash
bash replication/reproduce_section6.sh
```

## Inputs

```text
data/processed/medium_urban_epoch3648_gps/geometry_matrix.npz
data/processed/medium_urban_epoch3648_gps/metadata.json
```

## Outputs

| Manuscript-supporting object | Output file |
|---|---|
| Geometry summary | `results/section6_medium_urban_epoch3648_gps/geometry_summary.json` |
| Threshold and identity sanity checks | `results/section6_medium_urban_epoch3648_gps/sanity_summary.csv` |
| Detection probability curves | `results/section6_medium_urban_epoch3648_gps/detection_probability.csv` |
| Detection probability figure | `figures/section6_detection_probability.pdf` |
| MDB summary table | `results/section6_medium_urban_epoch3648_gps/mdb_summary.csv` |
| Per-measurement MDB values | `results/section6_medium_urban_epoch3648_gps/theoretical_mdb_by_measurement.csv` |
| Type I / type II tradeoff data | `results/section6_medium_urban_epoch3648_gps/type_tradeoff.csv` |
| Type I / type II figures | `figures/section6_type_tradeoff_*.pdf` |
| Run manifest | `experiments/manifests/section6_medium_urban_epoch3648_gps.json` |

## Interpretation Boundary

The validation checks detector-statistic relationships, threshold behavior, detection probability, and MDB calculations for one pilot GPS-only geometry under synthetic Gaussian noise and synthetic single-measurement biases.

It is an illustrative reproducibility package, not a full benchmark dataset.

