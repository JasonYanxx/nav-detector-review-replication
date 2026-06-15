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

| Paper artifact | Manuscript-supporting object | Output file |
|---|---|
| Table 3 | Geometry summary | `results/section6_medium_urban_epoch3648_gps/geometry_summary.json` |
| Table 4 | Detector statistics, false-alarm policies, and theoretical MDB values | `results/section6_medium_urban_epoch3648_gps/mdb_summary.csv` |
| Figure 2 | Detection probability figure | `figures/section6_detection_probability.pdf` |
| Figure 2 source data | Detection probability curves | `results/section6_medium_urban_epoch3648_gps/detection_probability.csv` |
| Figure 3 | Type I / type II tradeoff figure panels | `figures/section6_type_tradeoff_*.pdf` |
| Figure 3 source data | Type I / type II tradeoff data | `results/section6_medium_urban_epoch3648_gps/type_tradeoff.csv` |
| Diagnostic output | Threshold and identity sanity checks | `results/section6_medium_urban_epoch3648_gps/sanity_summary.csv` |
| Table 4 supporting calculation | Per-measurement MDB values used for the worst-case MDB column | `results/section6_medium_urban_epoch3648_gps/theoretical_mdb_by_measurement.csv` |
| Run record | Run manifest | `experiments/manifests/section6_medium_urban_epoch3648_gps.json` |

## Notes For Table 4

Paper Table 4 is reproduced from:

```text
results/section6_medium_urban_epoch3648_gps/mdb_summary.csv
```

The columns `theoretical_MDB_at_selected_k_sigma` and `theoretical_worst_case_workflow_MDB_sigma` correspond to the two MDB columns in Table 4. The selected measurement is recorded as `selected_measurement_k`, which is `3` in the default run. The per-measurement values behind the worst-case column are stored in:

```text
results/section6_medium_urban_epoch3648_gps/theoretical_mdb_by_measurement.csv
```

## Interpretation Boundary

The validation checks detector-statistic relationships, threshold behavior, detection probability, and MDB calculations for one pilot GPS-only geometry under synthetic Gaussian noise and synthetic single-measurement biases.

It is an illustrative reproducibility package, not a full benchmark dataset.
