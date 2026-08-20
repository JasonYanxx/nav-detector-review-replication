# Data Provenance

## Processed Geometry

The default public replication workflow uses the processed geometry files in:

```text
data/processed/medium_urban_epoch3549_gps/
```

These files contain a GPS-only local ENU plus clock design matrix for one medium-urban epoch:

- epoch index: `3549`;
- number of measurements: `n = 11`;
- state dimension: `m = 4`;
- redundancy: `q = 7`;
- matrix rank: `4`;
- coordinate frame: local ENU plus GPS clock.

The file `geometry_matrix.npz` provides the matrix used by the simulation script. The file `geometry.csv` provides a tabular version of the same geometry-related information. The file `metadata.json` records the geometry summary and processing notes.

The earlier epoch-3648 GPS geometry is retained only for the explicit
`--legacy-epoch3648` pilot option.

## What Is Not Included

The original raw MATLAB source file and the source project's scale-mismatch experiment code are not included in this release package.

The replication package uses only the processed geometry. It does not reuse:

- logistic pseudorange error models;
- LQLC estimation results;
- scale-mismatch Monte Carlo results;
- RMSE or standard-deviation curves from the source project.

## Synthetic Validation Model

All validation observations are generated in this repository using:

```text
z = Hx + v
```

with whitened Gaussian noise and synthetic single-measurement bias alternatives. The published run uses the sanitized epoch-3549 GPS geometry; its synthetic covariance is the identity, so the whitened matrix `H` equals the processed geometry matrix. No raw observations or private selection-audit files are distributed.
