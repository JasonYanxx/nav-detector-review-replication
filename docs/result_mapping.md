# Result Mapping

The default public run is `published_epoch3549_gps`, using the sanitized GPS
geometry at epoch 3549 (`n=11`, `m=4`, `q=7`). The median-$S_{kk}$ direction is
G09 (`k=5`), which is the direction used for Figures 3–5 and the workflow
tradeoff. The old epoch-3648 pilot remains a separately selectable legacy run;
its figures are isolated under `figures/legacy_epoch3648/`.

## Published outputs

| Published object | Generated file |
|---|---|
| Geometry summary (Table 5 support) | `results/published_epoch3549_gps/geometry_summary.json` |
| Matched local MDB / identity audit (Table 6 support) | `results/published_epoch3549_gps/matched_local_mdb_by_measurement.csv` |
| Main local detection probability (Figure 3) | `results/published_epoch3549_gps/main_epoch3549_local_detection_probability.csv`; `figures/section6_baarda_local_detection_probability.pdf` |
| Workflow detection probability (Figure 4) | `results/published_epoch3549_gps/main_epoch3549_workflow_detection_probability.csv`; `figures/section6_detection_probability.pdf` |
| Type-I / Type-II tradeoff (Figure 5, Table 8) | `results/published_epoch3549_gps/main_epoch3549_type_tradeoff.csv`; `figures/section6_type_tradeoff_*.pdf` |
| Run and provenance record | `results/published_epoch3549_gps/run_summary.json` |

The generated numerical files and PDFs are ignored so that each run records
its own runtime. Tables 5–8 are supported by the files above; no Route-B
GPS+BeiDou, appendix, raw observation, or private selection-audit material is
included in this release.
