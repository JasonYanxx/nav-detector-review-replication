"""Reproduce the published epoch-3549 GPS validation (or the retained pilot).

The public package deliberately contains only the main GPS case.  The random
draw layout is frozen to the published run: epoch 3549 uses a parent dimension
of 19, while the first 11 directions/columns are the released GPS geometry.
``--legacy-epoch3648`` keeps the earlier pilot available without making it the
default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detectors import compute_statistics, deleted_solution_matrix, projection_matrices


ROOT = Path(__file__).resolve().parents[2]
TRIALS = 10_000
MASTER_SEED = 20260509
BIAS_GRID = np.arange(0.0, 10.0 + 0.5, 0.5)
TRADEOFF_GRID = np.linspace(0.005, 0.2, 40)
ALPHA = ALPHA0 = TAU = 0.05
TARGET_PD = 0.95
WORKFLOWS = ("residual_parity", "jackknife_workflow", "solution_separation_hv")
WORKFLOW_LABELS = {
    "residual_parity": "Residual/parity",
    "jackknife_workflow": "Jackknife workflow",
    "solution_separation_hv": "Solution separation H/V",
}
PUBLISHED_EXPECTED_NPZ_SHA256 = "8818cdb567f91ff87d60309d2aa440b5b66cfc542dfcfb77ef6739dbb8a7f260"
PUBLISHED_EXPECTED_GEOMETRY_CSV_SHA256 = "3ece0cf088504fb0655a148c5c4f7d36a8871cce2f5bc991f9a3467adda8fda4"
PUBLISHED_EXPECTED_METADATA_SHA256 = "24cde4a65cffc541e58fdc643a02030b0ca01be60a6d738f12d41bfc78c0b5ba"
PUBLISHED_EXPECTED_CSV_HASHES = {
    "main_epoch3549_local_detection_probability.csv": "9dc34efdbbe98ecbb60c2301381b309681bf9879c0c5bd3fecabb8575ef6055c",
    "main_epoch3549_workflow_detection_probability.csv": "1c28b713e31c4fe325f54535d02196a81384aac6d03dff6d91014382938e08a9",
    "main_epoch3549_type_tradeoff.csv": "971bcf872568d5ccfd4ade8923c34f65c1ca3667b81e8429ffc0878dfeb49866",
}
PUBLISHED_EXPECTED_THRESHOLDS = {
    "residual_parity": 14.067140449340167,
    "matched_local": 3.8414588206941205,
    "jackknife_workflow": 8.05195624034816,
    "ss_h": 10.591050935012069,
    "ss_v": 9.315101996434459,
}
PUBLISHED_EXPECTED_S55 = 0.6769929377766291
PUBLISHED_EXPECTED_LOCAL_MDB = 4.381181162491282
PUBLISHED_EXPECTED_GLOBAL_MDB = 5.679539914832554
PUBLISHED_EXPECTED_LOCAL_THEORY_MAX_ERROR = 0.007676417455761864


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def assert_records_finite(records: list[dict[str, object]]) -> None:
    for record in records:
        for value in record.values():
            if isinstance(value, (float, int, np.floating, np.integer)) and not np.isfinite(value):
                raise AssertionError("A generated numeric value is non-finite.")


def assert_statistics_finite(statistics: dict[str, np.ndarray]) -> None:
    for value in statistics.values():
        if not np.isfinite(value).all():
            raise AssertionError("A detector statistic is non-finite.")


def mdb_lambda(threshold: float, df: int) -> float:
    def gap(value: float) -> float:
        return float(ncx2.sf(threshold, df, value) - TARGET_PD)

    upper = 1.0
    while gap(upper) < 0:
        upper *= 2.0
    return float(brentq(gap, 0.0, upper))


def load_geometry(legacy: bool) -> tuple[np.ndarray, dict[str, object], Path, Path]:
    data_dir = ROOT / "data/processed" / ("medium_urban_epoch3648_gps" if legacy else "medium_urban_epoch3549_gps")
    npz_path = data_dir / "geometry_matrix.npz"
    metadata_path = data_dir / "metadata.json"
    geometry_csv_path = data_dir / "geometry.csv"
    if not legacy:
        actual_hashes = {
            "npz": sha256_file(npz_path),
            "geometry_csv": sha256_file(geometry_csv_path),
            "metadata": sha256_file(metadata_path),
        }
        expected_hashes = {
            "npz": PUBLISHED_EXPECTED_NPZ_SHA256,
            "geometry_csv": PUBLISHED_EXPECTED_GEOMETRY_CSV_SHA256,
            "metadata": PUBLISHED_EXPECTED_METADATA_SHA256,
        }
        if actual_hashes != expected_hashes:
            raise AssertionError(f"Published input identity changed: {actual_hashes}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        if not legacy:
            if set(data.files) != {"H"}:
                raise AssertionError(f"Published NPZ key surface changed: {data.files}")
            H = np.asarray(data["H"])
        else:
            key = "H" if "H" in data.files else "G"
            H = np.asarray(data[key])
    if H.dtype != np.dtype(np.float64) or not np.isfinite(H).all():
        raise AssertionError("Released geometry must be finite float64.")
    if H.ndim != 2 or np.linalg.matrix_rank(H) != H.shape[1]:
        raise AssertionError("Released geometry is not full column rank.")
    if not legacy:
        if tuple(H.shape) != (11, 4):
            raise AssertionError(f"Published geometry shape changed: {H.shape}")
        expected = {
            "epoch_index": 3549, "configuration": "gps", "n": 11,
            "m": 4, "q": 7, "rank": 4,
            "satellite_ids": [1, 4, 7, 8, 9, 14, 16, 17, 21, 27, 30],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(f"Published metadata {key} changed.")
        if metadata["satellite_ids"][4] != 9:
            raise AssertionError("Published selected direction is not G09/k=5.")
        matrices = projection_matrices(H)
        s_diag = np.diag(matrices["S"])
        if not np.isclose(np.linalg.cond(H.T @ H), metadata["condition_number_hth"], atol=1e-12, rtol=0):
            raise AssertionError("Published geometry condition number changed.")
        if not np.isclose(np.min(s_diag), metadata["min_S_kk"], atol=1e-12, rtol=0) or not np.isclose(np.max(s_diag), metadata["max_S_kk"], atol=1e-12, rtol=0):
            raise AssertionError("Published residual-projector diagonal changed.")
    return H, metadata, npz_path, metadata_path


def thresholds(n: int, q: int) -> dict[str, float]:
    return {
        "residual_parity": float(chi2.ppf(1 - ALPHA, q)),
        "matched_local": float(chi2.ppf(1 - ALPHA0, 1)),
        "jackknife_workflow": float(chi2.ppf(1 - TAU / n, 1)),
        "ss_h": float(chi2.ppf(1 - 0.025 / (2 * n), 1)),
        "ss_v": float(chi2.ppf(1 - 0.025 / n, 1)),
    }


def alarms(stats: dict[str, np.ndarray], th: dict[str, float]) -> dict[str, np.ndarray]:
    ss = stats["SS"]
    return {
        "residual_parity": stats["T_R"] > th["residual_parity"],
        "jackknife_workflow": np.max(stats["J_tilde"] ** 2, axis=1) > th["jackknife_workflow"],
        "solution_separation_hv": (
            ((ss[:, :, 0] ** 2 > th["ss_h"]) | (ss[:, :, 1] ** 2 > th["ss_h"]) | (ss[:, :, 2] ** 2 > th["ss_v"]))
            .any(axis=1)
        ),
    }


def alarms_for_budget(stats: dict[str, np.ndarray], n: int, budget: float) -> dict[str, np.ndarray]:
    local = {
        "residual_parity": float(chi2.ppf(1 - budget, n - 4)),
        "jackknife_workflow": float(chi2.ppf(1 - budget / n, 1)),
        "ss_h": float(chi2.ppf(1 - budget / (4 * n), 1)),
        "ss_v": float(chi2.ppf(1 - budget / (2 * n), 1)),
    }
    return alarms(stats, local)


def _run_case(
    H: np.ndarray,
    metadata: dict[str, object],
    legacy: bool,
    result_dir: Path,
    figure_dir: Path,
    final_result_dir: Path,
    final_figure_dir: Path,
) -> None:
    epoch = int(metadata.get("epoch_index", 3648 if legacy else 3549))
    n, m = H.shape
    q = n - m
    run_id = "section6_medium_urban_epoch3648_gps" if legacy else "published_epoch3549_gps"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    mats = projection_matrices(H)
    s_diag = np.diag(mats["S"])
    order = np.lexsort((np.arange(n), s_diag))
    selected = int(order[n // 2])
    if not legacy and (selected != 4 or metadata.get("satellite_ids", [])[selected] != 9):
        raise AssertionError("Published median-S_kk direction is no longer G09/k=5.")
    th = thresholds(n, q)
    lambda_local = mdb_lambda(th["matched_local"], 1)
    lambda_global = mdb_lambda(th["residual_parity"], q)
    if not legacy:
        for key, expected in PUBLISHED_EXPECTED_THRESHOLDS.items():
            if not np.isclose(th[key], expected, atol=1e-12, rtol=0):
                raise AssertionError(f"Frozen threshold changed: {key}")

    # The released public geometry is the first 11 columns of the frozen parent
    # draw. This also keeps H0 and every bias paired.
    if legacy:
        rng = np.random.default_rng(MASTER_SEED)
        z0 = rng.normal(size=(TRIALS, n))
    else:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 3549]))
        z0 = rng.normal(size=(TRIALS, 19))[:, :11]
    stats0 = compute_statistics(H, z0)
    assert_statistics_finite(stats0)

    # Matched local MDB and identities for every released direction.
    local_rows: list[dict[str, object]] = []
    local_mdb = np.empty(n)
    solution_mdb = np.empty(n)
    selected_component = np.empty(n, dtype=int)
    max_mdb_identity_error = 0.0
    max_squared_identity_error = 0.0
    for k in range(n):
        deleted = deleted_solution_matrix(H, k)
        separation = mats["K"] - deleted
        candidates = []
        for component in range(min(3, m)):
            sigma = float(np.linalg.norm(separation[component]))
            eta = float(separation[component, k])
            if sigma > 1e-14 and abs(eta) > 1e-14:
                candidates.append((component, sigma * np.sqrt(lambda_local) / abs(eta)))
        if not candidates:
            raise AssertionError(f"No non-degenerate solution-separation direction k={k + 1}")
        component, value = candidates[0]
        selected_component[k] = component
        # Keep the analytic Baarda MDB as the public local-MDB column; the
        # normalized solution-separation value is retained separately.
        local_mdb[k] = float(np.sqrt(lambda_local) / np.sqrt(s_diag[k]))
        solution_mdb[k] = value
        w2 = stats0["W"][:, k] ** 2
        j2 = stats0["J_tilde"][:, k] ** 2
        ss2 = stats0["SS"][:, k, component] ** 2
        local_rows.append({
            "measurement_k": k + 1,
            "direction_identity": f"GPS:{metadata.get('satellite_ids', list(range(1, n + 1)))[k]}",
            "satellite_label": f"G{int(metadata.get('satellite_ids', list(range(1, n + 1)))[k]):02d}",
            "S_kk": float(s_diag[k]),
            "solution_separation_component": "ENU"[component],
            "nondegenerate_solution_components": ";".join("ENU"[i] for i, _ in candidates),
            "empirical_local_false_alarm": float(np.mean(w2 > th["matched_local"])),
            "baarda_local_mdb_sigma": float(local_mdb[k]),
            "jackknife_local_mdb_sigma": float(np.sqrt(lambda_local) * stats0["J_sigma"][k]),
            "solution_separation_local_mdb_sigma": float(value),
            "residual_parity_global_mdb_sigma": float(np.sqrt(lambda_global / s_diag[k])),
            "maximum_local_mdb_identity_error_sigma": float(max(abs(value - local_mdb[k]), abs(np.sqrt(lambda_local) * stats0["J_sigma"][k] - local_mdb[k]))),
            "maximum_squared_statistic_identity_error": float(max(np.max(abs(w2 - j2)), np.max(abs(w2 - ss2)))),
            "local_alpha0": ALPHA0,
            "target_detection_probability": TARGET_PD,
            "scope_note": "specified-direction matched local-test MDB; not a workflow MDB",
        })
        max_mdb_identity_error = max(max_mdb_identity_error, local_rows[-1]["maximum_local_mdb_identity_error_sigma"])
        max_squared_identity_error = max(max_squared_identity_error, local_rows[-1]["maximum_squared_statistic_identity_error"])
    if len(local_rows) != n or (not legacy and len(local_rows) != 11):
        raise AssertionError("Matched-local direction count changed.")
    if not legacy:
        if not np.isclose(s_diag[selected], PUBLISHED_EXPECTED_S55, atol=1e-12, rtol=0):
            raise AssertionError("Frozen S_55 changed.")
        if not np.isclose(local_mdb[selected], PUBLISHED_EXPECTED_LOCAL_MDB, atol=1e-12, rtol=0):
            raise AssertionError("Frozen local MDB changed.")
        if not np.isclose(np.sqrt(lambda_global / s_diag[selected]), PUBLISHED_EXPECTED_GLOBAL_MDB, atol=1e-12, rtol=0):
            raise AssertionError("Frozen global MDB changed.")
        if max_mdb_identity_error > 1e-10 or max_squared_identity_error > 1e-9:
            raise AssertionError("Matched-local detector identity tolerance exceeded.")
    assert_records_finite(local_rows)
    write_csv(result_dir / "matched_local_mdb_by_measurement.csv", local_rows)

    # Main local power: one selected direction, with 21 bias levels.
    local_power_rows: list[dict[str, object]] = []
    local_curves = {name: [] for name in ("baarda", "jackknife", "solution_separation", "theoretical")}
    workflow_curves = {name: [] for name in ("residual_parity", "jackknife_workflow", "solution_separation_hv", "solution_separation_max")}
    workflow_stats: list[dict[str, np.ndarray]] = []
    max_local_spread = 0.0
    max_local_theory_error = 0.0
    previous_theoretical = -np.inf
    direction_id = int(metadata.get("satellite_ids", list(range(1, n + 1)))[selected])
    for bias in BIAS_GRID:
        if legacy:
            z = bias * np.eye(1, n, selected)[0] + rng.normal(size=(TRIALS, n))
        else:
            parent = rng.normal(size=(19, TRIALS, 19))
            parent[np.arange(19), :, np.arange(19)] += bias
            z = parent[:11, :, :11].reshape(11 * TRIALS, 11)
            # Main local published output is the selected fault direction only.
            z = z[selected * TRIALS : (selected + 1) * TRIALS]
        st = compute_statistics(H, z)
        assert_statistics_finite(st)
        workflow_stats.append(st)
        component = selected_component[selected]
        vals = (
            float(np.mean(st["W"][:, selected] ** 2 > th["matched_local"])),
            float(np.mean(st["J_tilde"][:, selected] ** 2 > th["matched_local"])),
            float(np.mean(st["SS"][:, selected, component] ** 2 > th["matched_local"])),
            float(ncx2.sf(th["matched_local"], 1, bias * bias * s_diag[selected])),
        )
        max_local_spread = max(max_local_spread, max(vals[:3]) - min(vals[:3]))
        max_local_theory_error = max(max_local_theory_error, abs(vals[0] - vals[3]))
        if vals[3] + 1e-15 < previous_theoretical:
            raise AssertionError("Theoretical local detection probability is not monotone.")
        previous_theoretical = vals[3]
        for name, val in zip(local_curves, vals):
            local_curves[name].append(val)
        local_power_rows.append({
            "epoch_index": epoch, "configuration": "gps", "measurement_k": selected + 1,
            "direction_identity": f"GPS:{direction_id}", "satellite_label": f"G{direction_id:02d}",
            "selection_rule": "median S_kk before Monte Carlo", "S_kk": float(s_diag[selected]),
            "solution_separation_component": "ENU"[component], "bias_sigma": float(bias),
            "baarda_empirical_local_detection_probability": vals[0],
            "jackknife_empirical_local_detection_probability": vals[1],
            "solution_separation_empirical_local_detection_probability": vals[2],
            "theoretical_local_detection_probability": vals[3], "local_mdb_sigma": float(local_mdb[selected]),
        })
    write_csv(result_dir / "main_epoch3549_local_detection_probability.csv", local_power_rows)
    if len(local_power_rows) != 21:
        raise AssertionError("Main local schema record count changed.")
    if not legacy and (max_local_spread > 1e-12 or max_local_theory_error > 0.025):
        raise AssertionError("Matched empirical power gate failed.")
    if not legacy and not np.isclose(max_local_theory_error, PUBLISHED_EXPECTED_LOCAL_THEORY_MAX_ERROR, atol=1e-12, rtol=0):
        raise AssertionError("Frozen local empirical/theoretical error changed.")
    assert_records_finite(local_power_rows)

    # The workflow curves use one batch per fault direction, matching the
    # direction-conditioned records (4 workflows x 21 levels).
    workflow_rows: list[dict[str, object]] = []
    for st in workflow_stats:
        aa = alarms(st, th)
        ssmax = st["SS"][:, np.arange(n), selected_component]
        vals = [float(np.mean(aa[w])) for w in WORKFLOWS] + [float(np.mean(np.max(ssmax ** 2, axis=1) > th["jackknife_workflow"]))]
        for key, value in zip(workflow_curves, vals): workflow_curves[key].append(value)
    # False-alarm estimates come from the H0 batch and are reused across bias grid.
    ss0max = stats0["SS"][:, np.arange(n), selected_component]
    false_alarm = [float(np.mean(alarms(stats0, th)[w])) for w in WORKFLOWS] + [float(np.mean(np.max(ss0max ** 2, axis=1) > th["jackknife_workflow"]))]
    names = ["Chi-squared/parity space", "Jackknife workflow", "Solution separation H/V split", "Solution separation max-sensitivity"]
    for detector, key, pfa in zip(names, workflow_curves, false_alarm):
        for bias, value in zip(BIAS_GRID, workflow_curves[key]):
            workflow_rows.append({"epoch_index": epoch, "configuration": "gps", "measurement_k": selected + 1, "direction_identity": f"GPS:{direction_id}", "satellite_label": f"G{direction_id:02d}", "selection_rule": "median S_kk before Monte Carlo", "detector": detector, "bias_sigma": float(bias), "empirical_workflow_false_alarm": pfa, "empirical_detection_probability": value})
    write_csv(result_dir / "main_epoch3549_workflow_detection_probability.csv", workflow_rows)
    if len(workflow_rows) != 84:
        raise AssertionError("Workflow schema record count changed.")
    assert_records_finite(workflow_rows)

    # Type-I/II tradeoff at the residual-parity MDB reference bias.
    reference_bias = float(np.sqrt(lambda_global / s_diag[selected]))
    if legacy:
        z0_trade = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, epoch, 7001])).normal(size=(TRIALS, n))
        z1_trade = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, epoch, 7001])).normal(size=(TRIALS, n)); z1_trade[:, selected] += reference_bias
    else:
        trade_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 3549, 7001]))
        z0_trade = trade_rng.normal(size=(TRIALS, n)); z1_trade = trade_rng.normal(size=(TRIALS, n)); z1_trade[:, selected] += reference_bias
    st0_trade, st1_trade = compute_statistics(H, z0_trade), compute_statistics(H, z1_trade)
    assert_statistics_finite(st0_trade)
    assert_statistics_finite(st1_trade)
    trade_rows: list[dict[str, object]] = []
    trade_curves = {w: ([], []) for w in WORKFLOWS}
    for workflow in WORKFLOWS:
        for budget in TRADEOFF_GRID:
            a0, a1 = alarms_for_budget(st0_trade, n, float(budget)), alarms_for_budget(st1_trade, n, float(budget))
            t1, t2 = float(np.mean(a0[workflow])), float(np.mean(~a1[workflow]))
            trade_curves[workflow][0].append(t1); trade_curves[workflow][1].append(t2)
            trade_rows.append({"epoch_index": epoch, "configuration": "gps", "measurement_k": selected + 1, "direction_identity": f"GPS:{direction_id}", "satellite_label": f"G{direction_id:02d}", "selection_rule": "median S_kk before Monte Carlo", "workflow": workflow, "workflow_label": WORKFLOW_LABELS[workflow], "family_wise_false_alarm_budget": float(budget), "reference_bias_sigma": reference_bias, "reference_bias_definition": "residual/parity Pd=0.95 at alpha=0.05", "empirical_type_I_error": t1, "empirical_type_II_error": t2, "trials": TRIALS, "seed_entropy": "20260509;3549;7001"})
        if not np.all(np.diff(trade_curves[workflow][0]) >= -1e-15):
            raise AssertionError(f"{workflow} Type I error is not monotone increasing.")
        if not np.all(np.diff(trade_curves[workflow][1]) <= 1e-15):
            raise AssertionError(f"{workflow} Type II error is not monotone decreasing.")
    if len(trade_rows) != 120:
        raise AssertionError("Tradeoff schema record count changed.")
    assert_records_finite(trade_rows)
    write_csv(result_dir / "main_epoch3549_type_tradeoff.csv", trade_rows)

    # Figures 3--5 retain the public filenames and plotting style.
    def save(path: Path) -> None:
        plt.tight_layout(); plt.savefig(path, metadata={"CreationDate": None, "ModDate": None}); plt.close()
    plt.figure(figsize=(8.6, 5.6));
    for key, style, label in (("baarda", "o-", "Baarda w-test"), ("jackknife", "s--", "Standardized Jackknife"), ("solution_separation", "^:", "Normalized solution separation"), ("theoretical", "--", "Theoretical noncentral chi-square")):
        plt.plot(BIAS_GRID, local_curves[key], style, label=label)
    plt.axhline(TARGET_PD, color="0.35", linestyle=":"); plt.xlabel("Bias magnitude (whitened sigma units)"); plt.ylabel("Local detection probability"); plt.ylim(-.02, 1.02); plt.grid(alpha=.25); plt.legend(); save(figure_dir / "section6_baarda_local_detection_probability.pdf")
    plt.figure(figsize=(8.6, 5.6));
    for key, label in zip(("residual_parity", "jackknife_workflow", "solution_separation_hv", "solution_separation_max"), names): plt.plot(BIAS_GRID, workflow_curves[key], marker="o", label=label)
    plt.axhline(TARGET_PD, color="0.35", linestyle=":"); plt.xlabel("Bias magnitude (whitened sigma units)"); plt.ylabel("Detection probability"); plt.ylim(-.02, 1.02); plt.grid(alpha=.25); plt.legend(); save(figure_dir / "section6_detection_probability.pdf")
    for workflow, filename in zip(WORKFLOWS, ("section6_type_tradeoff_residual_parity.pdf", "section6_type_tradeoff_jackknife_max.pdf", "section6_type_tradeoff_solution_separation.pdf")):
        plt.figure(figsize=(6.8, 4.6)); plt.plot(TRADEOFF_GRID, trade_curves[workflow][0], "o-", label="Type I error"); plt.plot(TRADEOFF_GRID, trade_curves[workflow][1], "s-", label="Type II error"); plt.axvline(ALPHA, color="0.35", linestyle=":"); plt.ylim(-.02, 1.02); plt.xlabel("Family-wise false alarm budget"); plt.ylabel("Empirical error probability"); plt.grid(alpha=.25); plt.legend(); save(figure_dir / filename)

    # Geometry and provenance are deliberately sanitized; no private selection
    # audit or multi-constellation data is copied into the public package.
    satellite_ids = [int(value) for value in metadata.get("satellite_ids", list(range(1, n + 1)))]
    geometry_summary = {
        "run_id": run_id,
        "epoch": epoch,
        "constellation": "GPS",
        "n": n,
        "m": m,
        "q": q,
        "rank_H": int(np.linalg.matrix_rank(H)),
        "condition_number_hth": float(np.linalg.cond(H.T @ H)),
        "min_S_kk": float(np.min(s_diag)),
        "max_S_kk": float(np.max(s_diag)),
        "pdop": metadata.get("pdop"),
        "hdop": metadata.get("hdop"),
        "vdop": metadata.get("vdop"),
        "satellite_ids": satellite_ids,
        "selected_measurement_k": selected + 1,
        "selected_direction_identity": f"GPS:{direction_id}",
        "selected_satellite_label": f"G{direction_id:02d}",
        "selected_S_kk": float(s_diag[selected]),
        "trials": TRIALS,
        "bias_grid_sigma": BIAS_GRID.tolist(),
    }
    (result_dir / "geometry_summary.json").write_text(json.dumps(geometry_summary, indent=2) + "\n", encoding="utf-8")
    nominal = {r["workflow"]: {"empirical_type_I_error": r["empirical_type_I_error"], "empirical_type_II_error": r["empirical_type_II_error"]} for r in trade_rows if abs(float(r["family_wise_false_alarm_budget"]) - 0.05) < 1e-12}
    nominal_points_passed = True
    if not legacy:
        expected_nominal = {
            "residual_parity": {"empirical_type_I_error": 0.0506, "empirical_type_II_error": 0.05},
            "jackknife_workflow": {"empirical_type_I_error": 0.0462, "empirical_type_II_error": 0.0305},
            "solution_separation_hv": {"empirical_type_I_error": 0.0234, "empirical_type_II_error": 0.0493},
        }
        for workflow, expected in expected_nominal.items():
            for key, value in expected.items():
                if not np.isclose(nominal[workflow][key], value, atol=1e-12, rtol=0):
                    nominal_points_passed = False
                    raise AssertionError(f"Frozen nominal tradeoff changed: {workflow}/{key}")
    csv_hashes_passed = True
    if not legacy:
        for filename, expected_hash in PUBLISHED_EXPECTED_CSV_HASHES.items():
            actual_hash = sha256_file(result_dir / filename)
            if actual_hash != expected_hash:
                csv_hashes_passed = False
                raise AssertionError(f"Frozen CSV hash changed: {filename}")
    common_rank_finite = bool(np.linalg.matrix_rank(H) == m and np.isfinite(H).all())
    common_record_counts = bool(
        len(local_rows) == n
        and len(local_power_rows) == 21
        and len(workflow_rows) == 84
        and len(trade_rows) == 120
    )
    common_monotonic = bool(
        np.all(np.diff(local_curves["theoretical"]) >= -1e-15)
        and all(
            np.all(np.diff(trade_curves[workflow][0]) >= -1e-15)
            and np.all(np.diff(trade_curves[workflow][1]) <= 1e-15)
            for workflow in WORKFLOWS
        )
    )
    common_finite = bool(
        common_rank_finite
        and all(np.isfinite(value).all() for value in stats0.values())
        and all(np.isfinite(value).all() for value in st0_trade.values())
        and all(np.isfinite(value).all() for value in st1_trade.values())
    )
    legacy_smoke_passed = bool(
        common_rank_finite and common_record_counts and common_monotonic and common_finite
    )
    if legacy and not legacy_smoke_passed:
        raise AssertionError("Legacy smoke gates failed.")
    if not legacy:
        published_gate_fields = {
            "thresholds_within_1e-12": bool(all(np.isclose(th[k], PUBLISHED_EXPECTED_THRESHOLDS[k], atol=1e-12, rtol=0) for k in PUBLISHED_EXPECTED_THRESHOLDS)),
            "selected_S55_within_1e-12": bool(np.isclose(s_diag[selected], PUBLISHED_EXPECTED_S55, atol=1e-12, rtol=0)),
            "local_mdb_within_1e-12": bool(np.isclose(local_mdb[selected], PUBLISHED_EXPECTED_LOCAL_MDB, atol=1e-12, rtol=0)),
            "global_mdb_within_1e-12": bool(np.isclose(np.sqrt(lambda_global / s_diag[selected]), PUBLISHED_EXPECTED_GLOBAL_MDB, atol=1e-12, rtol=0)),
            "nominal_table8_within_1e-12": nominal_points_passed,
            "csv_oracle_hashes_exact": csv_hashes_passed,
            "all_required_gates_passed": bool(
                max_mdb_identity_error <= 1e-10
                and max_squared_identity_error <= 1e-9
                and max_local_spread <= 1e-12
                and max_local_theory_error <= 0.025
                and common_record_counts
                and common_finite
                and common_monotonic
                and nominal_points_passed
                and csv_hashes_passed
            ),
        }
    else:
        published_gate_fields = {
            "thresholds_within_1e-12": None,
            "selected_S55_within_1e-12": None,
            "local_mdb_within_1e-12": None,
            "global_mdb_within_1e-12": None,
            "nominal_table8_within_1e-12": None,
            "csv_oracle_hashes_exact": None,
            "all_required_gates_passed": None,
        }
    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "generator": "src/validation/simulate_snapshot_validation.py",
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_basename": Path(sys.executable).name,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "geometry_npz_sha256": sha256_file(ROOT / "data/processed" / ("medium_urban_epoch3648_gps" if legacy else "medium_urban_epoch3549_gps") / "geometry_matrix.npz"),
        "geometry_csv_sha256": sha256_file(ROOT / "data/processed" / ("medium_urban_epoch3648_gps" if legacy else "medium_urban_epoch3549_gps") / "geometry.csv"),
        "seed_entropy": [MASTER_SEED, epoch],
        "tradeoff_seed_entropy": [MASTER_SEED, 3549, 7001],
        "simulation": {
            "trials_per_direction": TRIALS,
            "bias_grid_sigma": BIAS_GRID.tolist(),
            "denominator_per_probability": TRIALS,
            "noise_model": "whitened N(0,I)",
            "fault_model": "synthetic single-measurement additive bias",
        },
        "table7_threshold_policy": None if legacy else {
            "thresholds": th,
            "allocations": {
                "residual_parity": "chi2(q, 1-alpha), alpha=0.05",
                "matched_local": "chi2(1, 1-alpha0), alpha0=0.05",
                "jackknife_workflow": "chi2(1, 1-tau/n), tau=0.05",
                "solution_separation_h": "chi2(1, 1-tau_h/(2n)), tau_h=0.025",
                "solution_separation_v": "chi2(1, 1-tau_v/n), tau_v=0.025",
            },
            "workflow_names": list(WORKFLOWS),
        },
        "nominal_tradeoff_budget_0.05": None if legacy else nominal,
        "validation": {
            "full_column_rank": int(np.linalg.matrix_rank(H)) == m,
            "selected_direction": "GPS:9" if not legacy else f"GPS:{direction_id}",
            "scope": "GPS-only synthetic Gaussian/noise and single-measurement bias; no workflow MDB claim",
            "gates": {
                **published_gate_fields,
                "maximum_local_mdb_identity_error_sigma": float(max_mdb_identity_error),
                "maximum_squared_statistic_identity_error": float(max_squared_identity_error),
                "maximum_empirical_matched_power_spread": float(max_local_spread),
                "maximum_empirical_theoretical_power_error": float(max_local_theory_error),
                "record_counts": {"matched_local": len(local_rows), "main_local": len(local_power_rows), "workflow": len(workflow_rows), "tradeoff": len(trade_rows)},
                "all_numeric_finite": common_finite,
                "denominator_per_probability": TRIALS,
                "theoretical_power_monotone": bool(np.all(np.diff(local_curves["theoretical"]) >= -1e-15)),
                "tradeoff_type_i_increasing_type_ii_decreasing": common_monotonic,
                "legacy_smoke_passed": legacy_smoke_passed if legacy else None,
            },
        },
        "limits": ["GPS-only geometry", "synthetic whitened Gaussian noise", "synthetic single-measurement bias", "not a full RAIM benchmark"],
        "outputs": [str(p.relative_to(ROOT)) for p in (
            final_result_dir / "geometry_summary.json",
            final_result_dir / "matched_local_mdb_by_measurement.csv",
            final_result_dir / "main_epoch3549_local_detection_probability.csv",
            final_result_dir / "main_epoch3549_workflow_detection_probability.csv",
            final_result_dir / "main_epoch3549_type_tradeoff.csv",
            final_result_dir / "run_summary.json",
            final_figure_dir / "section6_baarda_local_detection_probability.pdf",
            final_figure_dir / "section6_detection_probability.pdf",
            final_figure_dir / "section6_type_tradeoff_residual_parity.pdf",
            final_figure_dir / "section6_type_tradeoff_jackknife_max.pdf",
            final_figure_dir / "section6_type_tradeoff_solution_separation.pdf",
        )],
    }
    (result_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "geometry": geometry_summary, "nominal_tradeoff_budget_0.05": nominal}, indent=2))


def main_case(H: np.ndarray, metadata: dict[str, object], legacy: bool) -> None:
    """Run into a sibling staging tree and publish atomically after all gates."""
    run_id = "section6_medium_urban_epoch3648_gps" if legacy else "published_epoch3549_gps"
    final_result_dir = ROOT / "results" / run_id
    final_figure_dir = ROOT / "figures" / ("legacy_epoch3648" if legacy else "")
    final_summary = final_result_dir / "run_summary.json"
    final_summary.unlink(missing_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=ROOT))
    stage_result_dir = stage_root / "results" / run_id
    stage_figure_dir = stage_root / "figures" / ("legacy_epoch3648" if legacy else "")
    expected_result_names = {
        "geometry_summary.json",
        "matched_local_mdb_by_measurement.csv",
        "main_epoch3549_local_detection_probability.csv",
        "main_epoch3549_workflow_detection_probability.csv",
        "main_epoch3549_type_tradeoff.csv",
        "run_summary.json",
    }
    expected_figure_names = {
        "section6_baarda_local_detection_probability.pdf",
        "section6_detection_probability.pdf",
        "section6_type_tradeoff_residual_parity.pdf",
        "section6_type_tradeoff_jackknife_max.pdf",
        "section6_type_tradeoff_solution_separation.pdf",
    }
    try:
        _run_case(
            H,
            metadata,
            legacy,
            stage_result_dir,
            stage_figure_dir,
            final_result_dir,
            final_figure_dir,
        )
        staged_results = {path.name for path in stage_result_dir.iterdir() if path.is_file()}
        staged_figures = {path.name for path in stage_figure_dir.iterdir() if path.is_file()}
        if staged_results != expected_result_names or staged_figures != expected_figure_names:
            raise AssertionError("Staged output inventory changed before publication.")
        for path in sorted(stage_root.rglob("*")):
            if not path.is_file() or path.name == "run_summary.json":
                continue
            target = ROOT / path.relative_to(stage_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
        stage_summary = stage_result_dir / "run_summary.json"
        if not stage_summary.is_file():
            raise AssertionError("Staged run summary is missing.")
        final_result_dir.mkdir(parents=True, exist_ok=True)
        os.replace(stage_summary, final_summary)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-epoch3648", action="store_true", help="run the retained pilot geometry")
    args = parser.parse_args()
    run_id = "section6_medium_urban_epoch3648_gps" if args.legacy_epoch3648 else "published_epoch3549_gps"
    (ROOT / "results" / run_id / "run_summary.json").unlink(missing_ok=True)
    H, metadata, _, _ = load_geometry(args.legacy_epoch3648)
    main_case(H, metadata, args.legacy_epoch3648)


if __name__ == "__main__":
    main()
