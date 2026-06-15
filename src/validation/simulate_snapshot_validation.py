"""Run the Section 6 illustrative validation on the pilot GNSS geometry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

from detectors import compute_statistics, deleted_solution_matrix, projection_matrices


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "data/processed/medium_urban_epoch3648_gps/geometry_matrix.npz"
METADATA = ROOT / "data/processed/medium_urban_epoch3648_gps/metadata.json"
RESULT_DIR = ROOT / "results/section6_medium_urban_epoch3648_gps"
FIG_DIR = ROOT / "figures"
MANIFEST = ROOT / "experiments/manifests/section6_medium_urban_epoch3648_gps.json"


def alarm_rate(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values > threshold))


def first_crossing(xs: np.ndarray, ys: np.ndarray, target: float = 0.95) -> float:
    above = np.flatnonzero(ys >= target)
    if above.size == 0:
        return float("inf")
    idx = int(above[0])
    if idx == 0:
        return float(xs[0])
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def noncentrality_for_detection_probability(
    threshold: float, degrees_of_freedom: int, target: float
) -> float:
    """Solve ``P(chi2_df(lambda) > threshold) = target`` for lambda."""

    def power_gap(noncentrality: float) -> float:
        return float(1 - ncx2.cdf(threshold, degrees_of_freedom, noncentrality) - target)

    upper = 1.0
    while power_gap(upper) < 0.0:
        upper *= 2.0
    return float(brentq(power_gap, 0.0, upper))


def scalar_mdb_from_sensitivity(noncentrality: float, sensitivity: float) -> float:
    if np.isclose(sensitivity, 0.0):
        return float("inf")
    return float(np.sqrt(noncentrality) / abs(sensitivity))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads(METADATA.read_text())
    data = np.load(GEOMETRY, allow_pickle=True)
    H = data["H"]
    n, m = H.shape
    q = n - m
    seed = 20260509
    trials = 10_000
    rng = np.random.default_rng(seed)

    alpha = 0.05
    alpha0 = 0.05
    tau = 0.05
    tau_hor = 0.025
    tau_ver = 0.025
    pd_target = 0.95
    bias_grid = np.arange(0.0, 10.0 + 0.5, 0.5)

    mats = projection_matrices(H)
    S_diag = np.diag(mats["S"])
    order = np.argsort(S_diag)
    selected = {
        "low": int(order[0]),
        "median": int(order[len(order) // 2]),
        "high": int(order[-1]),
    }
    representative_k = selected["median"]

    gamma_global = float(chi2.ppf(1 - alpha, q))
    gamma_w = float(chi2.ppf(1 - alpha0, 1))
    gamma_j = float(chi2.ppf(1 - tau / n, 1))
    gamma_ss_hor = float(chi2.ppf(1 - tau_hor / (2 * n), 1))
    gamma_ss_ver = float(chi2.ppf(1 - tau_ver / n, 1))
    gamma_ss_scalar = float(chi2.ppf(1 - tau / n, 1))
    ss_sensitivity = np.abs(mats["K"][:3, :])
    ss_selected_component = np.argmax(ss_sensitivity, axis=0)
    ss_component_names = np.array(["E", "N", "U"])

    Z0 = rng.normal(size=(trials, n))
    stats0 = compute_statistics(H, Z0)
    ss0 = stats0["SS"]
    ss0_alarm = (
        (ss0[:, :, 0] ** 2 > gamma_ss_hor)
        | (ss0[:, :, 1] ** 2 > gamma_ss_hor)
        | (ss0[:, :, 2] ** 2 > gamma_ss_ver)
    ).any(axis=1)
    ss0_scalar = ss0[:, np.arange(n), ss_selected_component]
    ss0_scalar_alarm = (ss0_scalar**2 > gamma_ss_scalar).any(axis=1)

    sanity_rows = [
        {
            "check_type": "threshold",
            "detector_statistic": "Chi-squared/parity space",
            "threshold_rule": "chi2(q, 1-alpha)",
            "nominal_level": alpha,
            "threshold": gamma_global,
            "empirical_false_alarm": alarm_rate(stats0["T_R"], gamma_global),
            "relationship_residual": "",
            "trials": trials,
        },
        {
            "check_type": "threshold",
            "detector_statistic": "Baarda w-test",
            "threshold_rule": "max_k w_k^2 > chi2(1, 1-alpha0)",
            "nominal_level": alpha0,
            "threshold": gamma_w,
            "empirical_false_alarm": alarm_rate(np.max(stats0["W"] ** 2, axis=1), gamma_w),
            "relationship_residual": "",
            "trials": trials,
        },
        {
            "check_type": "threshold",
            "detector_statistic": "Jackknife maximum",
            "threshold_rule": "chi2(1, 1-tau/n)",
            "nominal_level": tau,
            "threshold": gamma_j,
            "empirical_false_alarm": alarm_rate(np.max(stats0["J_tilde"] ** 2, axis=1), gamma_j),
            "relationship_residual": "",
            "trials": trials,
        },
        {
            "check_type": "threshold",
            "detector_statistic": "Solution separation H/V split",
            "threshold_rule": "horizontal/vertical Bonferroni",
            "nominal_level": tau,
            "threshold": "",
            "empirical_false_alarm": float(np.mean(ss0_alarm)),
            "relationship_residual": "",
            "trials": trials,
        },
        {
            "check_type": "threshold",
            "detector_statistic": "Solution separation max-sensitivity",
            "threshold_rule": "per-k selected component, chi2(1, 1-tau/n)",
            "nominal_level": tau,
            "threshold": gamma_ss_scalar,
            "empirical_false_alarm": float(np.mean(ss0_scalar_alarm)),
            "relationship_residual": "",
            "trials": trials,
        },
        {
            "check_type": "identity",
            "detector_statistic": "Residual vs parity",
            "threshold_rule": "",
            "nominal_level": "",
            "threshold": "",
            "empirical_false_alarm": "",
            "relationship_residual": float(np.max(np.abs(stats0["T_R"] - stats0["T_P"]))),
            "trials": trials,
        },
        {
            "check_type": "identity",
            "detector_statistic": "Baarda w vs standardized jackknife",
            "threshold_rule": "",
            "nominal_level": "",
            "threshold": "",
            "empirical_false_alarm": "",
            "relationship_residual": float(np.max(np.abs(stats0["W"] - stats0["J_tilde"]))),
            "trials": trials,
        },
    ]

    pd_rows: list[dict[str, float | str | int]] = []
    mdb_rows: list[dict[str, float | str | int]] = []
    theoretical_mdb_rows: list[dict[str, float | str | int]] = []
    curves: dict[str, list[float]] = {
        "global": [],
        "baarda": [],
        "jackknife": [],
        "ss": [],
        "ss_scalar": [],
    }
    matched_curves: dict[str, list[float]] = {
        "global": [],
        "baarda": [],
        "jackknife": [],
        "ss": [],
        "ss_scalar": [],
    }
    k = representative_k
    e = np.zeros(n)
    e[k] = 1.0

    for b in bias_grid:
        Z = b * e + rng.normal(size=(trials, n))
        st = compute_statistics(H, Z)
        global_alarm = st["T_R"] > gamma_global
        baarda_alarm = np.max(st["W"] ** 2, axis=1) > gamma_w
        jackknife_alarm = np.max(st["J_tilde"] ** 2, axis=1) > gamma_j
        ss_all = st["SS"]
        ss_alarm = (
            (ss_all[:, :, 0] ** 2 > gamma_ss_hor)
            | (ss_all[:, :, 1] ** 2 > gamma_ss_hor)
            | (ss_all[:, :, 2] ** 2 > gamma_ss_ver)
        ).any(axis=1)
        ss_scalar = ss_all[:, np.arange(n), ss_selected_component]
        ss_scalar_alarm = (ss_scalar**2 > gamma_ss_scalar).any(axis=1)
        curves["global"].append(float(np.mean(global_alarm)))
        curves["baarda"].append(float(np.mean(baarda_alarm)))
        curves["jackknife"].append(float(np.mean(jackknife_alarm)))
        curves["ss"].append(float(np.mean(ss_alarm)))
        curves["ss_scalar"].append(float(np.mean(ss_scalar_alarm)))
        matched_curves["global"].append(curves["global"][-1])
        matched_curves["baarda"].append(
            float(np.mean(st["W"][:, k] ** 2 > gamma_w))
        )
        matched_curves["jackknife"].append(
            float(np.mean(st["J_tilde"][:, k] ** 2 > gamma_j))
        )
        matched_ss_alarm = (
            (ss_all[:, k, 0] ** 2 > gamma_ss_hor)
            | (ss_all[:, k, 1] ** 2 > gamma_ss_hor)
            | (ss_all[:, k, 2] ** 2 > gamma_ss_ver)
        )
        matched_curves["ss"].append(float(np.mean(matched_ss_alarm)))
        matched_curves["ss_scalar"].append(
            float(np.mean(ss_all[:, k, ss_selected_component[k]] ** 2 > gamma_ss_scalar))
        )
        pd_rows.append(
            {
                "bias_sigma": float(b),
                "k": int(k + 1),
                "global_empirical": curves["global"][-1],
                "baarda_empirical": curves["baarda"][-1],
                "jackknife_empirical": curves["jackknife"][-1],
                "ss_empirical": curves["ss"][-1],
                "ss_scalar_empirical": curves["ss_scalar"][-1],
                "global_matched_empirical": matched_curves["global"][-1],
                "baarda_matched_empirical": matched_curves["baarda"][-1],
                "jackknife_matched_empirical": matched_curves["jackknife"][-1],
                "ss_matched_empirical": matched_curves["ss"][-1],
                "ss_scalar_matched_empirical": matched_curves["ss_scalar"][-1],
            }
        )

    lambda_global = noncentrality_for_detection_probability(gamma_global, q, pd_target)
    lambda_w = noncentrality_for_detection_probability(gamma_w, 1, pd_target)
    lambda_j = noncentrality_for_detection_probability(gamma_j, 1, pd_target)
    lambda_ss_hor = noncentrality_for_detection_probability(gamma_ss_hor, 1, pd_target)
    lambda_ss_ver = noncentrality_for_detection_probability(gamma_ss_ver, 1, pd_target)
    lambda_ss_scalar = noncentrality_for_detection_probability(gamma_ss_scalar, 1, pd_target)
    theoretical_mdb_by_detector: dict[str, list[float]] = {
        "Chi-squared/parity space": [],
        "Baarda w-test": [],
        "Jackknife maximum": [],
        "Solution separation H/V split": [],
        "Solution separation max-sensitivity": [],
    }
    for k_all in range(n):
        sqrt_skk = float(np.sqrt(S_diag[k_all]))
        K_deleted = deleted_solution_matrix(H, k_all)
        separation_operator = mats["K"] - K_deleted
        directional_mdbs = []
        for component_idx in range(3):
            sigma_delta = float(np.linalg.norm(separation_operator[component_idx, :]))
            sensitivity = (
                0.0 if np.isclose(sigma_delta, 0.0)
                else float(separation_operator[component_idx, k_all] / sigma_delta)
            )
            component_lambda = lambda_ss_ver if component_idx == 2 else lambda_ss_hor
            directional_mdbs.append(
                scalar_mdb_from_sensitivity(component_lambda, sensitivity)
            )
        selected_component = int(ss_selected_component[k_all])
        selected_sigma = float(np.linalg.norm(separation_operator[selected_component, :]))
        selected_sensitivity = (
            0.0 if np.isclose(selected_sigma, 0.0)
            else float(separation_operator[selected_component, k_all] / selected_sigma)
        )
        per_detector_mdbs = {
            "Chi-squared/parity space": scalar_mdb_from_sensitivity(
                lambda_global, sqrt_skk
            ),
            "Baarda w-test": scalar_mdb_from_sensitivity(lambda_w, sqrt_skk),
            "Jackknife maximum": scalar_mdb_from_sensitivity(lambda_j, sqrt_skk),
            "Solution separation H/V split": float(np.min(directional_mdbs)),
            "Solution separation max-sensitivity": scalar_mdb_from_sensitivity(
                lambda_ss_scalar, selected_sensitivity
            ),
        }
        for detector_name, mdb_value in per_detector_mdbs.items():
            theoretical_mdb_by_detector[detector_name].append(mdb_value)
            theoretical_mdb_rows.append(
                {
                    "detector": detector_name,
                    "measurement_k": int(k_all + 1),
                    "S_kk": float(S_diag[k_all]),
                    "theoretical_MDB_k_sigma": mdb_value,
                }
            )

    statistic_forms = {
        "Chi-squared/parity space": r"$T_R=\mathbf{r}^T\mathbf{r}=\mathbf{p}^T\mathbf{p}$",
        "Baarda w-test": r"$\max_k w_k^2$",
        "Jackknife maximum": r"$\max_k \tilde{J}_k^2$",
        "Solution separation H/V split": r"$\{T_{SS,k,q}: k=1,\ldots,n,\ q=1,2,3\}$",
        "Solution separation max-sensitivity": r"$\max_k T_{SS,k,q(k)}$",
    }
    worst_case_mdb = {
        detector_name: float(np.max(mdb_values))
        for detector_name, mdb_values in theoretical_mdb_by_detector.items()
    }
    theoretical_mdb_at_selected_k = {
        detector_name: float(mdb_values[k])
        for detector_name, mdb_values in theoretical_mdb_by_detector.items()
    }

    mdb_rows.extend(
        [
            {
                "detector": "Chi-squared/parity space",
                "statistic_form": statistic_forms["Chi-squared/parity space"],
                "selected_measurement_k": int(k + 1),
                "direction_or_rule": "global",
                "S_kk": float(S_diag[k]),
                "threshold": gamma_global,
                "false_alarm_policy": "alpha=0.05",
                "theoretical_worst_case_workflow_MDB_sigma": worst_case_mdb[
                    "Chi-squared/parity space"
                ],
                "theoretical_MDB_at_selected_k_sigma": theoretical_mdb_at_selected_k[
                    "Chi-squared/parity space"
                ],
            },
            {
                "detector": "Baarda w-test",
                "statistic_form": statistic_forms["Baarda w-test"],
                "selected_measurement_k": int(k + 1),
                "direction_or_rule": "max over k with local threshold",
                "S_kk": float(S_diag[k]),
                "threshold": gamma_w,
                "false_alarm_policy": "alpha0=0.05",
                "theoretical_worst_case_workflow_MDB_sigma": worst_case_mdb[
                    "Baarda w-test"
                ],
                "theoretical_MDB_at_selected_k_sigma": theoretical_mdb_at_selected_k[
                    "Baarda w-test"
                ],
            },
            {
                "detector": "Jackknife maximum",
                "statistic_form": statistic_forms["Jackknife maximum"],
                "selected_measurement_k": int(k + 1),
                "direction_or_rule": "max over k",
                "S_kk": float(S_diag[k]),
                "threshold": gamma_j,
                "false_alarm_policy": "tau=0.05",
                "theoretical_worst_case_workflow_MDB_sigma": worst_case_mdb[
                    "Jackknife maximum"
                ],
                "theoretical_MDB_at_selected_k_sigma": theoretical_mdb_at_selected_k[
                    "Jackknife maximum"
                ],
            },
            {
                "detector": "Solution separation H/V split",
                "statistic_form": statistic_forms["Solution separation H/V split"],
                "selected_measurement_k": int(k + 1),
                "direction_or_rule": "all-k ENU aggregate",
                "S_kk": float(S_diag[k]),
                "threshold": "hor/ver",
                "false_alarm_policy": "tau_hor=0.025, tau_ver=0.025",
                "theoretical_worst_case_workflow_MDB_sigma": worst_case_mdb[
                    "Solution separation H/V split"
                ],
                "theoretical_MDB_at_selected_k_sigma": theoretical_mdb_at_selected_k[
                    "Solution separation H/V split"
                ],
            },
            {
                "detector": "Solution separation max-sensitivity",
                "statistic_form": statistic_forms["Solution separation max-sensitivity"],
                "selected_measurement_k": int(k + 1),
                "direction_or_rule": "per-k max-sensitivity component",
                "S_kk": float(S_diag[k]),
                "threshold": gamma_ss_scalar,
                "false_alarm_policy": "tau=0.05 over selected scalar monitors",
                "theoretical_worst_case_workflow_MDB_sigma": worst_case_mdb[
                    "Solution separation max-sensitivity"
                ],
                "theoretical_MDB_at_selected_k_sigma": theoretical_mdb_at_selected_k[
                    "Solution separation max-sensitivity"
                ],
            },
        ]
    )

    for name, rows in [
        ("sanity_summary.csv", sanity_rows),
        ("detection_probability.csv", pd_rows),
        ("mdb_summary.csv", mdb_rows),
        ("theoretical_mdb_by_measurement.csv", theoretical_mdb_rows),
    ]:
        with (RESULT_DIR / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    geometry_summary = {
        "epoch": meta["epoch_index"],
        "constellation": "GPS",
        "n": n,
        "m": m,
        "q": q,
        "rank_H": int(np.linalg.matrix_rank(H)),
        "cond_HtH": float(np.linalg.cond(H.T @ H)),
        "min_Skk": float(np.min(S_diag)),
        "max_Skk": float(np.max(S_diag)),
        "selected_low_k": int(selected["low"] + 1),
        "selected_median_k": int(selected["median"] + 1),
        "selected_high_k": int(selected["high"] + 1),
    }
    (RESULT_DIR / "geometry_summary.json").write_text(json.dumps(geometry_summary, indent=2))

    plt.figure(figsize=(8.6, 5.6))
    plt.plot(
        bias_grid,
        curves["global"],
        "o-",
        linewidth=2.0,
        markersize=4.5,
        label="Chi-squared/parity space",
    )
    plt.plot(
        bias_grid,
        curves["baarda"],
        "s-",
        linewidth=2.0,
        markersize=4.5,
        label="Baarda w-test",
    )
    plt.plot(
        bias_grid,
        curves["jackknife"],
        "^-",
        linewidth=2.0,
        markersize=4.5,
        label="Jackknife maximum",
    )
    plt.plot(
        bias_grid,
        curves["ss"],
        "d-",
        linewidth=2.0,
        markersize=4.5,
        label="Solution separation H/V split",
    )
    plt.plot(
        bias_grid,
        curves["ss_scalar"],
        "P-",
        linewidth=2.0,
        markersize=4.8,
        label="Solution separation max-sensitivity",
    )
    detection_axis_fontsize = 14
    plt.axhline(pd_target, color="0.35", linestyle=":", linewidth=1.5)
    plt.xlabel("Bias magnitude (whitened sigma units)", fontsize=detection_axis_fontsize)
    plt.ylabel("Detection probability", fontsize=detection_axis_fontsize)
    plt.ylim(-0.02, 1.02)
    plt.xticks(fontsize=detection_axis_fontsize)
    plt.yticks(fontsize=detection_axis_fontsize)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=14, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "section6_detection_probability.pdf")
    plt.close()

    def alarms_for_policy(st: dict[str, np.ndarray], false_alarm_level: float) -> dict[str, np.ndarray]:
        ss = st["SS"]
        ss_gamma_hor = chi2.ppf(1 - false_alarm_level / (4 * n), 1)
        ss_gamma_ver = chi2.ppf(1 - false_alarm_level / (2 * n), 1)
        return {
            "Chi-squared/parity space": st["T_R"] > chi2.ppf(1 - false_alarm_level, q),
            "Baarda w-test": np.max(st["W"] ** 2, axis=1)
            > chi2.ppf(1 - false_alarm_level, 1),
            "Jackknife maximum": np.max(st["J_tilde"] ** 2, axis=1)
            > chi2.ppf(1 - false_alarm_level / n, 1),
            "Solution separation H/V split": (
                (ss[:, :, 0] ** 2 > ss_gamma_hor)
                | (ss[:, :, 1] ** 2 > ss_gamma_hor)
                | (ss[:, :, 2] ** 2 > ss_gamma_ver)
            ).any(axis=1),
        }

    alphas = np.linspace(0.005, 0.2, 40)
    trade_detectors = [
        (
            "Chi-squared/parity space",
            "residual_parity",
            "Chi-squared/parity space",
            float(mdb_rows[0]["theoretical_MDB_at_selected_k_sigma"]),
        ),
        (
            "Baarda w-test",
            "baarda_local",
            "Baarda w-test",
            float(mdb_rows[1]["theoretical_MDB_at_selected_k_sigma"]),
        ),
        (
            "Jackknife maximum",
            "jackknife_max",
            "Jackknife maximum / SS max-sensitivity",
            float(mdb_rows[2]["theoretical_MDB_at_selected_k_sigma"]),
        ),
        (
            "Solution separation H/V split",
            "solution_separation",
            "Solution separation H/V split",
            float(mdb_rows[3]["theoretical_MDB_at_selected_k_sigma"]),
        ),
    ]
    trade_rows: list[dict[str, float | str]] = []
    trade_curves: dict[str, dict[str, list[float]]] = {}
    trade_figures: list[Path] = []
    for detector_name, detector_slug, figure_title, mdb_k3_bias in trade_detectors:
        Z1_trade = mdb_k3_bias * e + rng.normal(size=(trials, n))
        st1_trade = compute_statistics(H, Z1_trade)
        trade_curves[detector_name] = {"type1": [], "type2": []}
        for a in alphas:
            h0_alarm = alarms_for_policy(stats0, float(a))[detector_name]
            h1_alarm = alarms_for_policy(st1_trade, float(a))[detector_name]
            type1 = float(np.mean(h0_alarm))
            type2 = float(np.mean(~h1_alarm))
            trade_curves[detector_name]["type1"].append(type1)
            trade_curves[detector_name]["type2"].append(type2)
            trade_rows.append(
                {
                    "detector": detector_name,
                    "nominal_false_alarm_parameter": float(a),
                    "exact_mdb_k3_bias_sigma": mdb_k3_bias,
                    "empirical_type_I_error": type1,
                    "empirical_type_II_error": type2,
                }
            )

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot(
            alphas,
            trade_curves[detector_name]["type1"],
            "o-",
            linewidth=2.0,
            markersize=4.8,
            label="Type I error",
        )
        ax.plot(
            alphas,
            trade_curves[detector_name]["type2"],
            "s-",
            linewidth=2.0,
            markersize=4.8,
            label="Type II error",
        )
        ax.axvline(alpha, color="0.35", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Family-wise false alarm budget", fontsize=20)
        ax.set_ylabel("Empirical error probability", fontsize=20)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=14, loc="best", framealpha=0.95)
        ax.set_title(figure_title, fontsize=20)
        fig.tight_layout()
        figure_path = FIG_DIR / f"section6_type_tradeoff_{detector_slug}.pdf"
        fig.savefig(figure_path)
        plt.close(fig)
        trade_figures.append(figure_path)

    with (RESULT_DIR / "type_tradeoff.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows)

    manifest = {
        "run_id": "section6_medium_urban_epoch3648_gps",
        "geometry": str(GEOMETRY.relative_to(ROOT)),
        "metadata": str(METADATA.relative_to(ROOT)),
        "seed": seed,
        "trials": trials,
        "bias_grid": bias_grid.tolist(),
        "alpha": alpha,
        "alpha0": alpha0,
        "tau": tau,
        "tau_hor": tau_hor,
        "tau_ver": tau_ver,
        "ss_selected_components_one_based": {
            str(idx + 1): ss_component_names[ss_selected_component[idx]].item()
            for idx in range(n)
        },
        "selected_measurements_one_based": {name: int(idx + 1) for name, idx in selected.items()},
        "outputs": [
            str((RESULT_DIR / "geometry_summary.json").relative_to(ROOT)),
            str((RESULT_DIR / "sanity_summary.csv").relative_to(ROOT)),
            str((RESULT_DIR / "detection_probability.csv").relative_to(ROOT)),
            str((RESULT_DIR / "mdb_summary.csv").relative_to(ROOT)),
            str((RESULT_DIR / "theoretical_mdb_by_measurement.csv").relative_to(ROOT)),
            str((RESULT_DIR / "type_tradeoff.csv").relative_to(ROOT)),
            str((FIG_DIR / "section6_detection_probability.pdf").relative_to(ROOT)),
        ]
        + [str(path.relative_to(ROOT)) for path in trade_figures],
        "limits": "Pilot GPS-only geometry; synthetic Gaussian noise and synthetic single-measurement bias only.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({"geometry": geometry_summary, "mdb": mdb_rows}, indent=2))


if __name__ == "__main__":
    main()
