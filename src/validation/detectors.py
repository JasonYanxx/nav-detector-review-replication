"""Section 3-style detector implementations for the validation harness.

The functions below intentionally follow the detector construction order in the
manuscript. Relationship identities from Section 4 are checked by the validation
script, but they are not used as shortcuts for implementing the detectors.
"""

from __future__ import annotations

import numpy as np


def projection_matrices(H: np.ndarray) -> dict[str, np.ndarray]:
    """Return common least-squares matrices used for setup and audits."""
    n = H.shape[0]
    A = H.T @ H
    A_inv = np.linalg.inv(A)
    K = A_inv @ H.T
    P = H @ K
    S = np.eye(n) - P
    u, _, _ = np.linalg.svd(H, full_matrices=True)
    Q = u[:, H.shape[1] :]
    return {"A": A, "A_inv": A_inv, "K": K, "P": P, "S": S, "Q": Q}


def full_solution(H: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the all-in-view least-squares solution as in Section 3."""
    K = np.linalg.inv(H.T @ H) @ H.T
    X_hat = Z @ K.T
    return X_hat, K


def residual_detector(H: np.ndarray, Z: np.ndarray) -> dict[str, np.ndarray]:
    """Implement the residual/chi-square detector from the fitted solution."""
    X_hat, K = full_solution(H, Z)
    Z_fit = X_hat @ H.T
    R = Z - Z_fit
    T_R = np.einsum("ij,ij->i", R, R)
    S = np.eye(H.shape[0]) - H @ K
    return {"X_hat": X_hat, "K": K, "R": R, "T_R": T_R, "S": S}


def parity_detector(H: np.ndarray, Z: np.ndarray) -> dict[str, np.ndarray]:
    """Implement the parity-space detector from an explicit null-space basis."""
    u, _, _ = np.linalg.svd(H, full_matrices=True)
    Q = u[:, H.shape[1] :]
    Pvec = Z @ Q
    T_P = np.einsum("ij,ij->i", Pvec, Pvec)
    return {"Q": Q, "Pvec": Pvec, "T_P": T_P}


def baarda_w_test(R: np.ndarray, S: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all local Baarda standardized residual statistics."""
    S_diag = np.diag(S)
    W = R / np.sqrt(S_diag)
    return {"W": W, "S_diag": S_diag}


def deleted_solution_matrix(H: np.ndarray, omitted: int) -> np.ndarray:
    """Compute the deleted-measurement solution matrix in full ``n`` columns."""
    n = H.shape[0]
    L = np.eye(n)
    L[omitted, omitted] = 0.0
    return np.linalg.inv(H.T @ L @ H) @ H.T @ L


def jackknife_detector(H: np.ndarray, Z: np.ndarray) -> dict[str, np.ndarray]:
    """Compute leave-one-out prediction residuals by explicit deletion."""
    n = H.shape[0]
    J = np.empty((Z.shape[0], n))
    J_sigma = np.empty(n)
    K_deleted_all = []

    for k in range(n):
        K_deleted = deleted_solution_matrix(H, k)
        X_deleted = Z @ K_deleted.T
        Z_pred_k = X_deleted @ H[k]
        J[:, k] = Z[:, k] - Z_pred_k

        P_deleted = H @ K_deleted
        deleted_residual_row = (np.eye(n) - P_deleted)[k]
        J_sigma[k] = np.linalg.norm(deleted_residual_row)
        K_deleted_all.append(K_deleted)

    J_tilde = J / J_sigma
    return {"J": J, "J_sigma": J_sigma, "J_tilde": J_tilde, "K_deleted_all": K_deleted_all}


def solution_separation_detector(
    H: np.ndarray,
    Z: np.ndarray,
    X_hat: np.ndarray,
    K: np.ndarray,
    K_deleted_all: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute solution separation from all-in-view and deleted solutions."""
    n = H.shape[0]
    m = H.shape[1]
    SS_state = np.empty((Z.shape[0], n, m))
    sigma_state = np.empty((n, m))
    delta_all = np.empty((Z.shape[0], n, H.shape[1]))

    for k, K_deleted in enumerate(K_deleted_all):
        X_deleted = Z @ K_deleted.T
        delta = X_hat - X_deleted
        delta_all[:, k, :] = delta

        delta_operator = K - K_deleted
        sigma = np.sqrt(np.sum(delta_operator**2, axis=1))
        sigma_state[k] = sigma
        SS_state[:, k, :] = delta / sigma

    return {
        "SS": SS_state[:, :, :3],
        "sigma_ss": sigma_state[:, :3],
        "SS_state": SS_state,
        "sigma_state": sigma_state,
        "delta": delta_all,
    }


def compute_statistics(H: np.ndarray, Z: np.ndarray) -> dict[str, np.ndarray]:
    """Compute detector statistics without using Section 4 relationship shortcuts.

    Parameters
    ----------
    H
        Whitened geometry matrix with shape ``(n, m)``.
    Z
        Simulated observations with shape ``(num_trials, n)``.
    """
    residual = residual_detector(H, Z)
    parity = parity_detector(H, Z)
    baarda = baarda_w_test(residual["R"], residual["S"])
    jackknife = jackknife_detector(H, Z)
    solution_separation = solution_separation_detector(
        H,
        Z,
        residual["X_hat"],
        residual["K"],
        jackknife["K_deleted_all"],
    )

    return {
        "T_R": residual["T_R"],
        "T_P": parity["T_P"],
        "R": residual["R"],
        "Pvec": parity["Pvec"],
        "W": baarda["W"],
        "J": jackknife["J"],
        "J_sigma": jackknife["J_sigma"],
        "J_tilde": jackknife["J_tilde"],
        "SS": solution_separation["SS"],
        "SS_state": solution_separation["SS_state"],
        "S_diag": baarda["S_diag"],
        "sigma_ss": solution_separation["sigma_ss"],
        "sigma_state": solution_separation["sigma_state"],
        "delta": solution_separation["delta"],
    }
