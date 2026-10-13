"""Numerical helpers for residual weighting and energy reporting."""
from __future__ import annotations
import math
import numpy as np

def predictive_denominator(j_rho: np.ndarray, v_rho: np.ndarray, sigma_i: float, sigma_warp: float) -> np.ndarray:
    """D_i = sigma_I^2 + J_rho,i^2 * V_rho,i + sigma_warp^2 (per-pixel, scalar residual).

    Non-finite or negative inputs contribute 0 to the propagated term, so D >= sigma_I^2.
    """
    j2 = np.asarray(j_rho, dtype=np.float64) ** 2
    v = np.asarray(v_rho, dtype=np.float64)
    prop = np.where(np.isfinite(j2) & np.isfinite(v) & (v > 0.0), j2 * v, 0.0)
    return float(sigma_i) ** 2 + prop + float(sigma_warp) ** 2

def tdist_weight_std(r: np.ndarray, denom: np.ndarray, nu: float) -> np.ndarray:
    """w_i = (nu+1)/(nu + z_i^2) with z_i^2 = r_i^2 / D_i; non-finite residual -> w=1."""
    r = np.asarray(r, dtype=np.float64)
    d = np.maximum(np.asarray(denom, dtype=np.float64), 1e-300)
    z2 = r * r / d
    return np.where(np.isfinite(r), (float(nu) + 1.0) / (float(nu) + z2), 1.0)

def stage_index(stage_label) -> int:
    """1-based chain stage index for the continuation ramp.  Only 'subNN' labels
    carry a chain position; anything else ('main', final replay, pair-prefixed
    labels) returns a large index so nu_effective yields the TARGET nu there —
    never the L2 end of the ramp ('main' used to parse to 0)."""
    s = str(stage_label)
    if s.startswith('sub'):
        digits = ''.join((ch for ch in s[3:] if ch.isdigit()))
        if digits:
            return int(digits)
    return 10 ** 9

def nu_effective(stage_idx: int, nu_target: float, cont_stages: int) -> float:
    """Continuation (3.3): linear ramp of 1/nu from 0 (~L2) to 1/nu_target over
    cont_stages stages. cont_stages <= 0 disables (returns nu_target). stage_idx is
    1-based; stage_idx >= cont_stages -> nu_target exactly."""
    if cont_stages <= 0 or stage_idx >= cont_stages:
        return float(nu_target)
    frac = max(0.0, float(stage_idx)) / float(cont_stages)
    inv = frac / float(nu_target)
    return float(1.0 / inv) if inv > 0.0 else 1000000000000.0

def gauge_axis_factors(c: float, d_op: int=7) -> np.ndarray:
    """Diagonal of A for x' = A x: rot unchanged, translation x 1/c, rho x c.
    (rho' = c rho <=> d' = d/c; projections are invariant when t' = t/c.)"""
    a = np.ones((int(d_op),), dtype=np.float64)
    a[3:6] = 1.0 / float(c)
    if d_op >= 7:
        a[6] = float(c)
    return a

def irls_energies(r: np.ndarray, w: np.ndarray, s0: float, active: np.ndarray) -> dict[str, float]:
    """Quadratic photometric energies over the active set:
    E_data_w1 = sum r^2/s0^2 (unweighted), E_data_w = sum w r^2/s0^2 (IRLS)."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    m = np.asarray(active, dtype=bool).reshape(-1) & np.isfinite(r) & np.isfinite(w)
    e = (r[m] / float(s0)) ** 2
    return {'E_data_w1': float(np.sum(e)), 'E_data_w': float(np.sum(w[m] * e)), 'n': float(m.sum())}
