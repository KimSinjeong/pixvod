#!/usr/bin/env python3
"""Upsample a camera trajectory by ScLERP (screw linear interpolation on SE(3)).

    python replica/sclerp_upsample.py --input traj.txt --output traj_sub9.txt --n 9

Input/output: one pose per line, 16 floats = row-major 4x4 T_wc (camera -> world), the NICE-SLAM Replica format.
Between consecutive poses T0, T1 it inserts N poses T(t) = T0 @ Exp(t * Log(inv(T0) @ T1)), t = k/(N+1), k = 1..N, and
keeps every original pose, so M poses become (M - 1) * (N + 1) + 1 (Replica: 2,000 -> 19,991 for N = 9).

Self-contained copy of jax-practice/sclerp_subsample.py with the functions it used from jax-practice/SceneFlowBlender/
lie.py (unchanged, float64).  Run on the GPU it reproduces the project's traj_sub9.txt files byte for byte (checked on
office2 with JAX 0.6.2 on an RTX 4090); on the CPU the interpolated poses differ in the last bit (|diff| <= 4.4e-16)."""
import argparse

import numpy as np
import jax.numpy as jnp
from jax import config as _jax_config
_jax_config.update('jax_enable_x64', True)
from jax import jit, lax

_EPS = 1e-3
_EPS_PI = 1e-4


def _skew(w):
    wx, wy, wz = w
    return jnp.array([[0.0, -wz,  wy],
                      [wz,  0.0, -wx],
                      [-wy, wx,  0.0]])

def _vee(S):
    return jnp.array([S[2,1], S[0,2], S[1,0]])

def _A(theta):  # sin θ / θ
    def small(_):
        return 1.0 - theta**2/6.0 + theta**4/120.0
    def general(_):
        return jnp.sin(theta) / theta
    return lax.cond(theta < _EPS, small, general, operand=None)

def _B(theta):  # (1 - cos θ)/θ^2
    def small(_):
        return 0.5 - theta**2/24.0 + theta**4/720.0
    def general(_):
        return (1.0 - jnp.cos(theta)) / (theta**2)
    return lax.cond(theta < _EPS, small, general, operand=None)

def _C(theta):  # (θ - sin θ)/θ^3
    def small(_):
        return 1.0/6.0 - theta**2/120.0 + theta**4/5040.0
    def general(_):
        return (theta - jnp.sin(theta)) / (theta**3)
    return lax.cond(theta < _EPS, small, general, operand=None)

@jit
def so3_Exp(w):
    th = jnp.linalg.norm(w)
    W = _skew(w)
    A, B = _A(th), _B(th)
    return jnp.eye(3) + A*W + B*(W @ W)

@jit
def SO3_Log(R):
    Wvee = _vee(0.5 * (R - R.T))               # ~ sin(theta) * axis
    s = jnp.linalg.norm(Wvee)                  # = |sin(theta)|
    c = jnp.clip((jnp.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = jnp.arctan2(s, c)

    def small(_):
        return Wvee

    def general(_):
        sin_th = jnp.sin(theta)
        sin_th = jnp.where(jnp.abs(sin_th) < _EPS, _EPS, sin_th)
        scale = theta / sin_th
        return Wvee * scale

    return lax.cond(s < _EPS, small, general, operand=None)

@jit
def so3_Jl_inv(w):
    th = jnp.linalg.norm(w)
    W = _skew(w)
    I = jnp.eye(3)
    def small(_):
        return I - 0.5*W + (1.0/12.0)*(W @ W)
    def general(_):
        th_safe = jnp.minimum(th, jnp.pi - _EPS_PI)
        s = jnp.sin(th_safe)
        s = jnp.where(jnp.abs(s) < _EPS, _EPS, s)
        coef = (1.0/(th_safe*th_safe)) - (1.0 + jnp.cos(th_safe)) / (2.0*th_safe*s)
        return I - 0.5*W + coef*(W @ W)
    return lax.cond(th < _EPS, small, general, operand=None)

@jit
def se3_Exp(xi):
    w = xi[:3]; v = xi[3:]
    R = so3_Exp(w)
    th = jnp.linalg.norm(w)
    W = _skew(w)
    B, C = _B(th), _C(th)
    Vmat = jnp.eye(3) + B*W + C*(W @ W)
    t = Vmat @ v
    T = jnp.eye(4).at[:3,:3].set(R).at[:3,3].set(t)
    return T

@jit
def SE3_Log(T):
    R = T[:3,:3]
    t = T[:3,3]
    w = SO3_Log(R)
    V_inv = so3_Jl_inv(w)
    v = V_inv @ t
    return jnp.concatenate([w, v])


def ensure_se3(T: np.ndarray) -> np.ndarray:
    """Force bottom row [0,0,0,1] and orthonormalize R lightly if needed."""
    T = T.copy()
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=T.dtype)
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    T[:3, :3] = R_ortho
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    Tin = np.eye(4, dtype=T.dtype)
    Tin[:3, :3] = Rt
    Tin[:3, 3] = -Rt @ t
    return Tin


def parse_traj(txt_path: str) -> np.ndarray:
    mats = []
    with open(txt_path, 'r') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) != 16:
                raise ValueError(f"Each line must have 16 floats; got {len(parts)} in line: {s[:80]}")
            vals = np.array([float(x) for x in parts], dtype=np.float64)
            mats.append(ensure_se3(vals.reshape(4, 4)))
    if not mats:
        raise ValueError("No valid 4x4 matrices found in input.")
    return np.stack(mats, axis=0)


def write_traj(txt_path: str, Ts: np.ndarray):
    with open(txt_path, 'w') as f:
        for T in Ts:
            f.write(" ".join(f"{x:.18e}" for x in T.reshape(-1)) + "\n")


def subsample_traj(Ts: np.ndarray, N: int) -> np.ndarray:
    if N < 0:
        raise ValueError("--n must be >= 0")
    out = []
    for i in range(Ts.shape[0] - 1):
        (T0, T1) = (Ts[i], Ts[i + 1])
        out.append(T0)
        for k in range(1, N + 1):
            t = k / float(N + 1)
            xi = SE3_Log(inv_se3(T0) @ T1)
            out.append(np.array(T0 @ se3_Exp(t * xi)))
    out.append(Ts[-1])
    return np.stack(out, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="trajectory .txt (M x 16 row-major floats, T_wc)")
    ap.add_argument("--output", required=True, help="upsampled trajectory .txt")
    ap.add_argument("--n", type=int, required=True, help="poses inserted between consecutive poses")
    a = ap.parse_args()
    Ts = parse_traj(a.input)
    Ts_sub = subsample_traj(Ts, a.n)
    write_traj(a.output, Ts_sub)
    print(f"[OK] Read {Ts.shape[0]} poses; wrote {Ts_sub.shape[0]} poses to: {a.output}")


if __name__ == "__main__":
    main()
