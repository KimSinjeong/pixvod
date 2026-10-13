"""PixVOD dataset runner. python run_pixvod.py runs the default gates of gbp_engine/defaults.py (champion,
float32, TF32; PIXVOD_BASELINE=legacy = the previous default) with the parameters in configs/replica.yaml."""
import os
import sys
import argparse
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import cv2
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')
if str(Path(__file__).resolve().parent) not in sys.path:  # as REPO_ROOT below, so the defaults load from this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import gbp_engine.defaults  # sets the default gates (champion, float32, TF32) before any gate below is read
_PLOT_EVERY = max(1, int(os.environ.get('PIXVOD_PLOT_EVERY', '1') or '1'))
# PIXVOD_RERUN=1: stream the run live to a Rerun viewer on 127.0.0.1 (rerun_live.py); output only, unset or 0 = off.
_RERUN = os.environ.get('PIXVOD_RERUN', '0')
if _RERUN not in ('0', '1'):
    raise SystemExit(f'PIXVOD_RERUN must be 0 or 1; got {_RERUN!r}')
_RERUN = _RERUN == '1'
# Accelerations (gbp_engine/defaults.py ACCEL, all on by default, PIXVOD_ACCEL=0 = all off).
from gbp_engine.defaults import gate as _accel_gate
import trend_plots as _trend_plots
_ASYNC_PLOT = _accel_gate('PIXVOD_ASYNC_PLOT')
_DIAG_FETCH = _accel_gate('PIXVOD_DIAG_FETCH')
_STAGE_CONST = _accel_gate('PIXVOD_STAGE_CONST')
_COND_SPLIT = _accel_gate('PIXVOD_COND_SPLIT')
_DEV_REFRESH = _accel_gate('PIXVOD_DEV_REFRESH')
_DONATE = _accel_gate('PIXVOD_DONATE')
# PIXVOD_DONATE: a KeyboardInterrupt raised while a donating call is in flight would leave self.vars pointing at donated
# (deleted) buffers, so with donation the first Ctrl-C is deferred to the next iteration boundary (a second one stops at once).
_SIGINT_PENDING = [False]


def _sigint_deferred(signum, frame):
    if _SIGINT_PENDING[0]:
        raise KeyboardInterrupt
    _SIGINT_PENDING[0] = True
    print('[interrupt] Ctrl-C received; stopping at the next GBP iteration boundary (press again to stop at once)', flush=True)


def _raise_if_sigint_pending():
    if _SIGINT_PENDING[0]:
        _SIGINT_PENDING[0] = False
        raise KeyboardInterrupt
_SMALLSOLVE = _accel_gate('PIXVOD_SMALLSOLVE')
# PIXVOD_UNCOMMITTED: jax.device_put(x, jax.devices('gpu')[0]) commits x to GPU 0, and jit outputs computed from a committed
# input are committed too; the committed-or-not pattern of the arguments is part of jit's cache key, so the updates were
# compiled again (several times over the first two stages) as committed arrays spread through the factor sets.  Putting
# these arrays on the default device (GPU 0, the same device) without committing them keeps every argument uncommitted.
_put_gpu = (lambda x: jax.device_put(x)) if _accel_gate('PIXVOD_UNCOMMITTED') else (lambda x: jax.device_put(x, jax.devices('gpu')[0]))
_SKIP_EMPTY_PORTS = os.environ.get('PIXVOD_SKIP_EMPTY_PORTS', '1') != '0'
# Reference-unit (scale-invariant) options.  All unset = legacy.
#   PIXVOD_INIT_SCALE=k      exact move of the initial problem by the scale gauge (initial belief mean and prior
#                            anchor: rho x k, translation x 1/k); k a power of two in [2^-8, 2^8].
#   PIXVOD_EXACT_READOUT=1   stage-boundary readout = exact belief mean Lam^-1 eta (no damping).
#   PIXVOD_REF_GAUGE=tokens  evaluate fixed-unit terms in reference units, where the build-time median leaf prior
#                            anchor g0 is 1 (identical to legacy when g0 = 1): prior_rho, prior_t, between_t
#                            (prior / between precisions), marg (marginalisation loading), safe (safe_mu threshold
#                            and ridge in retract / rebase).
_INIT_K = os.environ.get('PIXVOD_INIT_SCALE')
if _INIT_K is not None:
    try:
        _INIT_K = float(_INIT_K)
    except ValueError:
        raise SystemExit(f'PIXVOD_INIT_SCALE must be a power of two in [2^-8, 2^8]; got {_INIT_K!r}')
    import math as _math
    if not (_math.isfinite(_INIT_K) and _INIT_K > 0.0 and _math.frexp(_INIT_K)[0] == 0.5 and 2.0 ** -8 <= _INIT_K <= 2.0 ** 8):
        raise SystemExit(f'PIXVOD_INIT_SCALE must be a power of two in [2^-8, 2^8]; got {_INIT_K!r}')
_EXACT_READOUT = os.environ.get('PIXVOD_EXACT_READOUT', '0')
if _EXACT_READOUT not in ('0', '1'):
    raise SystemExit(f'PIXVOD_EXACT_READOUT must be 0 or 1; got {_EXACT_READOUT!r}')
_EXACT_READOUT = _EXACT_READOUT == '1'
# PIXVOD_READOUT_GUARD: 1 -> where the exact readout's solve is not finite (a singular belief), _mu_exact
# returns the damped readout g.mu() for that variable; every finite readout is unchanged.  Logs '[READOUT-GUARD]' when used.
_READOUT_GUARD = os.environ.get('PIXVOD_READOUT_GUARD', '0')
if _READOUT_GUARD not in ('0', '1'):
    raise SystemExit(f'PIXVOD_READOUT_GUARD must be 0 or 1; got {_READOUT_GUARD!r}')
_READOUT_GUARD = _READOUT_GUARD == '1'
_REF_GAUGE = os.environ.get('PIXVOD_REF_GAUGE')
if _REF_GAUGE is not None:
    _rg_list = _REF_GAUGE.split(',')
    if not _rg_list or any((t not in ('prior_rho', 'prior_t', 'between_t', 'marg', 'safe') for t in _rg_list)) or len(set(_rg_list)) != len(_rg_list):
        raise SystemExit(f'PIXVOD_REF_GAUGE must be distinct comma-separated tokens from prior_rho, prior_t, between_t, marg, safe; got {_REF_GAUGE!r}')
    _REF_GAUGE = frozenset(_rg_list)
else:
    _REF_GAUGE = frozenset()
if _INIT_K is not None or _EXACT_READOUT or _REF_GAUGE:
    print(f'[SCALE-INV] init_scale={_INIT_K} exact_readout={int(_EXACT_READOUT)} ref_gauge={",".join(sorted(_REF_GAUGE)) or "-"}', flush=True)
# BINI weights: bilateral normal-integration weights (Cao et al., ECCV 2022) on the normal factors.
#   PIXVOD_BINI=1      each normal factor keeps its total precision 2/sigma_n^2 but splits it as diag(2w, 2(1-w)) between
#                      its backward (left/up) and forward (right/down) one-sided components, with
#                      w = sigmoid(k f^2 (h_fwd^2 - h_bwd^2)) from the current depth: BINI's rule (the side with the
#                      smaller one-sided log-depth slope gets the weight).  Our h is BINI's slope term divided by the
#                      focal length f, so f^2 restores BINI's pixel units.  Recomputed every 100 prewarm iterations and
#                      once at each stage entry, held fixed in between (BINI's solve-then-reweight alternation).
#   PIXVOD_BINI_K=<k>  BINI's k in its own pixel units; default 2 (the paper's value).  Other values: unit checks only.
import math as _math
_BINI = os.environ.get('PIXVOD_BINI', '0')
if _BINI not in ('0', '1'):
    raise SystemExit(f'PIXVOD_BINI must be 0 or 1; got {_BINI!r}')
_BINI = _BINI == '1'
try:
    _BINI_K = float(os.environ.get('PIXVOD_BINI_K', '2'))
except ValueError:
    raise SystemExit(f'PIXVOD_BINI_K must be a finite number >= 0; got {os.environ.get("PIXVOD_BINI_K")!r}')
if not (_math.isfinite(_BINI_K) and _BINI_K >= 0.0):
    raise SystemExit(f'PIXVOD_BINI_K must be a finite number >= 0; got {_BINI_K!r}')
if _BINI:
    print(f'[BINI] k={_BINI_K!r} (BINI pixel units), reweight every 100 prewarm iterations and at each stage entry', flush=True)
# Border normal factors.  BINI's discretisation (generate_dx_dy over the whole mask) gives every pixel its own
# one-sided normal residual towards each existing 4-neighbour.  The legacy configuration builds normal factors for interior pixels
# only, so a border pixel had a single link (one copy, through its interior neighbour's factor) and a corner none.
#   PIXVOD_NORMAL_BORDER=1 adds, for the border pixels only:
#     nb3  a two-sided factor centred on each non-corner border pixel along the border (vertical on the left/right
#          columns, horizontal on the top/bottom rows): the interior factor's residual, measurement and ports;
#     nb1  a one-sided factor for each other direction that has a neighbour (the inward link of every border pixel, both
#          links of every corner): h = s nz_p (log rho_p - log rho_q) against N/f, s = +1 towards left/up (the interior
#          factor's first component), -1 towards right/down (its second), same floor clamp and gating.
#   Ports follow the interior layout (centre 7/10, as left/up neighbour 8/11, as right/down neighbour 9/12); every one of
#   them was unused before.  The new factors are registered in adj_factor_idx (variable-to-factor messages) and in the
#   occupied-slot list of the accelerated rebase, use the same sigma_n, robust setting and reference-unit loading as the
#   interior normal factors, and are updated right after them.  Under PIXVOD_BINI a one-sided row gets BINI's weight for
#   a pixel whose other side is missing (that side counts as slope 0): precision 2 sigmoid(-k f^2 h^2) / sigma_n^2.
_NB = os.environ.get('PIXVOD_NORMAL_BORDER', '0')
if _NB not in ('0', '1'):
    raise SystemExit(f'PIXVOD_NORMAL_BORDER must be 0 or 1; got {_NB!r}')
_NB = _NB == '1'
if _NB:
    print('[NORMAL-BORDER] one-sided normal residuals for every border pixel and direction with a neighbour', flush=True)
#   PIXVOD_NORMAL_BORDER_SETS=nb3|nb1|nb3,nb1  (default both; only with PIXVOD_NORMAL_BORDER=1) builds only the named
#   border sets -- a decomposition switch: nb3 = the along-border two-sided factors, nb1 = the one-sided inward/corner rows.
_NB_SETS = os.environ.get('PIXVOD_NORMAL_BORDER_SETS')
if _NB_SETS is not None:
    _toks = _NB_SETS.split(',')
    if not _NB or not _toks or len(set(_toks)) != len(_toks) or not set(_toks) <= {'nb3', 'nb1'}:
        raise SystemExit(f'PIXVOD_NORMAL_BORDER_SETS needs PIXVOD_NORMAL_BORDER=1 and distinct tokens from nb3, nb1; got {_NB_SETS!r}')
    _NB_SETS = frozenset(_toks)
    print(f'[NORMAL-BORDER] sets {sorted(_NB_SETS)}', flush=True)
else:
    _NB_SETS = frozenset(('nb3', 'nb1'))
# Floor handling of the normal residuals.  h_J_fn_normal_ipd clamps log rho at the 1e-6 floor and zeroes the
# Jacobian of a variable at or below it, so a floored pixel enters every normal residual that touches it as the constant
# log(1e-6) and pulls the other variables of that residual towards it.
#   PIXVOD_NORMAL_FLOOR=invalid  a residual component that involves a variable at or below the floor is undefined (log of
#                                a non-positive inverse depth) and carries no information: its Jacobian row and residual are
#                                zeroed; the factor's other component is unchanged.  Applies to the interior normal factors
#                                and to the border sets.  Under PIXVOD_BINI the weights treat such a component as a missing
#                                side (slope 0, BINI's convention for a pixel without that neighbour), so the floored value no
#                                longer decides the other component's weight.  Default 'anchor' = the legacy handling.
_NFLOOR = os.environ.get('PIXVOD_NORMAL_FLOOR', 'anchor')
if _NFLOOR not in ('anchor', 'invalid'):
    raise SystemExit(f'PIXVOD_NORMAL_FLOOR must be anchor or invalid; got {_NFLOOR!r}')
_NFLOOR_INV = _NFLOOR == 'invalid'
if _NFLOOR_INV:
    print('[NORMAL-FLOOR] residual components touching a floored inverse depth carry no information', flush=True)
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from run_pixvo import RotationStandaloneRunner, read_yaml, h_J_fn_normal_ipd, h_J_fn_general, update_factor_flow_hJ, update_factor_normal, build_se3d_graph, h2_fn, h3_fn, save_plot_twist
import jax
import jax.numpy as jnp
import jax.scipy.linalg  # PIXVOD_DEV_REFRESH (solve_triangular)
from gbp_engine.gbp import update_variable_SE3D, update_factor_manifold_hJ, l2, huber
from functools import partial
from gbp_engine.gbp import update_variable_SE3D_ref, update_factor_manifold_hJ_ref, compute_ftov_msg_ref
from gbp_engine.gbp import _DON  # PIXVOD_DONATE
from run_pixvo import factor_update_normal
from run_pixvo import _depth_block_indices_se3d, Gaussian, Factor
from gbp_engine.gbp import compute_ftov_msg
from gbp_engine.lie import SE3_Log, se3_Exp, se3_geo_dist_explicit
from run_pixvo import bilinear_interpolate_jax_xy, photometric_array_xy, photometric_flow_inbounds, project_point, se3_move_point, unproject_point
from gbp_engine.weights import gauge_axis_factors, irls_energies, nu_effective, predictive_denominator, stage_index, tdist_weight_std
from run_pixvo import _OOB_MSG, _IPD as _OOB_IPD
# PIXVOD_OOB_MSG (mode zero only): rows the photometric update does not write store an exactly zero message instead of the held one.
if _OOB_MSG == 'rebase':
    raise SystemExit('PIXVOD_OOB_MSG=rebase is not ported to this source (zero only)')
if _OOB_MSG:
    print(f'[OOB-MSG] stored photometric message of rows not written: {_OOB_MSG}', flush=True)


@jax.jit
def _oob_held_rows(lin_rows, p, K, H, W, mask_ftov):
    """PIXVOD_OOB_MSG: rows the photometric update just did not write (the same gate as run_pixvo.py); unit checks only."""
    inb = jax.vmap(lambda xi, pi: photometric_flow_inbounds(xi, pi, K, H, W, _OOB_IPD))(lin_rows, p)
    return jnp.logical_not(jnp.logical_and(inb, mask_ftov.reshape(-1)))

_INV_POOL = None

def _batched_inv_diag6(mats):
    """inv(mats)[:, 6, 6] evaluated over contiguous chunks on a small thread pool.  numpy's
    batched inverse is an independent LAPACK call per matrix, so splitting the batch cannot
    change any value; it only overlaps the calls.  A LinAlgError in any chunk propagates, so
    the caller's pinv fallback still sees the whole batch."""
    global _INV_POOL
    n = int(mats.shape[0])
    if n < 2048:
        return np.linalg.inv(mats)[:, 6, 6]
    if _INV_POOL is None:
        _INV_POOL = ThreadPoolExecutor(max_workers=8)
    bounds = np.linspace(0, n, 9).astype(int)
    spans = [(int(a), int(b)) for (a, b) in zip(bounds[:-1], bounds[1:]) if b > a]
    parts = list(_INV_POOL.map(lambda ab: np.linalg.inv(mats[ab[0]:ab[1]])[:, 6, 6], spans))
    return np.concatenate(parts, axis=0)

def _inv_last_diag(Lam):
    """PIXVOD_DEV_REFRESH: inv(Lam)[..., -1, -1] for a batch of symmetric matrices, in working precision, as the inverse
    of the Schur complement s = Lam_nn - b^T A^-1 b of the leading block A (b = Lam[:-1, -1]); A^-1 b enters through
    the Cholesky factor of A, s = Lam_nn - |L_A^-1 b|^2.  Only that entry is formed.  Where it is not finite (A not
    positive definite) the entry comes from an LU solve of the whole matrix, as the host inverse would give it.  On the
    real graph (unit/u_dev_refresh.json) every belief was positive definite and this was the most accurate of the float32
    formulations tried (max relative error 1.2e-7 against the host float64 inverse)."""
    n = Lam.shape[-1]
    A = Lam[..., :n - 1, :n - 1]
    b = Lam[..., :n - 1, n - 1]
    LA = jnp.linalg.cholesky(A)
    y = jax.scipy.linalg.solve_triangular(LA, b[..., None], lower=True)[..., 0]
    v_schur = 1.0 / (Lam[..., n - 1, n - 1] - jnp.sum(y * y, axis=-1))
    e = jnp.zeros(Lam.shape[:-1], dtype=Lam.dtype).at[..., n - 1].set(1.0)
    v_lu = jnp.linalg.solve(Lam, e[..., None])[..., n - 1, 0]
    return jnp.where(jnp.isfinite(v_schur), v_schur, v_lu)


def _median_sorted(s, n):
    """np.median of the first n entries of the ascending array s (n >= 1): the middle entry, or the mean of the two."""
    hi = s[n // 2]
    lo = s[jnp.maximum(n // 2 - 1, 0)]
    return jnp.where(n % 2 == 1, hi, (lo + hi) / 2)


def _percentile_sorted(s, n, q):
    """np.percentile(..., q) (method 'linear') of the first n entries of the ascending array s (n >= 1)."""
    vi = (n - 1).astype(s.dtype) * (q / 100.0)
    lo = jnp.floor(vi)
    g = vi - lo
    i0 = jnp.clip(lo.astype(jnp.int32), 0, n - 1)
    i1 = jnp.clip(i0 + 1, 0, n - 1)
    (a, b) = (s[i0], s[i1])
    d = b - a
    return jnp.where(g >= 0.5, b - d * (1 - g), a + d * g)


@partial(jax.jit, static_argnames=['stats', 'H', 'W'])
def _dev_tdist_refresh(linpoints, p, K, img, z, Lam, corner, k2, si2, sw2, nu, stats, H, W):
    """PIXVOD_DEV_REFRESH: the Student-t refresh of the photometric weights (run_pixvod.py refresh block) as one device
    program, in working precision: residual, in-bounds and corner support, J_rho, V_rho = inv(Lam)[6, 6] (Cholesky
    Schur complement), D = sI^2 + J_rho^2 k^2 V_rho + sw^2, w = (nu+1)/(nu+r^2/D) on the support and 1 elsewhere.
    stats=True also returns the [TDIST-UNI] statistics over the support: z_med, z_p95, wbar, jvj_share_med, n.
    The small products (K X, T p, ...) are evaluated at full working precision: under the default TF32 matmul precision
    they are rounded to TF32 inputs (about 0.1 px in the projection, op by op as well as fused), and the weight
    (nu+1)/(nu+r^2/D) with D ~ sI^2 = 2.5e-5 is very sensitive to that error in the residual.
    """
    with jax.default_matmul_precision('highest'):
        return _dev_tdist_refresh_body(linpoints, p, K, img, z, Lam, corner, k2, si2, sw2, nu, stats, H, W)


def _dev_tdist_refresh_body(linpoints, p, K, img, z, Lam, corner, k2, si2, sw2, nu, stats, H, W):
    z0 = z[:, 0]
    nd = z0.shape[0]
    lp = linpoints[:nd]
    pp = p[:nd]

    def fwd(x, q):
        dz = 1.0 / jnp.maximum(x[6], 1e-06)
        X = se3_move_point(x[:6], unproject_point(dz, q, K))
        return (project_point(X, K), X[2])
    (uv, qz) = jax.vmap(fwd)(lp, pp)
    h = jax.vmap(lambda u: jnp.reshape(bilinear_interpolate_jax_xy(img[..., 0], photometric_array_xy(u)), ()))(uv)
    r = z0 - h
    ok = jnp.isfinite(uv).all(axis=1) & jnp.isfinite(qz) & (qz > 0.0) & (uv[:, 0] >= 0.0) & (uv[:, 0] < W) & (uv[:, 1] >= 0.0) & (uv[:, 1] < H) & jnp.isfinite(r)
    active = ok & corner.reshape(-1)[:nd]
    J = jax.vmap(lambda x, q: jnp.atleast_2d(h_J_fn_general(x, q, K, img)[1]))(lp, pp)[:, 0, 6]
    v = _inv_last_diag(Lam[:nd])
    v = jnp.where(jnp.isfinite(v) & (v > 0.0), v, 0.0) * k2
    j2 = J * J
    D = si2 + jnp.where(jnp.isfinite(j2) & jnp.isfinite(v) & (v > 0.0), j2 * v, 0.0) + sw2
    Dc = jnp.maximum(D, jnp.finfo(D.dtype).tiny)
    w = jnp.where(jnp.isfinite(r), (nu + 1.0) / (nu + r * r / Dc), 1.0)
    w = jnp.where(active, w, 1.0).astype(jnp.float32)
    if not stats:
        return (w.reshape(-1, 1), None)
    n = jnp.sum(active).astype(jnp.int32)
    big = jnp.asarray(jnp.inf, dtype=D.dtype)
    zs = jnp.sort(jnp.where(active, jnp.abs(r) / jnp.sqrt(Dc), big))
    ss = jnp.sort(jnp.where(active, j2 * v / Dc, big))
    nn = jnp.maximum(n, 1)
    wbar = jnp.sum(jnp.where(active, w, 0.0)) / nn
    st = jnp.stack([_median_sorted(zs, nn), _percentile_sorted(zs, nn, 95.0), wbar.astype(D.dtype), _median_sorted(ss, nn)])
    st = jnp.where(n > 0, st, jnp.nan)
    return (w.reshape(-1, 1), (st, n))


def _update_prior_facs(prior_facs, vars_, vtof_msgs, linpoints):
    return update_factor_manifold_hJ(prior_facs, vars_, vtof_msgs, linpoints, h3_fn, l2)

def _mu_exact(g):
    """Exact Gaussian mean Lam^-1 eta (PIXVOD_EXACT_READOUT); the legacy zero-Lam branch is kept."""
    is_zero = jnp.allclose(g.Lam, 0)

    def zero_branch(_):
        return jnp.asarray(g.eta, dtype=g.Lam.dtype)

    def solve_branch(_):
        x = jnp.linalg.solve(g.Lam, jnp.asarray(g.eta, dtype=g.Lam.dtype))
        if _READOUT_GUARD:
            return jnp.where(jnp.all(jnp.isfinite(x)), x, g.mu())
        return x
    return jax.lax.cond(is_zero, zero_branch, solve_branch, operand=None)


def _readout_guard_count(belief):
    """PIXVOD_READOUT_GUARD diagnostic: variables whose exact readout needed the fallback (non-zero Lam, solve not finite)."""
    if _DIAG_FETCH:
        return int(_readout_guard_count_jit(belief))
    nz = ~jax.vmap(lambda g: jnp.allclose(g.Lam, 0))(belief)
    x = jax.vmap(lambda g: jnp.linalg.solve(g.Lam, jnp.asarray(g.eta, dtype=g.Lam.dtype)))(belief)
    return int(jnp.sum(nz & ~jnp.all(jnp.isfinite(x), axis=-1)))


@jax.jit
def _readout_guard_count_jit(belief):
    """PIXVOD_DIAG_FETCH: the same count as one compiled program (an integer; the per-variable solve is the same LU)."""
    nz = ~jax.vmap(lambda g: jnp.allclose(g.Lam, 0))(belief)
    x = jax.vmap(lambda g: jnp.linalg.solve(g.Lam, jnp.asarray(g.eta, dtype=g.Lam.dtype)))(belief)
    return jnp.sum(nz & ~jnp.all(jnp.isfinite(x), axis=-1))

@jax.jit
def _bini_zlam(adj_var_id, linpoints, nz, f_rows, base, k):
    """Normal-factor precisions under PIXVOD_BINI from the linearisation point (rows: 2p horizontal, 2p+1 vertical)."""
    xs = linpoints[adj_var_id][..., 6]
    h = jax.vmap(h_J_fn_normal_ipd)(xs, jnp.stack([nz, nz], axis=1).reshape(-1, 1))[0]
    arg = k * f_rows ** 2 * (h[:, 1] ** 2 - h[:, 0] ** 2)
    w0 = jnp.where(jnp.isfinite(arg), jax.nn.sigmoid(arg), 0.5).astype(base.dtype)
    d = jnp.stack([2.0 * w0, 2.0 * (1.0 - w0)], axis=1)
    return base * (d[:, :, None] * jnp.eye(2, dtype=base.dtype)[None])

@jax.jit
def _bini_zlam_inv(adj_var_id, linpoints, nz_rows, f_rows, base, k):
    """_bini_zlam (per-row nz) under PIXVOD_NORMAL_FLOOR=invalid: a component touching a floored inverse depth counts as a
    missing side, slope 0, so the surviving component gets BINI's missing-side weight sigmoid(-k f^2 h^2)."""
    xs = linpoints[adj_var_id][..., 6]
    h = jax.vmap(h_J_fn_normal_ipd)(xs, nz_rows)[0]
    ok = xs > 1e-06
    hb = jnp.where(ok[:, 0] & ok[:, 1], h[:, 0], 0.0)
    hf = jnp.where(ok[:, 0] & ok[:, 2], h[:, 1], 0.0)
    arg = k * f_rows ** 2 * (hf ** 2 - hb ** 2)
    w0 = jnp.where(jnp.isfinite(arg), jax.nn.sigmoid(arg), 0.5).astype(base.dtype)
    d = jnp.stack([2.0 * w0, 2.0 * (1.0 - w0)], axis=1)
    return base * (d[:, :, None] * jnp.eye(2, dtype=base.dtype)[None])

@jax.jit
def _bini_zlam_rows(adj_var_id, linpoints, nz_rows, f_rows, base, k):
    """_bini_zlam for a two-sided factor set whose nz and f are given per row (PIXVOD_NORMAL_BORDER nb3)."""
    xs = linpoints[adj_var_id][..., 6]
    h = jax.vmap(h_J_fn_normal_ipd)(xs, nz_rows)[0]
    arg = k * f_rows ** 2 * (h[:, 1] ** 2 - h[:, 0] ** 2)
    w0 = jnp.where(jnp.isfinite(arg), jax.nn.sigmoid(arg), 0.5).astype(base.dtype)
    d = jnp.stack([2.0 * w0, 2.0 * (1.0 - w0)], axis=1)
    return base * (d[:, :, None] * jnp.eye(2, dtype=base.dtype)[None])

def h_J_fn_normal_one(x, nz, s, rho_floor=1e-06):
    """One-sided normal residual of a factor [p, q] (PIXVOD_NORMAL_BORDER nb1): one component of h_J_fn_normal_ipd,
    h = s nz (log rho_p - log rho_q), with the same floor clamp and Jacobian gating."""
    r0 = jnp.maximum(x[0], rho_floor)
    r1 = jnp.maximum(x[1], rho_floor)
    h = (s * nz[0] * (jnp.log(r0) - jnp.log(r1))).reshape(-1)
    g0 = jnp.where(x[0] > rho_floor, 1.0, 0.0)
    g1 = jnp.where(x[1] > rho_floor, 1.0, 0.0)
    J = jnp.stack([s * nz[0] / r0 * g0, -s * nz[0] / r1 * g1]).reshape(1, 2)
    return (h, J)

def factor_update_normal_one(factor, xs_depth, nz, s, w):
    """factor_update_normal for the two-variable one-sided factor: potential on the 14-dim [p, q] block."""
    (h, J) = h_J_fn_normal_one(xs_depth.reshape(-1), nz, s)
    r = factor.z - h.reshape(-1)
    sw = w(r.T @ factor.z_Lam @ r, factor.threshold)
    Lam_small = sw * J.T @ factor.z_Lam @ J
    eta_small = sw * J.T @ factor.z_Lam @ r
    idx = _depth_block_indices_se3d(2)
    eta_full = jnp.zeros((14,), dtype=eta_small.dtype).at[idx].set(eta_small)
    Lam_full = jnp.zeros((14, 14), dtype=Lam_small.dtype).at[idx[:, None], idx[None, :]].set(Lam_small)
    return Gaussian(eta_full, Lam_full)

def factor_update_normal_inv(factor, xs_depth, nz, h_J_fn, w, rho_floor=1e-06):
    """run_pixvo.factor_update_normal under PIXVOD_NORMAL_FLOOR=invalid: component k (0: centre & left/up, 1: centre &
    right/down) carries no information when one of its two variables is at or below the floor."""
    x = xs_depth.reshape(-1)
    (h, J) = h_J_fn(x, nz)
    if J.ndim == 1:
        J = J[jnp.newaxis, :]
    ok = x > rho_floor
    valid = jnp.stack([ok[0] & ok[1], ok[0] & ok[2]]).astype(J.dtype)
    r = (factor.z - h.reshape(-1)) * valid
    J = J * valid[:, None]
    s = w(r.T @ factor.z_Lam @ r, factor.threshold)
    Lam_small = s * J.T @ factor.z_Lam @ J
    eta_small = s * J.T @ factor.z_Lam @ r
    depth_idx = _depth_block_indices_se3d()
    eta_full = jnp.zeros((21,), dtype=eta_small.dtype).at[depth_idx].set(eta_small)
    Lam_full = jnp.zeros((21, 21), dtype=Lam_small.dtype).at[depth_idx[:, None], depth_idx[None, :]].set(Lam_small)
    return Gaussian(eta_full, Lam_full)

def factor_update_normal_one_inv(factor, xs_depth, nz, s, w, rho_floor=1e-06):
    """factor_update_normal_one under PIXVOD_NORMAL_FLOOR=invalid."""
    x = xs_depth.reshape(-1)
    (h, J) = h_J_fn_normal_one(x, nz, s)
    valid = ((x[0] > rho_floor) & (x[1] > rho_floor)).astype(J.dtype)
    r = (factor.z - h.reshape(-1)) * valid
    J = J * valid
    sw = w(r.T @ factor.z_Lam @ r, factor.threshold)
    Lam_small = sw * J.T @ factor.z_Lam @ J
    eta_small = sw * J.T @ factor.z_Lam @ r
    idx = _depth_block_indices_se3d(2)
    eta_full = jnp.zeros((14,), dtype=eta_small.dtype).at[idx].set(eta_small)
    Lam_full = jnp.zeros((14, 14), dtype=Lam_small.dtype).at[idx[:, None], idx[None, :]].set(Lam_small)
    return Gaussian(eta_full, Lam_full)

def _chol_fwd(A, B):
    """PIXVOD_SMALLSOLVE: Y = L^-1 B for A = L L^T (A: (n, n) symmetric, B: (n, m)), Cholesky unrolled for a small static n
    in elementwise operations only (no matmul: no TF32 rounding; under vmap it fuses into the caller).  ok is False if a
    pivot is not positive and finite (A not numerically positive definite)."""
    n = A.shape[-1]
    L = [[None] * n for _ in range(n)]
    ok = jnp.asarray(True)
    for j in range(n):
        dj = A[j, j]
        for m in range(j):
            dj = dj - L[j][m] * L[j][m]
        ok = ok & (dj > 0) & jnp.isfinite(dj)
        L[j][j] = jnp.sqrt(dj)
        for i in range(j + 1, n):
            v = A[i, j]
            for m in range(j):
                v = v - L[i][m] * L[j][m]
            L[i][j] = v / L[j][j]
    Y = []
    for i in range(n):
        y = B[i]
        for m in range(i):
            y = y - L[i][m] * Y[m]
        Y.append(y / L[i][i])
    return (jnp.stack(Y), ok)


def _depth_ftov_ref(potential, vtof, u2):
    """PIXVOD_SMALLSOLVE: compute_ftov_msg_ref (marginalize_ref per target variable) for a factor whose potential involves
    only the depth coordinate (index 6) of each of its N variables (normal, nb3, nb1 factors).  The marginal over the other
    variables then changes only the [6] / [6, 6] entries of the target's block: with B_k = P_kk + V_k (potential block plus
    variable-to-factor message, as compute_ftov_msg_ref forms them) and marginalize_ref's ridge R = lam_damp / u2,
      A_k = (B_k + R)[:6, :6],  b_k = B_k[:6, 6],  f_k = eta_P,k + eta_k,
      s_k = (B_k + R)[6, 6] - b_k^T A_k^-1 b_k,  h_k = f_k[6] - b_k^T A_k^-1 f_k[:6],
      S = diag(s) + P_dd (depth-depth potential entries among the other variables),  c = P[n_d, others_d],
      Lam_msg = aa_Lam with [6, 6] - c^T S^-1 c,   eta_msg = aa_eta with [6] - c^T S^-1 h,
    which is marginalize_ref's  aa - ab bb_reg^-1 (ba, bb_eta)  in exact arithmetic (Schur complement onto the depth
    entries; every other entry of ab bb_reg^-1 ba is zero).  A_k^-1 enters through an unrolled Cholesky and S^-1
    (N-1 <= 2) in closed form, all in elementwise working-precision arithmetic, whereas the original forms ab @ X at the
    default matmul precision (TF32 in the default).  Returns (messages, ok): ok is False if a factorisation was not
    positive definite or a message is not finite (the caller then uses the original computation)."""
    (N, d) = vtof.eta.shape
    Pl = potential.Lam
    Pe = potential.eta
    blk = lambda M, a, b: M[a * d:(a + 1) * d, b * d:(b + 1) * d]
    ok = jnp.asarray(True)
    Bk = [blk(Pl, k, k) + vtof.Lam[k] for k in range(N)]
    fk = [Pe[k * d:(k + 1) * d] + vtof.eta[k] for k in range(N)]
    pairs = [(n, k) for n in range(N) for k in range(N) if k != n]
    Rn = []
    for n in range(N):
        others = [k for k in range(N) if k != n]
        bb_diag = jnp.concatenate([jnp.diagonal(Bk[k]) for k in others])
        u2_bb = jnp.tile(u2, len(others))
        scale = jnp.maximum(jnp.mean(jnp.abs(bb_diag) * u2_bb), 1e-12)
        lam_damp = jnp.maximum(1e-08, 0.001 * scale)
        Rn.append(lam_damp / u2)
    # every (target, other) pair through one vectorised Cholesky (one copy of the unrolled code in the program)
    A = jnp.stack([Bk[k][:d - 1, :d - 1] + jnp.diag(Rn[n][:d - 1]) for (n, k) in pairs])
    rhs = jnp.stack([jnp.stack([Bk[k][:d - 1, d - 1], fk[k][:d - 1]], axis=1) for (n, k) in pairs])
    (Y, okp) = jax.vmap(_chol_fwd)(A, rhs)
    ok = ok & jnp.all(okp)
    s_p = jnp.stack([Bk[k][d - 1, d - 1] + Rn[n][d - 1] for (n, k) in pairs]) - jnp.sum(Y[:, :, 0] * Y[:, :, 0], axis=1)
    h_p = jnp.stack([fk[k][d - 1] for (n, k) in pairs]) - jnp.sum(Y[:, :, 0] * Y[:, :, 1], axis=1)
    (etas, Lams) = ([], [])
    for n in range(N):
        others = [k for k in range(N) if k != n]
        aa_Lam = (blk(Pl, n, n) + vtof.Lam[n]) - vtof.Lam[n]
        aa_eta = (Pe[n * d:(n + 1) * d] + vtof.eta[n]) - vtof.eta[n]
        s = [s_p[pairs.index((n, k))] for k in others]
        h = [h_p[pairs.index((n, k))] for k in others]
        c = [Pl[n * d + d - 1, k * d + d - 1] for k in others]
        if len(others) == 1:
            ok = ok & (s[0] > 0)
            q = c[0] * c[0] / s[0]
            t = c[0] * h[0] / s[0]
        else:
            p = Pl[others[0] * d + d - 1, others[1] * d + d - 1]
            det = s[0] * s[1] - p * p
            ok = ok & (s[0] > 0) & (det > 0)
            q = (c[0] * c[0] * s[1] - 2.0 * c[0] * c[1] * p + c[1] * c[1] * s[0]) / det
            t = (c[0] * (s[1] * h[0] - p * h[1]) + c[1] * (s[0] * h[1] - p * h[0])) / det
        Lams.append(aa_Lam.at[d - 1, d - 1].set(aa_Lam[d - 1, d - 1] - q))
        etas.append(aa_eta.at[d - 1].set(aa_eta[d - 1] - t))
    (E, L) = (jnp.stack(etas), jnp.stack(Lams))
    return (Gaussian(E, L), ok & jnp.all(jnp.isfinite(E)) & jnp.all(jnp.isfinite(L)))


def _write_ftov(facs, vars, vtof, ref, u2):
    if ref and _SMALLSOLVE:  # closed-form depth-only marginalisation; the original if any factorisation fails (scalar cond)
        (ftov_new, ok) = jax.vmap(_depth_ftov_ref, in_axes=(0, 0, None))(facs.potential, vtof, u2)
        ftov = jax.lax.cond(jnp.all(ok), lambda ops: ftov_new,
                            lambda ops: jax.vmap(compute_ftov_msg_ref, in_axes=(0, 0, None))(ops[0], ops[1], jnp.tile(u2, ops[1].eta.shape[-2])), (facs, vtof))
    elif ref:
        ftov = jax.vmap(compute_ftov_msg_ref, in_axes=(0, 0, None))(facs, vtof, jnp.tile(u2, vtof.eta.shape[-2]))
    else:
        ftov = jax.vmap(compute_ftov_msg)(facs, vtof)
    vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov.eta)
    vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return vars

@partial(jax.jit, static_argnames=['w', 'ref', 'inv'], donate_argnames=_DON)
def update_factor_normal_rows(facs, vars, vtof_msgs, linpoints, nz_rows, w, u2, ref, inv=False):
    """update_factor_normal(_ref) for a two-sided factor set with nz given per row (nb3)."""
    vtof = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    xs = linpoints[facs.adj_var_id][..., 6]
    if inv:
        facs.potential = jax.vmap(factor_update_normal_inv, in_axes=(0, 0, 0, None, None))(facs, xs, nz_rows, h_J_fn_normal_ipd, w)
    else:
        facs.potential = jax.vmap(factor_update_normal, in_axes=(0, 0, 0, None, None, None, None))(facs, xs, nz_rows, h_J_fn_normal_ipd, w, 1e-06, False)
    return (facs, _write_ftov(facs, vars, vtof, ref, u2))

@partial(jax.jit, static_argnames=['w', 'ref', 'inv'], donate_argnames=_DON)
def update_factor_normal_one(facs, vars, vtof_msgs, linpoints, nz_rows, s_rows, w, u2, ref, inv=False):
    """The one-sided factor set's update (nb1), same message computation as the interior normal factors."""
    vtof = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    xs = linpoints[facs.adj_var_id][..., 6]
    fn = factor_update_normal_one_inv if inv else factor_update_normal_one
    facs.potential = jax.vmap(fn, in_axes=(0, 0, 0, 0, None))(facs, xs, nz_rows, s_rows, w)
    return (facs, _write_ftov(facs, vars, vtof, ref, u2))

@partial(jax.jit, static_argnames=['w', 'ref'], donate_argnames=_DON)
def update_factor_normal_inv(facs, vars, vtof_msgs, linpoints, nz, w, u2, ref):
    """Interior normal factor update under PIXVOD_NORMAL_FLOOR=invalid (legacy and reference-unit variants)."""
    vtof = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    xs = linpoints[facs.adj_var_id][..., 6]
    facs.potential = jax.vmap(factor_update_normal_inv, in_axes=(0, 0, 0, None, None))(facs, xs, jnp.stack([nz, nz], axis=1).reshape(-1, 1), h_J_fn_normal_ipd, w)
    return (facs, _write_ftov(facs, vars, vtof, ref, u2))

@jax.jit
def _bini_zlam_one(adj_var_id, linpoints, nz_rows, s_rows, f_rows, base, k):
    """BINI weight of a row whose opposite side is missing (slope 0 there): precision base * 2 sigmoid(-k f^2 h^2)."""
    xs = linpoints[adj_var_id][..., 6]
    h = jax.vmap(h_J_fn_normal_one)(xs, nz_rows, s_rows)[0][:, 0]
    arg = -k * f_rows ** 2 * h ** 2
    w = jnp.where(jnp.isfinite(arg), jax.nn.sigmoid(arg), 0.5).astype(base.dtype)
    return base * (2.0 * w)[:, None, None]

@partial(jax.jit, static_argnames=['f', 'w', 'dynamic_rho_floor'])
def update_factor_normal_ref(facs, vars, vtof_msgs, linpoints, nz, f, w, u2, rho_floor=1e-06, dynamic_rho_floor=False):
    """run_pixvo.update_factor_normal with the marginalisation loading in reference units (PIXVOD_REF_GAUGE marg)."""
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    xs_depth = linpoints_reordered[..., 6]
    facs.potential = jax.vmap(factor_update_normal, in_axes=(0, 0, 0, None, None, None, None))(facs, xs_depth, jnp.stack([nz, nz], axis=1).reshape(-1, 1), f, w, rho_floor, dynamic_rho_floor)
    u2_full = jnp.tile(u2, vtof_msgs_reordered.eta.shape[-2])
    ftov_msgs = jax.vmap(compute_ftov_msg_ref, in_axes=(0, 0, None))(facs, vtof_msgs_reordered, u2_full)
    vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.eta)
    vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)

def _parse_scalar_value(value_str: str):
    s = value_str.strip()
    lower = s.lower()
    if lower in ('true', 'false', 'null', 'none'):
        return {'true': True, 'false': False, 'null': None, 'none': None}[lower]
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return value_str

def _set_dotted_key(cfg: dict, dotted_key: str, value) -> None:
    keys = [k for k in dotted_key.split('.') if len(k) > 0]
    if not keys:
        return
    cur = cfg
    for k in keys[:-1]:
        if isinstance(cur, list):
            try:
                idx = int(k)
            except ValueError:
                print(f"[warn] Ignoring override {dotted_key}: non-integer index '{k}' into a list")
                return
            if not 0 <= idx < len(cur):
                print(f'[warn] Ignoring override {dotted_key}: index {idx} out of range')
                return
            cur = cur[idx]
            continue
        if k not in cur or not isinstance(cur.get(k), (dict, list)):
            cur[k] = {}
        cur = cur[k]
    if isinstance(cur, list):
        try:
            idx = int(keys[-1])
        except ValueError:
            print(f"[warn] Ignoring override {dotted_key}: non-integer index '{keys[-1]}' into a list")
            return
        if not 0 <= idx < len(cur):
            print(f'[warn] Ignoring override {dotted_key}: index {idx} out of range')
            return
        cur[idx] = value
    else:
        cur[keys[-1]] = value

def _apply_cli_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return cfg
    for item in overrides:
        if item is None:
            continue
        if '=' not in item:
            print(f"[warn] Ignoring override without '=': {item}")
            continue
        (k, v) = item.split('=', 1)
        key = k.strip()
        val = _parse_scalar_value(v)
        _set_dotted_key(cfg, key, val)
    return cfg

class TrajectoryPairRunner(RotationStandaloneRunner):

    def __init__(self, blend_path, cfg, config_path=None):
        super().__init__(blend_path, cfg, config_path=config_path)
        self._covariant_execution_mode = 'parametric'
        self._gauge_total: float = 1.0
        self.D_op = 7
        pia = np.array(jax.device_get(self.prior_init_axis))
        if pia.ndim == 0:
            pia = np.zeros((self.D_op,), dtype=np.float32)
        if pia.shape[0] != self.D_op:
            if pia.shape[0] < self.D_op:
                pia = np.concatenate([pia.astype(np.float32), np.zeros((self.D_op - pia.shape[0],), dtype=np.float32)], axis=0)
            else:
                pia = pia[:self.D_op].astype(np.float32)
        self.prior_init_axis = jnp.array(pia, dtype=jnp.float32)
        self.sub_frames: int = int(cfg.get('sub_frames', 0))
        self.skip_frames: int = int(cfg.get('skip_frames', 0))
        if self.skip_frames < 0:
            self.skip_frames = 0
        self.zero_acceleration: bool = bool(cfg.get('zero_acceleration', False))
        self.accel_probe_B: bool = bool(cfg.get('accel_probe_B', False))
        self.photo_autodiff_jac: bool = bool(cfg.get('photo_autodiff_jac', False))
        self.random_theta_max_deg: float = float(cfg.get('random_theta_max_deg', 5.0))
        self.random_translation_magnitude_mean: float = float(cfg.get('random_translation_magnitude_mean', 0.15))
        self.random_translation_magnitude_std: float = float(cfg.get('random_translation_magnitude_std', 0.05))
        self.gt_init: bool = bool(cfg.get('gt_init', False))
        self.gt_with_noise_init: bool = bool(cfg.get('gt_with_noise_init', False))
        self.gt_noise_random_theta_max_deg: float = float(cfg.get('gt_noise_random_theta_max_deg', 0.0))
        self.gt_noise_random_translation_magnitude_mean: float = float(cfg.get('gt_noise_random_translation_magnitude_mean', 0.0))
        self.gt_noise_random_translation_magnitude_std: float = float(cfg.get('gt_noise_random_translation_magnitude_std', 0.0))
        self._gt_init_applied: bool = False
        self.ratio_s: float = float(cfg.get('ratio_s', 1.0))
        self.normal_s: float = float(cfg.get('normal_s', 0.001))
        self.use_robust_normal: bool = bool(cfg.get('use_robust_normal', False))
        self.normal_robust_thres: float = float(cfg.get('normal_robust_thres', 0.1))
        self.normal_noise_enable: bool = bool(cfg.get('normal_noise_enable', False))
        self.normal_noise_sigma_deg: float = float(cfg.get('normal_noise_sigma_deg', 5.0))
        dm = cfg.get('depth_model', 'normal_integration')
        self.depth_model: str = str(dm).strip().lower()
        self.use_gt_depth_prior: bool = bool(cfg.get('use_gt_depth_prior', False))
        pg = cfg.get('pose_graph_type', 'quadtree')
        self.pose_graph_type: str = str(pg).strip().lower()
        if self.pose_graph_type not in ('quadtree', 'planar'):
            raise ValueError(f"Unsupported pose_graph_type={self.pose_graph_type}. Use 'quadtree' or 'planar'.")
        self._depth_prior_weaken_factor: float = 1.0
        self.prior_s_rot: float = float(cfg.get('prior_s_rot', self.prior_s))
        self.prior_s_tr: float = float(cfg.get('prior_s_tr', self.prior_s))
        self.prior_s_depth: float = float(cfg.get('prior_s_depth', self.prior_s))
        lim = cfg.get('global_iter_limit', None)
        self.global_iter_limit: int | None = int(lim) if lim is not None else None
        if self.global_iter_limit is not None and self.global_iter_limit <= 0:
            self.global_iter_limit = None
        qr = cfg.get('quiet_render', None)
        if isinstance(qr, str):
            self.quiet_render = qr.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            self.quiet_render = bool(qr) if qr is not None else False
        vb = cfg.get('verbose', None)
        if isinstance(vb, str):
            self.verbose = vb.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            self.verbose = bool(vb) if vb is not None else False
        self._comb_iters: list[int] = []
        self._comb_norm_err: list[float] = []
        self._comb_norm_err_cur: list[float] = []
        self._comb_est: list[np.ndarray] = []
        self._comb_est_std: list[np.ndarray] = []
        self._comb_root_est: list[np.ndarray] = []
        self._comb_root_err_cur: list[float] = []
        self._comb_gt: list[np.ndarray] = []
        self._comb_gt_cur: list[np.ndarray] = []
        self._comb_is_sub: list[bool] = []
        self._comb_iter_counter: int = 0
        self._comb_initialized: bool = False
        self._overlay_paths = []
        self.depth_grid_facs = None

    def _compute_var_levels(self):
        """
        Compute pyramid level index for each variable id (0 for leaves, increasing toward root).
        Variables are laid out contiguously per level by build_se3d_graph.
        """
        N_total = int(self.N_total) if self.N_total is not None else int(self.prior_facs.z_Lam.shape[0])
        H = int(self.H)
        if str(getattr(self, 'pose_graph_type', 'quadtree')).strip().lower() != 'quadtree':
            return np.zeros((N_total,), dtype=np.int32)
        L = int(np.log2(max(1, H))) + 1
        sizes = np.array([H // 2 ** l for l in range(L)], dtype=np.int64)
        counts = sizes * sizes
        offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(counts[:-1])])
        var_level = np.empty((N_total,), dtype=np.int32)
        for l in range(L):
            start = int(offsets[l])
            end = int(start + counts[l])
            end = min(end, N_total)
            var_level[start:end] = l
        return var_level

    def _build_skip_schedule(self, num_subframes: int, stage_iters: int):
        """Return list of (relative_subframe_index_1based, iters_for_that_stage).
        If skip_frames == 0, schedule includes all subframes with stage_iters each.
        Otherwise, process every (skip_frames+1)-th subframe, allocating
        (skip_frames+1) * stage_iters iterations to each processed subframe.
        """
        r = int(getattr(self, 'skip_frames', 0) or 0)
        n = int(num_subframes)
        s = int(stage_iters)
        if n <= 0 or s <= 0:
            return []
        if r <= 0:
            return [(i, s) for i in range(1, n + 1)]
        g = r + 1
        out = []
        k = g
        while k <= n:
            out.append((k, g * s))
            k += g
        return out

    def _apply_hp_updates(self, updates: dict) -> None:
        """Extend schedule updates to support 'ratio_s'."""
        super()._apply_hp_updates(updates)
        if updates is None:
            return
        if 'ratio_s' in updates:
            self.ratio_s = float(updates['ratio_s'])
            print(f'[HP] Applied ratio_s={self.ratio_s} at iter {self.global_iter_count}')
        for k in ('normal_s', 'use_robust_normal', 'normal_robust_thres', 'prior_s_rot', 'prior_s_tr', 'prior_s_depth'):
            if k in updates:
                val = updates[k]
                if k == 'use_robust_normal':
                    setattr(self, k, bool(val))
                else:
                    setattr(self, k, float(val))

    def _check_global_iter_limit(self) -> bool:
        """If a global iteration limit is configured and reached, set killer.stop and report once."""
        limit = self.global_iter_limit
        cur = int(self.global_iter_count)
        if limit is not None and cur >= int(limit):
            if not getattr(self.killer, 'stop', False):
                print(f'[stop] Global iteration limit reached: {cur} >= {limit}. Stopping optimization.', flush=True)
            self.killer.stop = True
            return True
        return False

    def set_hyperparameters(self):
        """
        Override: apply per-level scaling to smoothness/prior sigmas using ratio_s.
        - Prior at variable level l: sigma_prior(l) = (interior/boundary leaf prior) * ratio_s**l
        - Between edge (child->parent) uses parent level for sigma_smoothness(l_parent)
        """
        _pose_schur_token = '0'
        try:
            _pose_schur_beta_cfg = float(_pose_schur_token)
        except (TypeError, ValueError, OverflowError) as _pose_schur_exc:
            raise ValueError('TDIST_POSE_SCHUR_BETA must be a finite number in [0,1]') from _pose_schur_exc
        if not np.isfinite(_pose_schur_beta_cfg) or _pose_schur_beta_cfg < 0.0 or _pose_schur_beta_cfg > 1.0:
            raise ValueError('TDIST_POSE_SCHUR_BETA must be a finite number in [0,1]')
        var_level = self._compute_var_levels()
        level_scale = (self.ratio_s ** var_level).astype(np.float32)
        (H, W) = (int(self.H), int(self.W))
        cut = int(self.cut)
        (yy, xx) = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
        interior_mask_leaves_np = (yy >= cut) & (yy < H - cut) & (xx >= cut) & (xx < W - cut)
        num_prior = int(self.prior_facs.z_Lam.shape[0])
        leaf_count = min(int(self.N), num_prior)
        sigma_prior_base = np.full((num_prior,), float(self.prior_s), dtype=np.float32)
        if _STAGE_CONST:  # the same booleans on the host, without the device round trip
            boundary_flags = (~interior_mask_leaves_np.reshape(-1))[:leaf_count]
        else:
            interior_mask_leaves = jnp.array(interior_mask_leaves_np.reshape(-1))
            boundary_flags = (~interior_mask_leaves)[:leaf_count]
        sigma_prior_base[:leaf_count] = np.where(np.array(boundary_flags), float(self.prior_boundary_s), float(self.prior_s)).astype(np.float32)
        sigma_rot = (float(self.prior_s_rot) * level_scale[:num_prior]).astype(np.float32)
        sigma_tr = (float(self.prior_s_tr) * level_scale[:num_prior]).astype(np.float32)
        sigma_dep = (float(self.prior_s_depth) * float(getattr(self, '_depth_prior_weaken_factor', 1.0)) * level_scale[:num_prior]).astype(np.float32)
        inv_rot = 1.0 / sigma_rot ** 2
        inv_tr = 1.0 / sigma_tr ** 2
        inv_dep = 1.0 / sigma_dep ** 2
        if 'prior_t' in _REF_GAUGE:
            inv_tr = (inv_tr.astype(np.float64) * self._rg_g0 ** 2).astype(np.float32)
        if 'prior_rho' in _REF_GAUGE:
            inv_dep = (inv_dep.astype(np.float64) / self._rg_g0 ** 2).astype(np.float32)
        z_lam = np.zeros((num_prior, self.D_op, self.D_op), dtype=np.float32)
        if int(self.D_op) >= 7:
            diag_cols = [inv_rot, inv_rot, inv_rot, inv_tr, inv_tr, inv_tr, inv_dep]
        else:
            diag_cols = [inv_rot, inv_rot, inv_rot, inv_tr, inv_tr, inv_tr][:int(self.D_op)]
        _d = np.arange(len(diag_cols))
        if _STAGE_CONST:  # the device array is built once per distinct diagonal (bit-identical bytes)
            _dg = np.stack(diag_cols, axis=1).astype(np.float32)

            def _build_prior_zlam():
                z_lam[:, _d, _d] = _dg
                return jnp.array(z_lam)
            self.prior_facs.z_Lam = self._stage_const(('prior_zlam', z_lam.shape, _dg.tobytes()), _build_prior_zlam)
        else:
            z_lam[:, _d, _d] = np.stack(diag_cols, axis=1).astype(np.float32)
            self.prior_facs.z_Lam = jnp.array(z_lam)
        if getattr(self, '_carry_zlam', None) is not None:
            self.prior_facs.z_Lam = self._carry_zlam
        if _STAGE_CONST:  # adj_var_id never changes: one device-to-host copy per run
            if getattr(self, '_sc_parent_ids', None) is None:
                self._sc_parent_ids = np.asarray(self.between_facs.adj_var_id)[:, 1].astype(np.int64)
            parent_ids = self._sc_parent_ids
        else:
            parent_ids = np.asarray(self.between_facs.adj_var_id)[:, 1].astype(np.int64)
        parent_levels = var_level[parent_ids]
        rot_sigma_eff = (float(self.smoothness_s_rot) * self.ratio_s ** parent_levels).astype(np.float32)
        tr_sigma_eff = (float(self.smoothness_s_tr) * self.ratio_s ** parent_levels).astype(np.float32)
        inv_var_rot = 1.0 / rot_sigma_eff ** 2
        inv_var_tr = 1.0 / tr_sigma_eff ** 2
        if 'between_t' in _REF_GAUGE:
            inv_var_tr = (inv_var_tr.astype(np.float64) * self._rg_g0 ** 2).astype(np.float32)
        M = parent_levels.shape[0]
        z_lam = np.zeros((M, 6, 6), dtype=np.float32)
        diag_vals6 = np.stack([inv_var_rot, inv_var_rot, inv_var_rot, inv_var_tr, inv_var_tr, inv_var_tr], axis=1)
        _d6 = np.arange(6)
        _D_data = self.D
        _w_sym = float('0.5') if _D_data == 2 else 1.0
        if _STAGE_CONST:  # the same expressions, evaluated once per distinct input
            def _build_between_zlam():
                z_lam[:, _d6, _d6] = diag_vals6
                return jnp.array(z_lam)
            _sc = self._stage_const
            self.between_facs.z = _sc(('between_z', M), lambda: jnp.zeros((M, 6), dtype=jnp.float32))
            self.between_facs.z_Lam = _sc(('between_zlam', z_lam.shape, diag_vals6.dtype.str, diag_vals6.tobytes()), _build_between_zlam)
            _thr_b = self.between_facs.threshold
            self.between_facs.threshold = _sc(('between_thr', _thr_b.shape, str(_thr_b.dtype), self.robust_thres), lambda: jnp.ones_like(_thr_b) * self.robust_thres)
            _n_dt = self.dataterm_facs.z_Lam.shape[0]
            self.dataterm_facs.z_Lam = _sc(('dataterm_zlam', _D_data, _w_sym, self.dataterm_s, _n_dt), lambda: jnp.tile(jnp.eye(_D_data) * (_w_sym / self.dataterm_s ** 2), (_n_dt, 1, 1)))
            _thr_d = self.dataterm_facs.threshold
            self.dataterm_facs.threshold = _sc(('dataterm_thr', _thr_d.shape, str(_thr_d.dtype), self.robust_thres), lambda: jnp.ones_like(_thr_d) * self.robust_thres)
        else:
            z_lam[:, _d6, _d6] = diag_vals6
            self.between_facs.z = jnp.zeros((M, 6), dtype=jnp.float32)
            self.between_facs.z_Lam = jnp.array(z_lam)
            self.between_facs.threshold = jnp.ones_like(self.between_facs.threshold) * self.robust_thres
            self.dataterm_facs.z_Lam = jnp.tile(jnp.eye(_D_data) * (_w_sym / self.dataterm_s ** 2), (self.dataterm_facs.z_Lam.shape[0], 1, 1))
            self.dataterm_facs.threshold = jnp.ones_like(self.dataterm_facs.threshold) * self.robust_thres
        _ue_reset_sym = False
        self._td_uni_stage = None
        self._td_uni_w = None
        self._td_uni_wdev = False
        self._td_uni_D = None
        self._td_uni_effective_precision = None
        self._td_uni_residual = None
        self._td_uni_active = None
        self._td_uni_usable = None
        self._td_uni_nu = None
        self._td_uni_generation = 0
        self._td_ue_stage = None
        self._td_ue_mult = None
        self._td_ue_w = None
        self._td_ue_D = None
        self._td_ue_effective_precision = None
        self._td_ue_residual = None
        self._td_ue_active = None
        self._td_ue_usable = None
        self._td_ue_proper = None
        self._td_ue_nu = None
        self._td_ue_center = None
        self._td_ue_reject_stage = None
        self._td_ue_attempt_stage = None
        self._td_ue_generation = 0
        self._td_chart = None
        _ns_mode = ''
        _ns_lam = None
        if _STAGE_CONST:
            _n_nf = self.normal_facs.z_Lam.shape[0]
            self.normal_facs.z_Lam = self._stage_const(('normal_zlam', self.normal_s, _n_nf), lambda: jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (_n_nf, 1, 1)))
            _thr_n = self.normal_facs.threshold
            self.normal_facs.threshold = self._stage_const(('normal_thr', _thr_n.shape, str(_thr_n.dtype), self.normal_robust_thres), lambda: jnp.ones_like(_thr_n) * self.normal_robust_thres)
        else:
            self.normal_facs.z_Lam = jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (self.normal_facs.z_Lam.shape[0], 1, 1))
            self.normal_facs.threshold = jnp.ones_like(self.normal_facs.threshold) * self.normal_robust_thres
        if getattr(self, 'nb3_facs', None) is not None or getattr(self, 'nb1_facs', None) is not None:
            self._nb_set_hyper()
        if self.global_iter_count % max(1, int(self.iteration_log_interval)) == 0:
            L_max = int(var_level.max()) if var_level.size > 0 else 0
            (lvl0, lvlT) = (0, L_max)
            s_lvl0 = float(self.ratio_s ** lvl0)
            s_lvlT = float(self.ratio_s ** lvlT)
            print(f'[HP] ratio_s={self.ratio_s} | level_scale: l0={s_lvl0:.3e}, l{lvlT}={s_lvlT:.3e}')
            Lam_prior_0 = float(1.0 / (sigma_prior_base[0] * self.ratio_s ** var_level[0]) ** 2)
            Lam_prior_T = float(1.0 / (sigma_prior_base[min(num_prior - 1, 0)] * self.ratio_s ** var_level[min(num_prior - 1, 0)]) ** 2)
            print(f'[HP] prior Lam sample (lvl0,lvlT) ~ ({Lam_prior_0:.3e}, {Lam_prior_T:.3e})')
            if M > 0:
                print(f'[HP] smoothness invvars (rot,tr) sample -> ({inv_var_rot[0]:.3e}, {inv_var_tr[0]:.3e}) at parent level {int(parent_levels[0])}')

    def _render_gt_flow_overlay(self, K_prev: np.ndarray, depth_prev: np.ndarray, T_rel: np.ndarray, rgb_prev: np.ndarray, out_path: Path) -> None:
        """
        Draw only the GT SE(3) projection (red arrows) on the previous image plane.
        Matches the style of _render_twist_overlay but without the estimated flow.
        """
        (H, W) = (int(self.H), int(self.W))
        base = (np.clip(rgb_prev[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        vis = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        K = np.asarray(K_prev, dtype=np.float64)
        K_inv = np.linalg.inv(K)
        T_gt = np.asarray(T_rel, dtype=np.float64)
        s = max(1, int(self.viz_stride))
        mu = min(8, W - 1)
        mv = min(8, H - 1)
        color_gt = (0, 0, 255)
        thickness = 1
        tip_length = 0.3
        depth = np.asarray(depth_prev).reshape(H, W)
        for v in range(mv, H, s):
            for u in range(mu, W, s):
                d = float(depth[v, u])
                if not np.isfinite(d) or d <= 0.0:
                    continue
                u0 = float(u) + 0.5
                v0 = float(v) + 0.5
                p_cam = K_inv @ np.array([u0, v0, 1.0], dtype=np.float64)
                X_prev = np.array([p_cam[0] * d, p_cam[1] * d, d, 1.0], dtype=np.float64)
                X_gt = T_gt @ X_prev
                if abs(X_gt[3]) < 1e-12 or abs(X_gt[2]) < 1e-12:
                    continue
                q_gt = K @ (X_gt[:3] / X_gt[3])
                (ug, vg) = (q_gt[0] / q_gt[2], q_gt[1] / q_gt[2])
                (x0, y0) = (int(round(u0)), int(round(v0)))
                (xg, yg) = (int(round(np.clip(ug, 0, W - 1))), int(round(np.clip(vg, 0, H - 1))))
                cv2.arrowedLine(vis, (x0, y0), (xg, yg), color_gt, thickness=thickness, tipLength=tip_length)
        cv2.imwrite(str(out_path), vis)

    def _ensure_graph(self):
        if self.vars is None:
            (self.vars, self.dataterm_facs, self.between_facs, self.prior_facs, self.normal_facs) = build_se3d_graph(H=self.H, W=self.W, D_feat=self.D, N_xyz=self.prev_normal, D_op=self.D_op, z_prior=1.0, intrinsic=self.prev_intrinsic, graph_type=self.pose_graph_type)
            self.N_total = int(self.vars.belief.eta.shape[0])
            if _INIT_K is not None:
                self.vars.belief.eta = self.vars.belief.eta * jnp.asarray(gauge_axis_factors(_INIT_K, self.D_op))[None, :]
            self._set_prior_measurements(self.prior_init_noise_std)
            self.prior_facs.threshold = jnp.ones_like(self.prior_facs.threshold) * self.robust_thres
            if _NB:
                self._build_normal_border()
            self._msg_slots = self._build_msg_slots()
            if _REF_GAUGE:
                _z = np.asarray(jax.device_get(self.prior_facs.z), dtype=np.float64)
                g0 = float(np.median(_z[:int(self.N), 6]))
                if g0 != (1.0 if _INIT_K is None else _INIT_K):
                    raise SystemExit(f'[REF-GAUGE] build-time anchor level g0={g0!r} is not the initial scale')
                self._rg_g0 = g0
                self._rg_u2 = jnp.asarray(np.array([1.0, 1.0, 1.0, g0 ** -2, g0 ** -2, g0 ** -2, g0 ** 2], dtype=np.float64))
                print(f'[REF-GAUGE] g0={g0!r} tokens={sorted(_REF_GAUGE)}', flush=True)
        elif getattr(self, '_msg_slots', None) is None:
            self._msg_slots = self._build_msg_slots()

    def _build_msg_slots(self):
        """Flat indices of the message slots that some factor writes, with the owning variable
        of each.  Messages start at zero and are only ever written at (adj_var_id, adj_var_idx),
        so every other slot stays exactly zero for the whole run; update_variable_SE3D uses this
        to rebase only the occupied slots."""
        (n_var, n_port) = (int(self.vars.msgs.eta.shape[0]), int(self.vars.msgs.eta.shape[1]))
        occupied = np.zeros((n_var, n_port), dtype=bool)
        for fac in (self.dataterm_facs, self.between_facs, self.prior_facs, self.normal_facs, getattr(self, 'nb3_facs', None), getattr(self, 'nb1_facs', None)):
            if fac is None:
                continue
            vid = np.asarray(jax.device_get(fac.adj_var_id)).reshape(-1)
            pid = np.asarray(jax.device_get(fac.adj_var_idx)).reshape(-1)
            occupied[vid, pid] = True
        flat = np.nonzero(occupied.reshape(-1))[0].astype(np.int32)
        if _COND_SPLIT and 'safe' in _REF_GAUGE:
            # PIXVOD_COND_SPLIT (update_variable_SE3D_ref only): the slots of the factor classes whose messages are zero or
            # diagonal by construction (prior: identity Jacobian; normal and border normal: depth-only), and the others
            diag_cls = np.zeros((n_var, n_port), dtype=bool)
            for fac in (self.prior_facs, self.normal_facs, getattr(self, 'nb3_facs', None), getattr(self, 'nb1_facs', None)):
                if fac is None:
                    continue
                diag_cls[np.asarray(jax.device_get(fac.adj_var_id)).reshape(-1), np.asarray(jax.device_get(fac.adj_var_idx)).reshape(-1)] = True
            dg = np.nonzero((occupied & diag_cls).reshape(-1))[0].astype(np.int32)
            dn = np.nonzero((occupied & ~diag_cls).reshape(-1))[0].astype(np.int32)
            return (jnp.asarray(flat), jnp.asarray((flat // n_port).astype(np.int32)), jnp.asarray(dg), jnp.asarray((dg // n_port).astype(np.int32)),
                    jnp.asarray(dn), jnp.asarray((dn // n_port).astype(np.int32)))
        return (jnp.asarray(flat), jnp.asarray((flat // n_port).astype(np.int32)))

    def _bini_weights(self, linpoints):
        R = int(self.normal_facs.z_Lam.shape[0])
        K = np.asarray(self.prev_intrinsic, dtype=np.float64)
        if _STAGE_CONST and _NFLOOR_INV:  # the same arguments, built once (none of them is donated)
            _sc = self._stage_const
            _dt = linpoints.dtype
            base = _sc(('bini_base', R, self.normal_s), lambda: jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (R, 1, 1)), donated=False)
            f_rows = _sc(('bini_f', R, K.tobytes(), str(_dt)), lambda: jnp.asarray(np.tile([K[0, 0], K[1, 1]], R // 2), dtype=_dt), donated=False)
            nzp = self._cached_nz_param(_dt)
            nz_rows = _sc(('bini_nz', str(_dt), np.asarray(self._nz_proj_inner).tobytes()), lambda: jnp.stack([nzp, nzp], axis=1).reshape(-1, 1), donated=False)
            kk = _sc(('bini_k', _BINI_K, str(_dt)), lambda: jnp.asarray(_BINI_K, dtype=_dt), donated=False)
            return _bini_zlam_inv(self.normal_facs.adj_var_id, linpoints, nz_rows, f_rows, base, kk)
        base = jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (R, 1, 1))
        f_rows = jnp.asarray(np.tile([K[0, 0], K[1, 1]], R // 2), dtype=linpoints.dtype)
        if _NFLOOR_INV:
            nzp = self._cached_nz_param(linpoints.dtype)
            return _bini_zlam_inv(self.normal_facs.adj_var_id, linpoints, jnp.stack([nzp, nzp], axis=1).reshape(-1, 1), f_rows, base, jnp.asarray(_BINI_K, dtype=linpoints.dtype))
        return _bini_zlam(self.normal_facs.adj_var_id, linpoints, self._cached_nz_param(linpoints.dtype), f_rows, base, jnp.asarray(_BINI_K, dtype=linpoints.dtype))

    def _build_normal_border(self):
        """PIXVOD_NORMAL_BORDER: add the nb3 / nb1 factor sets and register their (variable, port) slots."""
        (H, W) = (int(self.H), int(self.W))
        K = np.asarray(self.prev_intrinsic, dtype=np.float32)
        (yy, xx) = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
        n = np.asarray(self.prev_normal, dtype=np.float32)
        nzf = (n[..., 0] * (xx - K[0, 2]) / K[0, 0] + n[..., 1] * (yy - K[1, 2]) / K[1, 1] + n[..., 2]).astype(np.float32)
        zx, zy = (n[..., 0] / K[0, 0]).astype(np.float32), (n[..., 1] / K[1, 1]).astype(np.float32)
        P = lambda i, j: i * W + j
        r3 = [(P(i, j), P(i - 1, j), P(i + 1, j), (10, 11, 12), 1) for i in range(1, H - 1) for j in (0, W - 1)]
        r3 += [(P(i, j), P(i, j - 1), P(i, j + 1), (7, 8, 9), 0) for j in range(1, W - 1) for i in (0, H - 1)]
        r1 = []

        def one(i, j, di, dj):
            q = P(i + di, j + dj)
            r1.append({(0, -1): (P(i, j), q, (7, 8), 1.0, 0), (0, 1): (P(i, j), q, (7, 9), -1.0, 0),
                       (-1, 0): (P(i, j), q, (10, 11), 1.0, 1), (1, 0): (P(i, j), q, (10, 12), -1.0, 1)}[(di, dj)])
        for i in range(1, H - 1):
            one(i, 0, 0, 1); one(i, W - 1, 0, -1)
        for j in range(1, W - 1):
            one(0, j, 1, 0); one(H - 1, j, -1, 0)
        for (i, j) in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
            one(i, j, 0, 1 if j == 0 else -1); one(i, j, 1 if i == 0 else -1, 0)
        A = np.array(jax.device_get(self.vars.adj_factor_idx))
        fid = int(max(np.max(np.asarray(jax.device_get(f.factor_id))) for f in (self.dataterm_facs, self.between_facs, self.prior_facs, self.normal_facs))) + 1
        sets = []
        for rows, nv, tag in ((r3, 3, 'nb3'), (r1, 2, 'nb1')):
            if tag not in _NB_SETS:
                sets.append((None, None, None))
                continue
            ids = np.arange(fid, fid + len(rows), dtype=np.int32); fid += len(rows)
            adj = np.array([r[:nv] for r in rows], dtype=np.int32)
            ports = np.array([r[nv] for r in rows], dtype=np.int32)
            if (A[adj, ports] != -1).any():
                raise RuntimeError('PIXVOD_NORMAL_BORDER: a border factor port is already taken')
            A[adj, ports] = ids[:, None]
            c = adj[:, 0]
            orient = np.array([r[-1] for r in rows])
            zc = np.where(orient == 0, zx.ravel()[c], zy.ravel()[c]).astype(np.float32)
            f_rows = np.where(orient == 0, K[0, 0], K[1, 1]).astype(np.float64)
            z = np.stack([zc, zc], 1) if nv == 3 else zc[:, None]
            zl = np.tile(np.eye(z.shape[1], dtype=np.float32), (len(rows), 1, 1))
            sets.append((Factor(ids, z, zl, np.ones((len(rows), 1), dtype=np.float32), None, adj, ports), nzf.ravel()[c].reshape(-1, 1), f_rows))
        for (fac, _, _) in sets:
            if fac is None:
                continue
            cnt = np.bincount(A[A >= int(np.min(fac.factor_id))].ravel() - int(np.min(fac.factor_id)))[:fac.factor_id.size]
            if not (cnt == fac.adj_var_id.shape[1]).all():
                raise RuntimeError('PIXVOD_NORMAL_BORDER: a border factor is not registered once per adjacent variable')
        self.vars.adj_factor_idx = A if isinstance(self.vars.adj_factor_idx, np.ndarray) else jnp.asarray(A)
        ((self.nb3_facs, self._nb3_nz, self._nb3_f), (self.nb1_facs, self._nb1_nz, self._nb1_f)) = sets
        self._nb1_s = np.array([r[3] for r in r1], dtype=np.float32)
        self._nb_cache = None
        print(f'[NORMAL-BORDER] nb3 {len(r3) if self.nb3_facs is not None else 0} two-sided, nb1 {len(r1) if self.nb1_facs is not None else 0} one-sided factors', flush=True)

    def _nb_dev(self, dtype):
        if self._nb_cache is None or self._nb_cache[0] != dtype:
            a = lambda v: None if v is None else jnp.asarray(v, dtype)
            self._nb_cache = (dtype, a(self._nb3_nz), a(self._nb1_nz), a(self._nb1_s), a(self._nb3_f), a(self._nb1_f), jnp.ones((7,), dtype=dtype))
        return self._nb_cache[1:]

    def _update_normal_border(self, vtof_msgs, linpoints):
        (nz3, nz1, s1, _, _, ones7) = self._nb_dev(linpoints.dtype)
        w = huber if self.use_robust_normal else l2
        ref = 'marg' in _REF_GAUGE
        u2 = self._rg_u2 if ref else ones7
        if self.nb3_facs is not None:
            (self.nb3_facs, self.vars) = update_factor_normal_rows(self.nb3_facs, self.vars, vtof_msgs, linpoints, nz3, w, u2, ref, _NFLOOR_INV)
        if self.nb1_facs is not None:
            (self.nb1_facs, self.vars) = update_factor_normal_one(self.nb1_facs, self.vars, vtof_msgs, linpoints, nz1, s1, w, u2, ref, _NFLOOR_INV)

    def _nb_set_hyper(self):
        if _STAGE_CONST:
            _sc = self._stage_const
            for (tag, fac, d) in (('nb3', self.nb3_facs, 2), ('nb1', self.nb1_facs, 1)):
                if fac is None:
                    continue
                n = int(fac.z_Lam.shape[0])
                fac.z_Lam = _sc((tag + '_zlam', n, self.normal_s), lambda n=n, d=d: jnp.tile(jnp.eye(d, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (n, 1, 1)))
                thr = fac.threshold
                fac.threshold = _sc((tag + '_thr', thr.shape, str(thr.dtype), self.normal_robust_thres), lambda thr=thr: jnp.ones_like(thr) * self.normal_robust_thres)
            return
        if self.nb3_facs is not None:
            self.nb3_facs.z_Lam = jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (int(self.nb3_facs.z_Lam.shape[0]), 1, 1))
            self.nb3_facs.threshold = jnp.ones_like(self.nb3_facs.threshold) * self.normal_robust_thres
        if self.nb1_facs is not None:
            self.nb1_facs.z_Lam = jnp.tile(jnp.eye(1, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (int(self.nb1_facs.z_Lam.shape[0]), 1, 1))
            self.nb1_facs.threshold = jnp.ones_like(self.nb1_facs.threshold) * self.normal_robust_thres

    def _nb_bini(self, linpoints):
        (nz3, nz1, s1, f3, f1, _) = self._nb_dev(linpoints.dtype)
        if _STAGE_CONST:  # the same constant arguments, built once (none of them is donated)
            _sc = self._stage_const
            _dt = linpoints.dtype
            k = _sc(('bini_k', _BINI_K, str(_dt)), lambda: jnp.asarray(_BINI_K, dtype=_dt), donated=False)
            if self.nb3_facs is not None:
                n3 = int(self.nb3_facs.z_Lam.shape[0])
                b3 = _sc(('nb3_bini_base', n3, self.normal_s), lambda: jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (n3, 1, 1)), donated=False)
                self.nb3_facs.z_Lam = (_bini_zlam_inv if _NFLOOR_INV else _bini_zlam_rows)(self.nb3_facs.adj_var_id, linpoints, nz3, f3, b3, k)
            if self.nb1_facs is not None:
                n1 = int(self.nb1_facs.z_Lam.shape[0])
                b1 = _sc(('nb1_bini_base', n1, self.normal_s), lambda: jnp.tile(jnp.eye(1, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (n1, 1, 1)), donated=False)
                self.nb1_facs.z_Lam = _bini_zlam_one(self.nb1_facs.adj_var_id, linpoints, nz1, s1, f1, b1, k)
            return
        k = jnp.asarray(_BINI_K, dtype=linpoints.dtype)
        if self.nb3_facs is not None:
            b3 = jnp.tile(jnp.eye(2, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (int(self.nb3_facs.z_Lam.shape[0]), 1, 1))
            self.nb3_facs.z_Lam = (_bini_zlam_inv if _NFLOOR_INV else _bini_zlam_rows)(self.nb3_facs.adj_var_id, linpoints, nz3, f3, b3, k)
        if self.nb1_facs is not None:
            b1 = jnp.tile(jnp.eye(1, dtype=jnp.float32) * (1.0 / self.normal_s ** 2), (int(self.nb1_facs.z_Lam.shape[0]), 1, 1))
            self.nb1_facs.z_Lam = _bini_zlam_one(self.nb1_facs.adj_var_id, linpoints, nz1, s1, f1, b1, k)

    def _stage_const(self, key, build, donated=True):
        """PIXVOD_STAGE_CONST: a device constant built once per key (the key holds every input of build).  A factor-set
        leaf is donated under PIXVOD_DONATE, so such a leaf gets a device-side copy of the cached array."""
        cache = self.__dict__.setdefault('_sc_cache', {})
        if key not in cache:
            cache[key] = build()
        v = cache[key]
        return jnp.copy(v) if donated and _DONATE else v

    def _cached_nz_param(self, dtype):
        """Device copy of self._nz_proj_inner, uploaded once per stage instead of once per
        iteration.  Invalidated wherever _nz_proj_inner is rebuilt."""
        cache = getattr(self, '_nz_param_cache', None)
        if cache is None or cache[0] != dtype:
            cache = (dtype, jnp.array(self._nz_proj_inner, dtype=dtype))
            self._nz_param_cache = cache
        return cache[1]

    def _cached_photo_grid(self):
        """Photometric sample grid and its corner mask.  Both depend only on self.grids /
        self.active_ids / cut / H / W, which are fixed once the grid is built, so they are
        computed once instead of on every iteration (same values, same dtypes)."""
        cache = getattr(self, '_photo_grid_cache', None)
        if cache is None or cache[0] is not self.grids:
            p_use = self.grids if int(self.cut) == 0 else self.grids[self.active_ids]
            g_np = np.asarray(jax.device_get(p_use))
            xs = g_np[:, 0]
            ys = g_np[:, 1]
            eps = 1e-06
            Wm = float(self.W) - 0.5
            Hm = float(self.H) - 0.5
            is_tl = (np.abs(xs - 0.5) < eps) & (np.abs(ys - 0.5) < eps)
            is_tr = (np.abs(xs - Wm) < eps) & (np.abs(ys - 0.5) < eps)
            is_bl = (np.abs(xs - 0.5) < eps) & (np.abs(ys - Hm) < eps)
            is_br = (np.abs(xs - Wm) < eps) & (np.abs(ys - Hm) < eps)
            mask_corners = jnp.array(~(is_tl | is_tr | is_bl | is_br), dtype=bool)
            cache = (self.grids, p_use, mask_corners)
            self._photo_grid_cache = cache
        return (cache[1], cache[2])

    def _set_prior_measurements(self, noise_std: float) -> None:
        super()._set_prior_measurements(noise_std)
        if _INIT_K is not None and self.prior_facs is not None:
            self.prior_facs.z = self.prior_facs.z * jnp.asarray(gauge_axis_factors(_INIT_K, self.D_op), dtype=self.prior_facs.z.dtype)

    def _save_normal_visual(self, normal: np.ndarray, out_path: Path) -> None:
        """Save normal map visualization ([-1,1] -> [0,255]) to out_path (PNG)."""
        n = np.asarray(normal, dtype=np.float32)
        if n.ndim != 3 or n.shape[-1] != 3:
            return
        vis = 1.0 - (1.0 - n) / 2.0
        vis = np.clip(vis, 0.0, 1.0)
        vis_u8 = (vis * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)

    def _maybe_apply_normal_noise_and_visualize(self, stage_tag: str='prev') -> None:
        """
        If enabled, apply directional noise to self.prev_normal and save both GT and noisy
        normal visualizations to logs directory before any subframe optimization begins.
        """
        if self.prev_normal is None:
            return
        gt_path = self.logs_dir / f'normal_{stage_tag}_gt.png'
        self._save_normal_visual(self.prev_normal, gt_path)
        return

    def _append_combined_point_at(self, x_value: int, norm_err_main: float, norm_err_cur: float, est_twist6, est_std6, root_est6, root_err_cur: float, gt_twist_main6, gt_twist_cur6, is_subframe: bool) -> None:
        """Append a single log point at an absolute x (iteration) value."""
        self._comb_iters.append(int(x_value))
        self._comb_norm_err.append(float(norm_err_main))
        self._comb_norm_err_cur.append(float(norm_err_cur))
        self._comb_est.append(np.asarray(jax.device_get(est_twist6), dtype=np.float32).reshape(-1))
        self._comb_est_std.append(np.asarray(jax.device_get(est_std6), dtype=np.float32).reshape(-1))
        self._comb_root_est.append(np.asarray(jax.device_get(root_est6), dtype=np.float32).reshape(-1))
        self._comb_root_err_cur.append(float(root_err_cur))
        self._comb_gt.append(np.asarray(jax.device_get(gt_twist_main6), dtype=np.float32).reshape(-1))
        self._comb_gt_cur.append(np.asarray(jax.device_get(gt_twist_cur6), dtype=np.float32).reshape(-1))
        self._comb_is_sub.append(bool(is_subframe))
        self._comb_initialized = True

    def _save_combined_plots(self) -> None:
        """Render combined error and pose-per-axis plots with dashed (subframe) and solid (main) segments (the body is
        trend_plots.save_combined_plots, unchanged).  PIXVOD_ASYNC_PLOT: write the CSV here and let the worker draw."""
        if not _ASYNC_PLOT:
            return _trend_plots.save_combined_plots(self)
        if len(self._comb_iters) == 0:
            return
        if getattr(self, '_trend_async', None) is None:
            self._trend_async = _trend_plots.AsyncTrendPlots()
        self._save_combined_csv()
        self._trend_async.submit(_trend_plots.snapshot(self))

    def _close_trend_async(self) -> None:
        """PIXVOD_ASYNC_PLOT: wait until the worker has drawn the last snapshot (every figure exists afterwards)."""
        ta = getattr(self, '_trend_async', None)
        if ta is None:
            return
        self._trend_async = None
        if sys.exc_info()[0] is None:
            ta.close()
            return
        try:  # an exception is already propagating: report a worker failure without replacing it
            ta.close()
        except Exception as e:
            print(f'{e}', flush=True)

    def _save_combined_csv(self) -> None:
        """Save a single CSV aggregating subframe and mainframe logs with a discriminator column."""
        if len(self._comb_iters) == 0:
            return
        iters = np.asarray(self._comb_iters, dtype=np.int64).reshape(-1, 1)
        norm_err_main = np.asarray(self._comb_norm_err, dtype=np.float32).reshape(-1, 1)
        norm_err_cur = np.asarray(self._comb_norm_err_cur, dtype=np.float32).reshape(-1, 1)
        est_full = np.asarray(self._comb_est, dtype=np.float32)
        if est_full.ndim == 1:
            est_full = est_full.reshape(-1, est_full.size)
        if est_full.shape[1] < 6:
            pad = np.zeros((est_full.shape[0], 6 - est_full.shape[1]), dtype=np.float32)
            est6 = np.concatenate([est_full, pad], axis=1)[:, :6]
        else:
            est6 = est_full[:, :6]
        gt_full = np.asarray(self._comb_gt, dtype=np.float32)
        if gt_full.ndim == 1:
            gt_full = gt_full.reshape(-1, gt_full.size)
        if gt_full.shape[1] < 6:
            pad = np.zeros((gt_full.shape[0], 6 - gt_full.shape[1]), dtype=np.float32)
            gt6 = np.concatenate([gt_full, pad], axis=1)[:, :6]
        else:
            gt6 = gt_full[:, :6]
        gt_cur_full = np.asarray(self._comb_gt_cur, dtype=np.float32)
        if gt_cur_full.ndim == 1:
            gt_cur_full = gt_cur_full.reshape(-1, gt_cur_full.size)
        if gt_cur_full.shape[1] < 6:
            padc = np.zeros((gt_cur_full.shape[0], 6 - gt_cur_full.shape[1]), dtype=np.float32)
            gt_cur6 = np.concatenate([gt_cur_full, padc], axis=1)[:, :6]
        else:
            gt_cur6 = gt_cur_full[:, :6]
        is_sub = np.asarray(self._comb_is_sub, dtype=np.int32).reshape(-1, 1)
        table = np.column_stack([iters, norm_err_cur, norm_err_main, est6, gt_cur6, gt6, is_sub])
        header = 'iteration,norm_geo_err_current,norm_geo_err_main,est_wx,est_wy,est_wz,est_tx,est_ty,est_tz,gtcur_wx,gtcur_wy,gtcur_wz,gtcur_tx,gtcur_ty,gtcur_tz,gtmain_wx,gtmain_wy,gtmain_wz,gtmain_tx,gtmain_ty,gtmain_tz,is_subframe(1=sub,0=main)'
        out_csv = self.logs_dir / 'iter_trend_combined.csv'
        np.savetxt(out_csv, table, delimiter=',', header=header, comments='')

    def _export_overlay_video(self, fps: float=12.0, out_filename: str='overlays_all_12fps.mp4') -> None:
        """Export all recorded 5-panel overlay images as a single video at given FPS."""
        if not hasattr(self, '_overlay_paths') or len(self._overlay_paths) == 0:
            return
        first_frame = None
        first_size = None
        frames = []
        for p in self._overlay_paths:
            try:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                continue
            if first_frame is None:
                first_frame = img
                first_size = (img.shape[1], img.shape[0])
                frames.append(img)
            else:
                if img.shape[0] != first_size[1] or img.shape[1] != first_size[0]:
                    img = cv2.resize(img, first_size)
                frames.append(img)
        if len(frames) == 0:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = self.logs_dir / out_filename
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (first_size[0], first_size[1]))
        if not writer.isOpened():
            fourcc_avi = cv2.VideoWriter_fourcc(*'MJPG')
            out_path = self.logs_dir / out_filename.replace('.mp4', '.avi')
            writer = cv2.VideoWriter(str(out_path), fourcc_avi, float(fps), (first_size[0], first_size[1]))
        for f in frames:
            writer.write(f)
        writer.release()

    def _run_pre_warmup(self) -> None:
        """Run photo_warmup iterations using only smoothness, prior, and normal integration before
        rendering second/sub frames. Excluded from iterations_per_pair and does not bump global_iter_count.
        """
        if 'normal_integration'.strip().lower() == 'grid_smoothness':
            self.photo_warmup = 0
            return
        try:
            num_pre = int(self.photo_warmup)
        except Exception:
            num_pre = 0
        if num_pre <= 0:
            return
        self._ensure_graph()
        try:
            K = np.asarray(self.prev_intrinsic, dtype=np.float32)
            (yy, xx) = np.meshgrid(np.arange(int(self.H), dtype=np.float32), np.arange(int(self.W), dtype=np.float32), indexing='ij')
            z_xyz = np.asarray(self.prev_normal, dtype=np.float32)
            nz_proj_full = z_xyz[..., 0] * (xx - K[0, 2]) / K[0, 0] + z_xyz[..., 1] * (yy - K[1, 2]) / K[1, 1] + z_xyz[..., 2]
            self._nz_proj_inner = nz_proj_full[1:-1, 1:-1].reshape(-1, 1).astype(np.float32)
        except Exception:
            self._nz_proj_inner = np.zeros((max(0, (self.H - 2) * (self.W - 2)), 1), dtype=np.float32)
        self._nz_param_cache = None
        self.set_hyperparameters()
        linpoints = jax.vmap(lambda g: g.mu())(self.vars.belief)
        for i in range(num_pre):
            _raise_if_sigint_pending()
            if _BINI and i > 0 and i % 100 == 0:
                self.normal_facs.z_Lam = self._bini_weights(linpoints)
                if _NB:
                    self._nb_bini(linpoints)
            if 'safe' in _REF_GAUGE:
                (self.vars, vtof_msgs, linpoints) = update_variable_SE3D_ref(self.vars, linpoints, alpha=self.retract_alpha, msg_slots=self._msg_slots if _SKIP_EMPTY_PORTS else None, u2=self._rg_u2)
            else:
                (self.vars, vtof_msgs, linpoints) = update_variable_SE3D(self.vars, linpoints, alpha=self.retract_alpha, msg_slots=self._msg_slots if _SKIP_EMPTY_PORTS else None)
            linpoints_6 = linpoints[..., :6]
            if 'marg' in _REF_GAUGE:
                (self.between_facs, self.vars) = update_factor_manifold_hJ_ref(self.between_facs, self.vars, vtof_msgs, linpoints_6, h2_fn, l2, self._rg_u2)
            else:
                (self.between_facs, self.vars) = update_factor_manifold_hJ(self.between_facs, self.vars, vtof_msgs, linpoints_6, h2_fn, l2)
            (self.prior_facs, self.vars) = _update_prior_facs(self.prior_facs, self.vars, vtof_msgs, linpoints)
            nz_param = self._cached_nz_param(linpoints.dtype)
            if _NFLOOR_INV:
                (self.normal_facs, self.vars) = update_factor_normal_inv(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, huber if self.use_robust_normal else l2, self._rg_u2 if 'marg' in _REF_GAUGE else jnp.ones((7,), dtype=linpoints.dtype), 'marg' in _REF_GAUGE)
            elif 'marg' in _REF_GAUGE:
                (self.normal_facs, self.vars) = update_factor_normal_ref(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, h_J_fn_normal_ipd, huber if self.use_robust_normal else l2, self._rg_u2)
            else:
                (self.normal_facs, self.vars) = update_factor_normal(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, h_J_fn_normal_ipd, huber if self.use_robust_normal else l2)
            if _NB:
                self._update_normal_border(vtof_msgs, linpoints)
            if (i + 1) % max(1, int(self.iteration_log_interval)) == 0 or i + 1 == num_pre:
                _lp_np = np.array(jax.device_get(linpoints))
                (H, W) = (int(self.H), int(self.W))
                N = int(self.N)
                out_depth = None
                if _lp_np.shape[0] >= N:
                    _grid = _lp_np[:N].reshape(H, W, -1)
                    est_depth = 1.0 / np.maximum(np.asarray(_grid[..., -1], dtype=np.float32), 1e-06)
                    dv_gt = np.asarray(self.prev_depth).astype(np.float32).squeeze()
                    cut = int(self.cut)
                    inner = cut + 1
                    (h0, h1) = (inner, max(inner, H - inner))
                    (w0, w1) = (inner, max(inner, W - inner))
                    if h1 <= h0 or w1 <= w0:
                        est_inner = est_depth[0:0, 0:0]
                        gt_inner = dv_gt[0:0, 0:0]
                    else:
                        est_inner = est_depth[h0:h1, w0:w1]
                        gt_inner = dv_gt[h0:h1, w0:w1]
                    mask_scale = np.isfinite(est_inner) & np.isfinite(gt_inner) & (est_inner > 1e-12) & (gt_inner > 0.0)
                    if np.any(mask_scale):
                        scale = float(np.mean(gt_inner[mask_scale] / est_inner[mask_scale]))
                    else:
                        scale = 1.0
                    est_scaled = (scale * est_depth).astype(np.float32)
                    valid_gt = np.isfinite(dv_gt) & (dv_gt > 0.0)
                    if np.any(valid_gt):
                        dmin = float(dv_gt[valid_gt].min())
                        dmax = float(dv_gt[valid_gt].max())
                        den = dmax - dmin if dmax - dmin > 1e-12 else 1.0
                    else:
                        dmin = 0.0
                        den = 1.0
                    dn = np.clip((est_scaled - dmin) / den, 0.0, 1.0)
                    dn_rgb = (np.stack([dn, dn, dn], axis=2) * 255.0).astype(np.uint8)
                    out_depth = self.logs_dir / f'prewarm_depth_it_{i + 1:06d}.png'
                    cv2.imwrite(str(out_depth), dn_rgb)
                    if _RERUN:
                        self._rerun.log_prewarm(i + 1, num_pre, est_scaled)
                pf_z = np.asarray(jax.device_get(self.prior_facs.z))
                pf_adj = np.asarray(jax.device_get(self.prior_facs.adj_var_id)).reshape(-1)
                mask_leaf = pf_adj < N
                if np.any(mask_leaf) and pf_z.shape[1] >= 7:
                    vals = pf_z[mask_leaf, 6]
                    vals = np.exp(vals)
                    v = vals[np.isfinite(vals)]
                    if self.verbose:
                        print(f'[prior] prewarm it={i + 1}: depth stats(min/mean/max) = {float(v.min()):.4e}/{float(v.mean()):.4e}/{float(v.max()):.4e}', flush=True)
                print(f"[prewarm] it={i + 1}/{num_pre}:{(' depth -> ' + str(out_depth) if out_depth is not None else '')}", flush=True)
        self.photo_warmup = 0

    def _get_current_linpoints(self):
        return jax.vmap(lambda g: g.mu())(self.vars.belief)

    def _set_belief_from_mu(self, mu_full) -> None:
        if self.vars is None or mu_full is None:
            return
        num_v = int(self.vars.belief.eta.shape[0])
        mu_full_np = np.asarray(mu_full, dtype=np.float32)
        if mu_full_np.ndim == 1:
            mu_full_np = np.repeat(mu_full_np.reshape(1, -1), num_v, axis=0)
        elif mu_full_np.shape[0] != num_v:
            if mu_full_np.shape[0] > num_v:
                mu_full_np = mu_full_np[:num_v]
            else:
                mu_full_np = np.repeat(mu_full_np[-1:].reshape(1, -1), num_v, axis=0)
        Lam = np.asarray(jax.device_get(self.vars.belief.Lam))
        if Lam.ndim == 3 and Lam.shape[1] == Lam.shape[2] == mu_full_np.shape[1]:
            eta_full = np.einsum('nij,nj->ni', Lam, mu_full_np)
        else:
            eta_full = mu_full_np
        self.vars.belief.eta = jnp.array(eta_full)

    def _match_linpoints_to_prior(self, linpoints_np: np.ndarray, num_prior: int) -> np.ndarray:
        if linpoints_np.shape[0] == num_prior:
            return linpoints_np
        if linpoints_np.shape[0] > num_prior:
            return linpoints_np[:num_prior]
        if linpoints_np.shape[0] == 0:
            return np.zeros((num_prior, linpoints_np.shape[1]), dtype=np.float32)
        pad_count = num_prior - linpoints_np.shape[0]
        pad = np.repeat(linpoints_np[-1:], pad_count, axis=0)
        return np.concatenate([linpoints_np, pad], axis=0)

    def _update_prior_measurements_from_linpoints(self, linpoints, pose_mode: str='keep', depth_mode: str='keep') -> None:
        if self.prior_facs is None or not hasattr(self.prior_facs, 'z'):
            return
        linpoints_np = np.asarray(jax.device_get(linpoints))
        z_np = np.array(jax.device_get(self.prior_facs.z), copy=True)
        num_prior = z_np.shape[0]
        if linpoints_np.ndim != 2 or z_np.ndim != 2:
            return
        lp_use = self._match_linpoints_to_prior(linpoints_np, num_prior)
        D_op = lp_use.shape[1]
        if pose_mode == 'current' and D_op >= 6 and (z_np.shape[1] >= 6):
            z_np[:, :6] = lp_use[:, :6]
        elif pose_mode == 'zero' and z_np.shape[1] >= 6:
            z_np[:, :6] = 0.0
        if depth_mode == 'current' and D_op >= 7 and (z_np.shape[1] >= 7):
            z_np[:, 6] = lp_use[:, 6]
        self.prior_facs.z = jnp.array(z_np)

    def _reset_pose_estimation_random(self, linpoints) -> None:
        if self.vars is None:
            return
        linpoints_np = np.asarray(jax.device_get(linpoints))
        if linpoints_np.ndim != 2 or linpoints_np.shape[1] == 0:
            return
        mu_new = np.array(linpoints_np, copy=True).astype(np.float32)
        pose_dim = min(6, mu_new.shape[1])
        if pose_dim == 0:
            return
        pia = np.asarray(jax.device_get(self.prior_init_axis))
        if pia.ndim == 0:
            pia = np.zeros((pose_dim,), dtype=np.float32)
        if pia.shape[0] < pose_dim:
            pia = np.pad(pia.astype(np.float32), (0, pose_dim - pia.shape[0]), constant_values=0.0)
        else:
            pia = pia[:pose_dim].astype(np.float32)
        noise_std = float(getattr(self, 'prior_init_noise_std', 0.001))
        noise = np.random.randn(mu_new.shape[0], pose_dim).astype(np.float32) * noise_std
        mu_new[:, :pose_dim] = pia.reshape(1, -1) + noise
        self._set_belief_from_mu(jnp.array(mu_new, dtype=jnp.float32))

    def _render_stage_montage(self, overlay_path, rgb_prev, rgb_cur, depth_vis_full, depth_prev):
        """Overlay the per-iteration twist overlay into a 5-panel montage and save it
        in place: [ref RGB | overlay | target RGB | reconstructed depth | GT depth].
        Records the path for the overlay video export."""
        ov = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if ov is None:
            return
        (H0, W0) = (ov.shape[0], ov.shape[1])
        ref_rgb = np.clip(rgb_prev[..., :3], 0.0, 1.0)
        ref_bgr = (ref_rgb * 255.0).astype(np.uint8)
        ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_RGB2BGR)
        ref_bgr = cv2.resize(ref_bgr, (W0, H0))
        tgt_rgb = np.clip(rgb_cur[..., :3], 0.0, 1.0)
        tgt_bgr = (tgt_rgb * 255.0).astype(np.uint8)
        tgt_bgr = cv2.cvtColor(tgt_bgr, cv2.COLOR_RGB2BGR)
        tgt_bgr = cv2.resize(tgt_bgr, (W0, H0))
        dv_est = depth_vis_full.astype(np.float32)
        dv_gt = np.asarray(depth_prev).astype(np.float32).squeeze()
        valid_gt = np.isfinite(dv_gt) & (dv_gt > 0.0)
        if np.any(valid_gt):
            dmin = float(dv_gt[valid_gt].min())
            dmax = float(dv_gt[valid_gt].max())
            den = dmax - dmin if dmax - dmin > 1e-12 else 1.0
        else:
            dmin = 0.0
            den = 1.0
        dn_est = np.clip((dv_est - dmin) / den, 0.0, 1.0)
        dn_gt = np.clip((dv_gt - dmin) / den, 0.0, 1.0)
        est_rgb = (np.stack([dn_est, dn_est, dn_est], axis=2) * 255.0).astype(np.uint8)
        gt_rgb = (np.stack([dn_gt, dn_gt, dn_gt], axis=2) * 255.0).astype(np.uint8)
        est_rgb = cv2.resize(est_rgb, (W0, H0))
        gt_rgb = cv2.resize(gt_rgb, (W0, H0))
        concatenated = np.concatenate([ref_bgr, ov, tgt_bgr, est_rgb, gt_rgb], axis=1)
        cv2.imwrite(str(overlay_path), concatenated)
        try:
            self._overlay_paths.append(str(overlay_path))
        except Exception:
            pass

    def _energy_line_batched(self, stage_label, it, linpoints, intr_prev_gpu, mask_corners, _hJ):
        """PIXVOD_DIAG_FETCH: the [ENERGY] line of _stage_optimize with the same device and host arithmetic, but the
        device values are fetched in one transfer, the constant index arrays once per run, and the projections the
        original fetched without reading (_uv_e, _qz_e) are not computed."""
        try:
            _nd_e = int(self.dataterm_facs.z.shape[0])
            _ipd_e = '1' == '1'
            if getattr(self, '_ef_vid', None) is None:
                self._ef_vid = np.asarray(jax.device_get(self.dataterm_facs.adj_var_id[:, 0]), dtype=np.int64)
            _vid_e = self._ef_vid
            if _vid_e.size != _nd_e:
                raise ValueError(f'ENERGY dataterm row mismatch: factors={_nd_e}, vids={_vid_e.size}')
            _lp_e = linpoints[_vid_e]
            _p_e = self.grids[_vid_e]
            _h_e = jax.vmap(lambda _x, _pp: jnp.reshape(_hJ(_x, _pp, intr_prev_gpu, self.feat_cuda_curr)[0], ()))(_lp_e, _p_e)
            dev = {'r': self.dataterm_facs.z[:, 0] - _h_e,
                   'ok': jax.vmap(lambda _x, _pp: photometric_flow_inbounds(_x, _pp, intr_prev_gpu, int(self.H), int(self.W), _ipd_e))(_lp_e, _p_e),
                   'corner': mask_corners}
            _mode_e = 'uni'
            _w_src = None
            if _mode_e == 'uni' and getattr(self, '_td_uni_wdev', False):
                _w_src = 'dev'
            elif _mode_e == 'uni' and getattr(self, '_td_uni_w', None) is not None and (self._td_uni_w.shape[0] == _nd_e):
                _w_src = 'host'
            elif self.use_robust:
                _w_src = 'huber'
            if _w_src in ('dev', 'huber'):
                dev['thr'] = self.dataterm_facs.threshold
            _do_n = getattr(self, 'normal_facs', None) is not None and getattr(self, '_nz_proj_inner', None) is not None
            if _do_n:
                _nz_e = jnp.array(self._nz_proj_inner, dtype=linpoints.dtype)
                _nz_rows = jnp.stack([_nz_e, _nz_e], axis=1).reshape(-1, 1)
                _xsd_e = linpoints[self.normal_facs.adj_var_id][..., 6]
                _hfn_e = h_J_fn_normal_ipd
                _hn_e = jax.vmap(lambda _xs, _nzr: _hfn_e(_xs.reshape(-1), _nzr)[0])(_xsd_e, _nz_rows)
                dev['rn'] = self.normal_facs.z - _hn_e
                dev['zln'] = self.normal_facs.z_Lam
            _do_p = getattr(self, 'prior_facs', None) is not None
            if _do_p:
                if getattr(self, '_ef_pid', None) is None:
                    self._ef_pid = np.asarray(jax.device_get(self.prior_facs.adj_var_id[:, 0]), dtype=np.int64)
                _pid_e = self._ef_pid
                dev['zp'] = self.prior_facs.z
                dev['xp'] = linpoints[_pid_e]
                dev['zlp'] = self.prior_facs.z_Lam
            _do_b = getattr(self, 'between_facs', None) is not None
            if _do_b:
                _xb_e = linpoints[self.between_facs.adj_var_id][..., :6]
                _hb_e = jax.vmap(lambda _xs: h2_fn(_xs)[0])(_xb_e)
                dev['rb'] = self.between_facs.z - _hb_e
                dev['zlb'] = self.between_facs.z_Lam
            host = jax.device_get(dev)
            _r_e = np.asarray(host['r'], dtype=np.float64)
            _ok_e = np.asarray(host['ok'], dtype=bool)
            _act_e = _ok_e & np.asarray(host['corner'], dtype=bool).reshape(-1)[:_nd_e]
            _s0_e = float(self.dataterm_s)
            _e_q = (_r_e / _s0_e) ** 2
            if _w_src == 'dev':
                _w_e = np.asarray(host['thr'], dtype=np.float64).reshape(-1)[:_nd_e]
            elif _w_src == 'host':
                _w_e = np.asarray(self._td_uni_w, dtype=np.float64)
            elif _w_src == 'huber':
                _t_h = np.asarray(host['thr'], dtype=np.float64).reshape(-1)[:_nd_e]
                _w_e = np.minimum(1.0, _t_h / np.maximum(np.sqrt(_e_q), 1e-300))
            else:
                _w_e = np.ones(_nd_e)
            _E_d = irls_energies(_r_e, _w_e, _s0_e, _act_e)
            (_NLL_d, _Q_fixed_d, _mask_churn_e, _NLL_n, _NLL_expected) = (float('nan'), float('nan'), float('nan'), 0, 0)
            (_objective_data_valid_e, _objective_kind_e, _irls_metric_accepted_e) = (False, 'none', bool(not False))
            _E_n = float('nan')
            if _do_n:
                _rn_e = np.asarray(host['rn'], dtype=np.float64)
                _zl_n = np.asarray(host['zln'], dtype=np.float64)
                _rn_fin = np.isfinite(_rn_e).all(axis=1)
                if bool(np.all(_rn_fin)) and np.isfinite(_zl_n).all():
                    _E_n = float(np.einsum('ni,nij,nj->', _rn_e, _zl_n, _rn_e))
            _E_p = float('nan')
            if _do_p:
                _zp_e = np.asarray(host['zp'], dtype=np.float64)
                _xp_e = np.asarray(host['xp'], dtype=np.float64)
                if _xp_e.shape == _zp_e.shape:
                    _rp_e = _zp_e - _xp_e
                    _zlp_e = np.asarray(host['zlp'], dtype=np.float64)
                    _rp_fin = np.isfinite(_rp_e).all(axis=1)
                    if bool(np.all(_rp_fin)) and np.isfinite(_zlp_e).all():
                        _E_p = float(np.einsum('ni,nij,nj->', _rp_e, _zlp_e, _rp_e))
            _E_b = float('nan')
            if _do_b:
                _rb_e = np.asarray(host['rb'], dtype=np.float64)
                _zlb_e = np.asarray(host['zlb'], dtype=np.float64)
                _rb_fin = np.isfinite(_rb_e).all(axis=1)
                if bool(np.all(_rb_fin)) and np.isfinite(_zlb_e).all():
                    _E_b = float(np.einsum('ni,nij,nj->', _rb_e, _zlb_e, _rb_e))
            _objective_all_valid_e = bool(False)
            _obj_total_e = float('nan')
            print(f"[ENERGY] stage={stage_label} it={it} e_data_w1={_E_d['E_data_w1']:.6e} e_data_w={_E_d['E_data_w']:.6e} e_normal={_E_n:.6e} e_prior={_E_p:.6e} e_between={_E_b:.6e} nll_data_t={_NLL_d:.6e} q_data_fixed={_Q_fixed_d:.6e} objective_total={_obj_total_e:.6e} objective_valid={int(_objective_all_valid_e)} data_objective_valid={int(_objective_data_valid_e)} objective_kind={_objective_kind_e} mask_churn={_mask_churn_e:.6e} irls_pending={int(not _irls_metric_accepted_e)} n_active={int(_E_d['n'])} n_frozen={int(_NLL_n)}/{int(_NLL_expected)}", flush=True)
        except Exception as _ee:
            print(f'[ENERGY] failed: {_ee}', flush=True)

    def _dev_refresh(self, linpoints, p_use, intr_prev_gpu, mask_corners, stage_label, it):
        """PIXVOD_DEV_REFRESH: the Student-t refresh as one device program (_dev_tdist_refresh); at the logged iteration the
        [TDIST-UNI] line is printed with the same fields from a few scalars fetched in one transfer.  The weights stay on
        the device: [ENERGY] reads them from the photometric factors' threshold (the values the factors use)."""
        _nu_eff_uni = nu_effective(stage_index(stage_label), float(self.cfg['student_t_nu']), int('0'))
        _k_uni = float(self.cfg['predictive_variance_scale'])
        _si_uni = float('1') * float(self.dataterm_s)
        _sw_uni = float('0')
        stats = it % max(1, int(self.iteration_log_interval)) == 0
        dt = linpoints.dtype
        (w, st) = _dev_tdist_refresh(linpoints, p_use, intr_prev_gpu, self.feat_cuda_curr, self.dataterm_facs.z, self.vars.belief.Lam, mask_corners,
                                     jnp.asarray(_k_uni * _k_uni, dt), jnp.asarray(_si_uni ** 2, dt), jnp.asarray(_sw_uni ** 2, dt), jnp.asarray(_nu_eff_uni, dt),
                                     stats=stats, H=int(self.H), W=int(self.W))
        self.dataterm_facs.threshold = w
        self._td_uni_stage = stage_label
        self._td_uni_w = None
        self._td_uni_wdev = True
        self._td_uni_active = None
        self._td_chart = None
        if stats:
            (vals, n) = jax.device_get(st)
            (z_med, z_p95, wbar, jvj) = (float(x) for x in vals)
            print(f'[TDIST-UNI] stage={stage_label} it={it} nu_eff={_nu_eff_uni:.4g} sI={_si_uni:g} swarp={_sw_uni:g} z_med={z_med:.3f} z_p95={z_p95:.3f} wbar={wbar:.4f} candidate_wbar={wbar:.4f} jvj_share_med={jvj:.3f} alpha={0.0:.4f} n={int(n)}', flush=True)

    def _stage_optimize(self, num_iters: int, rgb_prev: np.ndarray, depth_prev: np.ndarray, rgb_cur: np.ndarray, intrinsic_prev: np.ndarray, extrinsic_prev: np.ndarray, extrinsic_current: np.ndarray, extrinsic_eval_cur: np.ndarray, do_prior_reinit_after: bool, stage_label: str | None=None, ref_main_idx: int | None=None, return_linpoints: bool=False):
        if num_iters <= 0:
            return
        if self._check_global_iter_limit() or getattr(self.killer, 'stop', False):
            return
        _ui_capture = None
        self._ensure_graph()
        feat_prev = (rgb_prev[..., :3] * np.array([[[0.2989, 0.587, 0.114]]])).sum(axis=-1, keepdims=True).astype(np.float32)
        _photo_blur = float('0')
        self.prev_intrinsic = np.asarray(intrinsic_prev)
        _z_fw = feat_prev[self.cut:self.H - self.cut, self.cut:self.W - self.cut, :].reshape(-1, self.D)
        if _STAGE_CONST:  # the reference frame is the same at every stage: build each constant once per distinct input
            _pn = np.asarray(self.prev_normal)
            _pk = np.asarray(self.prev_intrinsic)
            self.dataterm_facs.z = self._stage_const(('dataterm_z', _z_fw.shape, str(_z_fw.dtype), _z_fw.tobytes()), lambda: jnp.array(_z_fw))
            self.normal_facs.z = self._stage_const(('normal_z', _pn.shape, str(_pn.dtype), _pn.tobytes(), _pk.shape, str(_pk.dtype), _pk.tobytes()), lambda: jax.vmap(lambda N, K: jnp.array([[N[0] / K[0, 0], N[0] / K[0, 0]], [N[1] / K[1, 1], N[1] / K[1, 1]]]), in_axes=(0, None))(self.prev_normal[1:-1, 1:-1, :].reshape(-1, 3), self.prev_intrinsic).reshape(-1, 2))
        else:
            self.dataterm_facs.z = jnp.array(_z_fw)
            self.normal_facs.z = jax.vmap(lambda N, K: jnp.array([[N[0] / K[0, 0], N[0] / K[0, 0]], [N[1] / K[1, 1], N[1] / K[1, 1]]]), in_axes=(0, None))(self.prev_normal[1:-1, 1:-1, :].reshape(-1, 3), self.prev_intrinsic).reshape(-1, 2)
        K = np.asarray(self.prev_intrinsic, dtype=np.float32)
        (yy, xx) = np.meshgrid(np.arange(int(self.H), dtype=np.float32), np.arange(int(self.W), dtype=np.float32), indexing='ij')
        z_xyz = np.asarray(self.prev_normal, dtype=np.float32)
        nz_proj_full = z_xyz[..., 0] * (xx - K[0, 2]) / K[0, 0] + z_xyz[..., 1] * (yy - K[1, 2]) / K[1, 1] + z_xyz[..., 2]
        _nz_new = nz_proj_full[1:-1, 1:-1].reshape(-1, 1).astype(np.float32)
        if not (_STAGE_CONST and getattr(self, '_nz_proj_inner', None) is not None and np.asarray(self._nz_proj_inner).dtype == _nz_new.dtype
                and np.asarray(self._nz_proj_inner).shape == _nz_new.shape and np.asarray(self._nz_proj_inner).tobytes() == _nz_new.tobytes()):
            self._nz_proj_inner = _nz_new
            self._nz_param_cache = None
        if self._maybe_apply_scheduled_hyperparameters():
            self._set_prior_measurements(self.prior_init_noise_std)
        self.set_hyperparameters()
        _pose_schur_beta_stage = float('0')
        _cov_gauge_on = '0' == '1'
        _gauge_q_stage = float(getattr(self, '_gauge_total', 1.0))
        _rho_floor_stage = 1e-06
        _gauge_axes_stage = jnp.asarray(gauge_axis_factors(_gauge_q_stage, int(self.D_op)), dtype=self.vars.belief.eta.dtype)
        if not self.disable_photometric:
            _gc = (rgb_cur[..., :3] * np.array([[[0.2989, 0.587, 0.114]]])).sum(axis=-1, keepdims=True).astype(np.float32)
            _pb = float('0')
            self.feat_cuda_curr = _put_gpu(_gc)
            if _STAGE_CONST:
                _ik = np.asarray(intrinsic_prev)
                intr_prev_gpu = self._stage_const(('intr', _ik.shape, str(_ik.dtype), _ik.tobytes()), lambda: _put_gpu(intrinsic_prev), donated=False)
            else:
                intr_prev_gpu = _put_gpu(intrinsic_prev)
        if self.grids is None and (not self.disable_photometric):
            grid_np = np.array([[n % self.W, n // self.W] for n in range(self.N)], dtype=np.float64) + 0.5
            self.grids = _put_gpu(grid_np)
            if int(self.cut) > 0:
                mask1 = np.zeros((self.H, self.W), dtype=bool)
                mask1[self.cut:self.H - self.cut, self.cut:self.W - self.cut] = True
            else:
                mask1 = np.ones((self.H, self.W), dtype=bool)
            if self.photometric_stride > 1:
                mask2 = np.zeros((self.H, self.W), dtype=bool)
                mask2[::self.photometric_stride, ::self.photometric_stride] = True
            else:
                mask2 = np.ones((self.H, self.W), dtype=bool)
            active_ids = np.nonzero((mask1 & mask2).flatten())[0].astype(np.int32)
            self.active_ids = _put_gpu(active_ids)
        T_prev_inv = np.linalg.inv(extrinsic_prev)
        T_rel_eval = extrinsic_eval_cur @ T_prev_inv
        T_rel_cur = extrinsic_current @ T_prev_inv
        gt_twist_eval = SE3_Log(jnp.array(T_rel_eval))
        gt_twist_cur = SE3_Log(jnp.array(T_rel_cur))
        _x1_mode = ''
        if _EXACT_READOUT:
            linpoints = jax.vmap(_mu_exact)(self.vars.belief)
            if _READOUT_GUARD and (_n_rg := _readout_guard_count(self.vars.belief)):
                print(f'[READOUT-GUARD] stage={stage_label} at=start fallback={_n_rg}', flush=True)
        else:
            linpoints = jax.vmap(lambda x: x.mu())(self.vars.belief)
        if _BINI:
            self.normal_facs.z_Lam = self._bini_weights(linpoints)
            if _NB:
                self._nb_bini(linpoints)
        est_twist0 = jnp.mean(linpoints[..., :6], axis=0)
        (disp0_main, _) = se3_geo_dist_explicit(jnp.zeros_like(est_twist0), gt_twist_eval)
        d0_main = float(np.linalg.norm(np.array(jax.device_get(disp0_main))))
        (disp0_cur, _) = se3_geo_dist_explicit(jnp.zeros_like(est_twist0), gt_twist_cur)
        d0_cur = float(np.linalg.norm(np.array(jax.device_get(disp0_cur))))
        est_twist_trace = []
        norm_geo_err_trace = []

        def _compute_depth_and_scale(_linpoints_np, _gt_depth_2d):
            (H, W) = (int(self.H), int(self.W))
            cut = int(self.cut)
            lp_grid = _linpoints_np[:self.N].reshape(H, W, -1)
            est_depth = 1.0 / np.maximum(np.asarray(lp_grid[..., -1], dtype=np.float64), _rho_floor_stage)
            inner = cut + 1
            (h0, h1) = (inner, max(inner, H - inner))
            (w0, w1) = (inner, max(inner, W - inner))
            if h1 <= h0 or w1 <= w0:
                est_inner = est_depth[0:0, 0:0]
                gt_inner = np.asarray(_gt_depth_2d, dtype=np.float64)[0:0, 0:0]
            else:
                est_inner = est_depth[h0:h1, w0:w1]
                gt_inner = np.asarray(_gt_depth_2d, dtype=np.float64)[h0:h1, w0:w1]
            mask = np.isfinite(est_inner) & np.isfinite(gt_inner) & (est_inner > 1e-12) & (gt_inner > 0.0)
            if np.any(mask):
                scale = float(np.mean(gt_inner[mask] / est_inner[mask]))
            else:
                scale = 1.0
            scaled_full = (scale * est_depth).astype(np.float32)
            return (scaled_full, scale)

        def _scale_twist_translation(_xi_jnp, _scale: float):
            T_est = se3_Exp(_xi_jnp)
            t_scaled = T_est[:3, 3] * _scale
            T_est_scaled = T_est.at[:3, 3].set(t_scaled)
            return SE3_Log(T_est_scaled)
        # PIXVOD_DIAG_FETCH: the stage-entry metrics below are read only by the first point of the trend (stage 1); later
        # stages skip them (a device-to-host copy of every linearisation point and a few device round trips).  The
        # stage-entry depth scale they set is overwritten at the stage's logged iteration; if a stage ends without one
        # (interrupted), it is computed in the finally block below from the same linearisation points.
        _entry_metrics = not (_DIAG_FETCH and self._comb_initialized)
        _entry_lp = linpoints
        if _entry_metrics:
            linpoints_np0 = np.array(jax.device_get(linpoints))
            gt_depth2d = np.asarray(depth_prev).squeeze()
            (scaled_depth_full0, scale0) = _compute_depth_and_scale(linpoints_np0, gt_depth2d)
            self._cur_depth_scale = scale0
            est_twist0_scaled = _scale_twist_translation(est_twist0, self._cur_depth_scale)
            try:
                leaf_twists0 = np.asarray(linpoints_np0[:int(self.N), :6], dtype=np.float32)
                est_std0 = np.std(leaf_twists0, axis=0).astype(np.float32)
                est_std0[3:6] = est_std0[3:6] * float(self._cur_depth_scale)
            except Exception:
                est_std0 = np.zeros((6,), dtype=np.float32)
        try:
            var_level = self._compute_var_levels()
            root_level = int(np.max(var_level))
            root_ids = np.where(var_level == root_level)[0]
            root_id = int(root_ids[-1]) if root_ids.size > 0 else int(np.argmax(var_level))
        except Exception:
            root_id = int((linpoints_np0 if _entry_metrics else linpoints).shape[0] - 1)
        if _entry_metrics:
            try:
                root_twist0 = jnp.array(linpoints[root_id, :6])
                root_est0_scaled = _scale_twist_translation(root_twist0, self._cur_depth_scale)
                (disp_root0_cur, _) = se3_geo_dist_explicit(root_est0_scaled, gt_twist_cur)
                geo_root0_cur = float(np.linalg.norm(np.array(jax.device_get(disp_root0_cur))))
                den_root0_cur = d0_cur if d0_cur > 1e-12 else 1e-12
                norm_err_root0_cur = geo_root0_cur / den_root0_cur
            except Exception:
                root_est0_scaled = est_twist0_scaled
                norm_err_root0_cur = float('nan')
        if not self._comb_initialized:
            (disp_init0_main, _) = se3_geo_dist_explicit(est_twist0_scaled, gt_twist_eval)
            geo0_main = float(np.linalg.norm(np.array(jax.device_get(disp_init0_main))))
            den0_main = d0_main if d0_main > 1e-12 else 1e-12
            norm_err0_main = geo0_main / den0_main
            (disp_init0_cur, _) = se3_geo_dist_explicit(est_twist0_scaled, gt_twist_cur)
            geo0_cur = float(np.linalg.norm(np.array(jax.device_get(disp_init0_cur))))
            den0_cur = d0_cur if d0_cur > 1e-12 else 1e-12
            norm_err0_cur = geo0_cur / den0_cur
            is_sub_stage = stage_label is not None and ('sub' in stage_label and 'main' not in stage_label)
            self._append_combined_point_at(self._comb_iter_counter, norm_err0_main, norm_err0_cur, est_twist0_scaled, est_std0, root_est0_scaled, norm_err_root0_cur, gt_twist_eval, gt_twist_cur, is_sub_stage)
            self._save_combined_plots()
        stage_iter_base = int(self._comb_iter_counter)
        self._aa_f = []
        self._aa_g = []
        _cc_tol = float('0')
        _cc_win = int('10')
        _cc_hits = 0
        _cc_stopped = False
        self._cc_deltas = []
        _cc_tail_n = int('0')
        _cc_tail_stages = set((int(x) for x in ''.split(',') if x.strip()))
        _cc_tail_on = False
        self._cc_tail = []
        _td_irls_on = '0' == '1'
        _td_irls_shadow = False
        _td_irls_controller = None
        _td_irls_checkpoint = None
        _td_irls_pretrial_checkpoint = None
        _td_irls_prev_lp = None
        _td_irls_settle_model = None
        _td_irls_trial_mask_ok = True
        _td_irls_support_rebase_pending = False
        _td_irls_shadow_last_q = None
        _td_irls_mode = None
        try:
            for it in range(int(num_iters)):
                _raise_if_sigint_pending()
                if self._check_global_iter_limit() or getattr(self.killer, 'stop', False):
                    break
                _td_irls_deferred_proposal = None
                _td_irls_deferred_commit = None
                if self._maybe_apply_scheduled_hyperparameters():
                    self.set_hyperparameters()
                    self._set_prior_measurements(self.prior_init_noise_std)
                _cc_prev_lp = None
                if it == 0 and '0' == '1' and (not getattr(self, '_sb_done', False)):
                    self._sb_done = True
                    _items = [('linpoints', linpoints), ('belief.eta', self.vars.belief.eta), ('belief.Lam', self.vars.belief.Lam), ('msgs.eta', self.vars.msgs.eta), ('msgs.Lam', self.vars.msgs.Lam)]
                    for (_nm, _fc) in [('prior', self.prior_facs), ('between', self.between_facs), ('dataterm', self.dataterm_facs), ('normal', getattr(self, 'normal_facs', None)), ('depth_grid', getattr(self, 'depth_grid_facs', None))]:
                        if _fc is None:
                            continue
                        for _a in ('z', 'z_Lam', 'linpoints', 'adj_var_id'):
                            _v = getattr(_fc, _a, None)
                            if _v is not None:
                                _items.append((f'{_nm}.{_a}', _v))
                    _tot = 0
                    for (_nm, _v) in _items:
                        _b = int(getattr(_v, 'nbytes', 0))
                        _tot += _b
                        print(f"[STATE-BYTES] {_nm}: shape={getattr(_v, 'shape', None)} dtype={getattr(_v, 'dtype', None)} {_b / 1000000.0:.2f} MB", flush=True)
                    print(f'[STATE-BYTES] TOTAL listed: {_tot / 1000000.0:.2f} MB', flush=True)
                if it == 0 and '0' == '1' and (self.prior_facs is not None):
                    print(f'[CARRY-DBG] stage={stage_label} entry z_Lam[0] diag(rot,tr,dep)={float(self.prior_facs.z_Lam[0, 0, 0]):.3e},{float(self.prior_facs.z_Lam[0, 3, 3]):.3e},{(float(self.prior_facs.z_Lam[0, 6, 6]) if int(self.prior_facs.z_Lam.shape[1]) >= 7 else 0):.3e}', flush=True)
                _ls_alpha = self.retract_alpha
                _eta_after = int('0')
                _eta_relin = int('10')
                _eta_burst = int('10')
                _eta_main_only = '0' == '1'
                _eta_stage_ok = True
                _eta_start_sub = int('0')
                _cur_sub = 0
                if stage_label is not None and 'sub' in stage_label:
                    try:
                        _cur_sub = int(stage_label.split('sub')[-1])
                    except Exception:
                        _cur_sub = 0
                _eta_end_sub = int('0')
                _eta_sub_ok = _cur_sub >= _eta_start_sub and True
                _eta_on = False
                if getattr(self, '_ao_off', False):
                    _eta_on = False
                _cyc = -1
                _eta_phase2 = False
                _eta_burst_end = False
                _gn_on = False
                _gn_freeze_now = False
                _mech_lp_before = None
                _td_irls_safe_boundary = bool(False)
                _aa_xb = linpoints
                if 'safe' in _REF_GAUGE:
                    (self.vars, vtof_msgs, linpoints) = update_variable_SE3D_ref(self.vars, linpoints, alpha=_ls_alpha, msg_slots=self._msg_slots if _SKIP_EMPTY_PORTS else None, u2=self._rg_u2)
                else:
                    (self.vars, vtof_msgs, linpoints) = update_variable_SE3D(self.vars, linpoints, alpha=_ls_alpha, msg_slots=self._msg_slots if _SKIP_EMPTY_PORTS else None)
                _aa_m = int('0')
                _mech_kick = None
                _msg_damp = float('0')
                _msg_eta_prev = None
                linpoints_6 = linpoints[..., :6]
                if 'marg' in _REF_GAUGE:
                    (self.between_facs, self.vars) = update_factor_manifold_hJ_ref(self.between_facs, self.vars, vtof_msgs, linpoints_6, h2_fn, l2, self._rg_u2)
                else:
                    (self.between_facs, self.vars) = update_factor_manifold_hJ(self.between_facs, self.vars, vtof_msgs, linpoints_6, h2_fn, l2)
                (self.prior_facs, self.vars) = _update_prior_facs(self.prior_facs, self.vars, vtof_msgs, linpoints)
                nz_param = self._cached_nz_param(linpoints.dtype)
                if _NFLOOR_INV:
                    (self.normal_facs, self.vars) = update_factor_normal_inv(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, huber if self.use_robust_normal else l2, self._rg_u2 if 'marg' in _REF_GAUGE else jnp.ones((7,), dtype=linpoints.dtype), 'marg' in _REF_GAUGE)
                elif 'marg' in _REF_GAUGE:
                    (self.normal_facs, self.vars) = update_factor_normal_ref(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, h_J_fn_normal_ipd, huber if self.use_robust_normal else l2, self._rg_u2, rho_floor=_rho_floor_stage, dynamic_rho_floor=_cov_gauge_on)
                else:
                    (self.normal_facs, self.vars) = update_factor_normal(self.normal_facs, self.vars, vtof_msgs, linpoints, nz_param, h_J_fn_normal_ipd, huber if self.use_robust_normal else l2, rho_floor=_rho_floor_stage, dynamic_rho_floor=_cov_gauge_on)
                if _NB:
                    self._update_normal_border(vtof_msgs, linpoints)
                if not self.disable_photometric and it + 1 >= self.photo_warmup:
                    _hJ = h_J_fn_general
                    _h_sym_any = False
                    (p_use, mask_corners) = self._cached_photo_grid()
                    _td_irls_current_active = None
                    _p_photo = p_use
                    _hyb_mode = ''
                    _hyb_force = '0' == '1'
                    _tdist_mode = 'uni'
                    if _tdist_mode in ('1', 'dvo', 'uni', 'ue') and 'normal_integration' != 'grid_smoothness' and (int(self.D) == 1) and (not _h_sym_any):
                        _skip_td = False
                        if not _td_irls_on and (not _skip_td) and (_tdist_mode in ('uni', 'ue')) and (getattr(self, '_td_uni_stage', None) == stage_label) and (it % max(1, int(self.cfg['weight_update_interval'])) != 0):
                            self._td_chart = None
                        elif _DEV_REFRESH:
                            self._dev_refresh(linpoints, p_use, intr_prev_gpu, mask_corners, stage_label, it)
                        else:
                            _nu_td = float(self.cfg['student_t_nu'])
                            _nd_td = int(self.dataterm_facs.z.shape[0])
                            _ipd_td = '1' == '1'
                            if not _DIAG_FETCH:  # never read (PIXVOD_DIAG_FETCH skips the copies)
                                _vid_td = np.asarray(jax.device_get(self.dataterm_facs.adj_var_id[:, 0]), dtype=np.int64)
                                _port_td = np.asarray(jax.device_get(self.dataterm_facs.adj_var_idx[:, 0]), dtype=np.int64)
                            _exact_rows_td = False
                            _lp_rows_td = linpoints[:_nd_td]
                            _p_rows_td_jax = p_use[:_nd_td]

                            def _fwd_td(_x, _pp):
                                _dz = 1.0 / jnp.maximum(_x[6], _rho_floor_stage)
                                _q = se3_move_point(_x[:6], unproject_point(_dz, _pp, intr_prev_gpu))
                                return (project_point(_q, intr_prev_gpu), _q[2])
                            (_uv_td, _qz_td) = jax.vmap(_fwd_td)(_lp_rows_td, _p_rows_td_jax)
                            _h_td = jax.vmap(lambda _q: jnp.reshape(bilinear_interpolate_jax_xy(self.feat_cuda_curr[..., 0], photometric_array_xy(_q)), ()))(_uv_td)
                            if _DIAG_FETCH:  # the same device values, fetched in one transfer
                                _J_uni_j = jax.vmap(lambda _x, _pp: jnp.atleast_2d(_hJ(_x, _pp, intr_prev_gpu, self.feat_cuda_curr)[1]))(_lp_rows_td, _p_rows_td_jax)
                                (_r_f, _uv_f, _qz_f, _cn_f, _J_f, _Lb_f) = jax.device_get((self.dataterm_facs.z[:, 0] - _h_td, _uv_td, _qz_td, mask_corners, _J_uni_j, self.vars.belief.Lam[:_nd_td]))
                                _r_all_td = np.asarray(_r_f, dtype=np.float64)
                                _uv_np_td = np.asarray(_uv_f, dtype=np.float64)
                                _qz_np_td = np.asarray(_qz_f, dtype=np.float64)
                                (_Ht_td, _Wt_td) = (int(self.H), int(self.W))
                                _ok_td = np.isfinite(_uv_np_td).all(axis=1) & np.isfinite(_qz_np_td) & (_qz_np_td > 0.0) & (_uv_np_td[:, 0] >= 0.0) & (_uv_np_td[:, 0] < _Wt_td) & (_uv_np_td[:, 1] >= 0.0) & (_uv_np_td[:, 1] < _Ht_td) & np.isfinite(_r_all_td)
                                _corner_np_td = np.asarray(_cn_f, dtype=bool).reshape(-1)
                                _active_td = _ok_td & _corner_np_td[:_nd_td]
                                _J_uni = np.asarray(_J_f[:, 0, :], dtype=np.float64)
                                _Lb_uni = np.asarray(_Lb_f, dtype=np.float64)
                            else:
                                _r_all_td = np.asarray(jax.device_get(self.dataterm_facs.z[:, 0] - _h_td), dtype=np.float64)
                                _uv_np_td = np.asarray(jax.device_get(_uv_td), dtype=np.float64)
                                _qz_np_td = np.asarray(jax.device_get(_qz_td), dtype=np.float64)
                                (_Ht_td, _Wt_td) = (int(self.H), int(self.W))
                                _ok_td = np.isfinite(_uv_np_td).all(axis=1) & np.isfinite(_qz_np_td) & (_qz_np_td > 0.0) & (_uv_np_td[:, 0] >= 0.0) & (_uv_np_td[:, 0] < _Wt_td) & (_uv_np_td[:, 1] >= 0.0) & (_uv_np_td[:, 1] < _Ht_td) & np.isfinite(_r_all_td)
                                _corner_np_td = np.asarray(jax.device_get(mask_corners), dtype=bool).reshape(-1)
                                _active_td = _ok_td & _corner_np_td[:_nd_td]
                                _J_uni_j = jax.vmap(lambda _x, _pp: jnp.atleast_2d(_hJ(_x, _pp, intr_prev_gpu, self.feat_cuda_curr)[1]))(_lp_rows_td, _p_rows_td_jax)
                                _J_uni = np.asarray(jax.device_get(_J_uni_j)[:, 0, :], dtype=np.float64)
                                _Lb_uni = np.asarray(jax.device_get(self.vars.belief.Lam[:_nd_td]), dtype=np.float64)
                            try:
                                _v_uni = _batched_inv_diag6(_Lb_uni)
                            except np.linalg.LinAlgError:
                                _v_uni = np.array([float(np.linalg.pinv(_L)[6, 6]) for _L in _Lb_uni])
                            _v_uni = np.where(np.isfinite(_v_uni) & (_v_uni > 0.0), _v_uni, 0.0)
                            _sigma_mode_uni = 'per_pixel'
                            _k_uni = float(self.cfg['predictive_variance_scale'])
                            _v_uni = _v_uni * (_k_uni * _k_uni)
                            _si_uni = float('1') * float(self.dataterm_s)
                            _sw_uni = float('0')
                            _D_uni = predictive_denominator(_J_uni[:, 6], _v_uni, _si_uni, _sw_uni)
                            _nu_eff_uni = nu_effective(stage_index(stage_label), _nu_td, int('0'))
                            _w_uni = tdist_weight_std(_r_all_td, _D_uni, _nu_eff_uni)
                            _w_uni = np.where(_active_td, _w_uni, 1.0)
                            _beta_uni = float('0')
                            (_alpha_uni, _ratio_uni) = (0.0, float('nan'))
                            _pose_schur_handled_uni = False
                            self.dataterm_facs.threshold = jnp.asarray(_w_uni.astype(np.float32).reshape(-1, 1))
                            self._td_uni_stage = stage_label
                            self._td_uni_w = _w_uni
                            self._td_uni_active = _active_td
                            self._td_chart = None
                            if it % max(1, int(self.iteration_log_interval)) == 0:
                                _zA = np.abs(_r_all_td[_active_td]) / np.sqrt(np.maximum(_D_uni[_active_td], 1e-300))
                                _shareJV = _J_uni[_active_td, 6] ** 2 * _v_uni[_active_td] / np.maximum(_D_uni[_active_td], 1e-300)
                                _irls_fields_uni = ''
                                if _DIAG_FETCH:  # the uploaded float32 weights themselves, without the round trip
                                    _applied_weights_uni = _w_uni.astype(np.float32).astype(np.float64)
                                else:
                                    _applied_weights_uni = np.asarray(jax.device_get(self.dataterm_facs.threshold[:, 0]), dtype=np.float64)
                                _applied_support_uni = _active_td
                                _applied_wbar_uni = float(np.mean(_applied_weights_uni[_applied_support_uni])) if np.any(_applied_support_uni) else float('nan')
                                _candidate_wbar_uni = float(np.mean(_w_uni[_active_td])) if np.any(_active_td) else float('nan')
                                print(f'[TDIST-UNI] stage={stage_label} it={it} nu_eff={_nu_eff_uni:.4g} {_irls_fields_uni}sI={_si_uni:g} swarp={_sw_uni:g} z_med={float(np.median(_zA)):.3f} z_p95={float(np.percentile(_zA, 95)):.3f} wbar={_applied_wbar_uni:.4f} candidate_wbar={_candidate_wbar_uni:.4f} jvj_share_med={float(np.median(_shareJV)):.3f} alpha={_alpha_uni:.4f} n={int(_active_td.sum())}', flush=True)
                    _upd_fn = update_factor_flow_hJ
                    _w_data = huber if self.use_robust else l2
                    _td_mode_weight = 'uni'
                    _td_common_weight_ok = '0' != '1' and 'normal_integration' != 'grid_smoothness' and (int(self.D) == 1) and (not _h_sym_any)
                    _ue_weight_ok = False
                    if _td_mode_weight in ('1', 'dvo', 'uni') and _td_common_weight_ok or _ue_weight_ok:
                        if getattr(self, '_tdist_wfn', None) is None:
                            _nu_c = float('10')

                            def _tdw(e, t, _nu=_nu_c):
                                return t
                            self._tdist_wfn = _tdw
                        _w_data = self._tdist_wfn
                    _photo_message_support = mask_corners
                    (self.dataterm_facs, self.vars) = _upd_fn(self.dataterm_facs, self.vars, vtof_msgs, linpoints, _p_photo, intr_prev_gpu, self.feat_cuda_curr, _hJ, _w_data, mask_ftov=_photo_message_support)
                    if _DIAG_FETCH and '1' == '1' and 'normal_integration' != 'grid_smoothness' and (int(self.D) == 1) and ((it + 1) % max(1, int(self.iteration_log_interval)) == 0 or it == int(num_iters) - 1):
                        self._energy_line_batched(stage_label, it, linpoints, intr_prev_gpu, mask_corners, _hJ)
                    if not _DIAG_FETCH and '1' == '1' and 'normal_integration' != 'grid_smoothness' and (int(self.D) == 1) and ((it + 1) % max(1, int(self.iteration_log_interval)) == 0 or it == int(num_iters) - 1):
                        try:
                            _nd_e = int(self.dataterm_facs.z.shape[0])
                            _ipd_e = '1' == '1'

                            def _fwd_e(_x, _pp):
                                _dz = 1.0 / jnp.maximum(_x[6], _rho_floor_stage)
                                _q = se3_move_point(_x[:6], unproject_point(_dz, _pp, intr_prev_gpu))
                                return (project_point(_q, intr_prev_gpu), _q[2])
                            _vid_e = np.asarray(jax.device_get(self.dataterm_facs.adj_var_id[:, 0]), dtype=np.int64)
                            if _vid_e.size != _nd_e:
                                raise ValueError(f'ENERGY dataterm row mismatch: factors={_nd_e}, vids={_vid_e.size}')
                            _lp_e = linpoints[_vid_e]
                            _p_e = self.grids[_vid_e]
                            (_uv_e, _qz_e) = jax.vmap(_fwd_e)(_lp_e, _p_e)
                            _h_e = jax.vmap(lambda _x, _pp: jnp.reshape(_hJ(_x, _pp, intr_prev_gpu, self.feat_cuda_curr)[0], ()))(_lp_e, _p_e)
                            _r_e = np.asarray(jax.device_get(self.dataterm_facs.z[:, 0] - _h_e), dtype=np.float64)
                            _uv_ne = np.asarray(jax.device_get(_uv_e), dtype=np.float64)
                            _qz_ne = np.asarray(jax.device_get(_qz_e), dtype=np.float64)
                            _ok_e = np.asarray(jax.device_get(jax.vmap(lambda _x, _pp: photometric_flow_inbounds(_x, _pp, intr_prev_gpu, int(self.H), int(self.W), _ipd_e))(_lp_e, _p_e)), dtype=bool)
                            _act_e = _ok_e & np.asarray(jax.device_get(mask_corners), dtype=bool).reshape(-1)[:_nd_e]
                            _s0_e = float(self.dataterm_s)
                            _e_q = (_r_e / _s0_e) ** 2
                            _mode_e = 'uni'
                            _ue_explicit_e = False
                            if _mode_e == 'uni' and getattr(self, '_td_uni_wdev', False):  # PIXVOD_DEV_REFRESH: the applied weights
                                _w_e = np.asarray(jax.device_get(self.dataterm_facs.threshold), dtype=np.float64).reshape(-1)[:_nd_e]
                            elif _mode_e == 'uni' and getattr(self, '_td_uni_w', None) is not None and (self._td_uni_w.shape[0] == _nd_e):
                                _w_e = np.asarray(self._td_uni_w, dtype=np.float64)
                            elif self.use_robust:
                                _t_h = np.asarray(jax.device_get(self.dataterm_facs.threshold), dtype=np.float64).reshape(-1)[:_nd_e]
                                _w_e = np.minimum(1.0, _t_h / np.maximum(np.sqrt(_e_q), 1e-300))
                            else:
                                _w_e = np.ones(_nd_e)
                            _E_d = irls_energies(_r_e, _w_e, _s0_e, _act_e)
                            _NLL_d = float('nan')
                            _Q_fixed_d = float('nan')
                            _mask_churn_e = float('nan')
                            _NLL_n = 0
                            _NLL_expected = 0
                            _objective_data_valid_e = False
                            _objective_kind_e = 'none'
                            _irls_metric_accepted_e = bool(not False)
                            _E_n = float('nan')
                            if getattr(self, 'normal_facs', None) is not None and getattr(self, '_nz_proj_inner', None) is not None:
                                _nz_e = jnp.array(self._nz_proj_inner, dtype=linpoints.dtype)
                                _nz_rows = jnp.stack([_nz_e, _nz_e], axis=1).reshape(-1, 1)
                                _xsd_e = linpoints[self.normal_facs.adj_var_id][..., 6]
                                _hfn_e = h_J_fn_normal_ipd
                                _hn_e = jax.vmap(lambda _xs, _nzr: _hfn_e(_xs.reshape(-1), _nzr)[0])(_xsd_e, _nz_rows)
                                _rn_e = np.asarray(jax.device_get(self.normal_facs.z - _hn_e), dtype=np.float64)
                                _zl_n = np.asarray(jax.device_get(self.normal_facs.z_Lam), dtype=np.float64)
                                _rn_fin = np.isfinite(_rn_e).all(axis=1)
                                if bool(np.all(_rn_fin)) and np.isfinite(_zl_n).all():
                                    _E_n = float(np.einsum('ni,nij,nj->', _rn_e, _zl_n, _rn_e))
                            _E_p = float('nan')
                            if getattr(self, 'prior_facs', None) is not None:
                                _zp_e = np.asarray(jax.device_get(self.prior_facs.z), dtype=np.float64)
                                _np_e = _zp_e.shape[0]
                                _pid_e = np.asarray(jax.device_get(self.prior_facs.adj_var_id[:, 0]), dtype=np.int64)
                                _xp_e = np.asarray(jax.device_get(linpoints[_pid_e]), dtype=np.float64)
                                if _xp_e.shape == _zp_e.shape:
                                    _rp_e = _zp_e - _xp_e
                                    _zlp_e = np.asarray(jax.device_get(self.prior_facs.z_Lam), dtype=np.float64)
                                    _rp_fin = np.isfinite(_rp_e).all(axis=1)
                                    if bool(np.all(_rp_fin)) and np.isfinite(_zlp_e).all():
                                        _E_p = float(np.einsum('ni,nij,nj->', _rp_e, _zlp_e, _rp_e))
                            _E_b = float('nan')
                            if getattr(self, 'between_facs', None) is not None:
                                _xb_e = linpoints[self.between_facs.adj_var_id][..., :6]
                                _hb_e = jax.vmap(lambda _xs: h2_fn(_xs)[0])(_xb_e)
                                _rb_e = np.asarray(jax.device_get(self.between_facs.z - _hb_e), dtype=np.float64)
                                _zlb_e = np.asarray(jax.device_get(self.between_facs.z_Lam), dtype=np.float64)
                                _rb_fin = np.isfinite(_rb_e).all(axis=1)
                                if bool(np.all(_rb_fin)) and np.isfinite(_zlb_e).all():
                                    _E_b = float(np.einsum('ni,nij,nj->', _rb_e, _zlb_e, _rb_e))
                            _objective_all_valid_e = bool(False)
                            _obj_total_e = float('nan')
                            print(f"[ENERGY] stage={stage_label} it={it} e_data_w1={_E_d['E_data_w1']:.6e} e_data_w={_E_d['E_data_w']:.6e} e_normal={_E_n:.6e} e_prior={_E_p:.6e} e_between={_E_b:.6e} nll_data_t={_NLL_d:.6e} q_data_fixed={_Q_fixed_d:.6e} objective_total={_obj_total_e:.6e} objective_valid={int(_objective_all_valid_e)} data_objective_valid={int(_objective_data_valid_e)} objective_kind={_objective_kind_e} mask_churn={_mask_churn_e:.6e} irls_pending={int(not _irls_metric_accepted_e)} n_active={int(_E_d['n'])} n_frozen={int(_NLL_n)}/{int(_NLL_expected)}", flush=True)
                        except Exception as _ee:
                            print(f'[ENERGY] failed: {_ee}', flush=True)
                _rd = int('0')
                if (it + 1) % max(1, int(self.iteration_log_interval)) == 0 or it + 1 == num_iters:
                    _lp_np_dbg = np.array(jax.device_get(linpoints))
                    (_H, _W) = (int(self.H), int(self.W))
                    _N = int(self.N)
                    _lp_grid_dbg = _lp_np_dbg[:_N].reshape(_H, _W, -1)
                    _depth_log_dbg = np.asarray(_lp_grid_dbg[..., -1], dtype=np.float64)
                    _dmin = float(np.nanmin(_depth_log_dbg))
                    _dmax = float(np.nanmax(_depth_log_dbg))
                    _dmean = float(np.nanmean(_depth_log_dbg))
                    _nonfinite = int(np.sum(~np.isfinite(_depth_log_dbg)))
                    _gt2d_dbg = np.asarray(depth_prev).squeeze()
                    _mask_pos = int(np.sum((_gt2d_dbg > 0.0) & np.isfinite(_gt2d_dbg)))
                    _large60 = int(np.sum(np.abs(_depth_log_dbg) > 60.0))
                    if self.verbose:
                        print(f'[DBG] depth_log stats: min={_dmin:.3e}, max={_dmax:.3e}, mean={_dmean:.3e}, nonfinite={_nonfinite}, |x|>60 cnt={_large60}, valid_gt_px={_mask_pos}')
                    est_twist = jnp.mean(linpoints[..., :6], axis=0)
                    linpoints_np_now = _lp_np_dbg
                    gt_depth2d_now = np.asarray(depth_prev).squeeze()
                    (scaled_depth_full, scale_now) = _compute_depth_and_scale(linpoints_np_now, gt_depth2d_now)
                    self._cur_depth_scale = scale_now
                    if self.verbose:
                        print(f'[DBG] depth_scale_now = {self._cur_depth_scale:.6e}')
                    if self.verbose or not _DIAG_FETCH:  # read only by the verbose print (PIXVOD_DIAG_FETCH skips the copies)
                        pf_z = np.asarray(jax.device_get(self.prior_facs.z))
                        pf_adj = np.asarray(jax.device_get(self.prior_facs.adj_var_id)).reshape(-1)
                        mask_leaf = pf_adj < _N
                        if np.any(mask_leaf) and pf_z.shape[1] >= 7:
                            vals = pf_z[mask_leaf, 6]
                            vals = np.exp(vals)
                            v = vals[np.isfinite(vals)]
                            if self.verbose:
                                print(f'[prior] iter={it + 1}: depth stats(min/mean/max) = {float(v.min()):.4e}/{float(v.mean()):.4e}/{float(v.max()):.4e}', flush=True)
                    est_twist_scaled = _scale_twist_translation(est_twist, self._cur_depth_scale)
                    (disp_main, _) = se3_geo_dist_explicit(est_twist_scaled, gt_twist_eval)
                    disp_main_np = np.array(jax.device_get(disp_main))
                    geo_dist_main = float(np.linalg.norm(disp_main_np))
                    denom_main = d0_main if d0_main > 1e-12 else 1e-12
                    norm_err_main = geo_dist_main / denom_main
                    (disp_cur, _) = se3_geo_dist_explicit(est_twist_scaled, gt_twist_cur)
                    disp_cur_np = np.array(jax.device_get(disp_cur))
                    geo_dist_cur = float(np.linalg.norm(disp_cur_np))
                    denom_cur = d0_cur if d0_cur > 1e-12 else 1e-12
                    norm_err_cur = geo_dist_cur / denom_cur
                    print(f'Iter {it + 1}:\n est_twist_scaled = {est_twist_scaled},\n GT(main) = {gt_twist_eval},\n GT(current) = {gt_twist_cur},\n geo(main) = {geo_dist_main:.6f}, geo(cur) = {geo_dist_cur:.6f}')
                    print(f' Norm geo err (vs main): {norm_err_main:.4f}, (vs current): {norm_err_cur:.4f}')
                    if _RERUN:
                        self._rerun.log_stage(self.global_iter_count + 1, stage_label, jax.device_get(est_twist_scaled), jax.device_get(gt_twist_eval), jax.device_get(gt_twist_cur),
                                              extrinsic_current, scaled_depth_full, self._cur_depth_scale, norm_err_main, norm_err_cur, rgb_cur)
                    _rot_cur = float(np.linalg.norm(disp_cur_np[:3]))
                    _tr_cur = float(np.linalg.norm(disp_cur_np[3:6]))
                    _rot_main = float(np.linalg.norm(disp_main_np[:3]))
                    _tr_main = float(np.linalg.norm(disp_main_np[3:6]))
                    print(f' geo split (cur): rot = {_rot_cur:.6f}, trans = {_tr_cur:.6f} | (main): rot = {_rot_main:.6f}, trans = {_tr_main:.6f}')
                    est_twist_trace.append(est_twist_scaled)
                    try:
                        leaf_twists_now = np.asarray(linpoints_np_now[:int(self.N), :6], dtype=np.float32)
                        est_std_now = np.std(leaf_twists_now, axis=0).astype(np.float32)
                        est_std_now[3:6] = est_std_now[3:6] * float(self._cur_depth_scale)
                    except Exception:
                        est_std_now = np.zeros((6,), dtype=np.float32)
                    try:
                        root_twist_now = jnp.array(linpoints[root_id, :6])
                        root_est_now_scaled = _scale_twist_translation(root_twist_now, self._cur_depth_scale)
                        (disp_root_cur, _) = se3_geo_dist_explicit(root_est_now_scaled, gt_twist_cur)
                        geo_root_cur = float(np.linalg.norm(np.array(jax.device_get(disp_root_cur))))
                        den_root_cur = d0_cur if d0_cur > 1e-12 else 1e-12
                        norm_err_root_cur = geo_root_cur / den_root_cur
                    except Exception:
                        root_est_now_scaled = est_twist_scaled
                        norm_err_root_cur = float('nan')
                    norm_geo_err_trace.append(norm_err_main)
                    is_sub_stage = stage_label is not None and ('sub' in stage_label and 'main' not in stage_label)
                    x_val = stage_iter_base + (it + 1)
                    self._append_combined_point_at(x_val, norm_err_main, norm_err_cur, est_twist_scaled, est_std_now, root_est_now_scaled, norm_err_root_cur, gt_twist_eval, gt_twist_cur, is_sub_stage)
                    _pe = int('1')
                    self._plot_ctr = getattr(self, '_plot_ctr', 0) + 1
                    if _PLOT_EVERY <= 1 or self._plot_ctr % _PLOT_EVERY == 0 or getattr(self, '_final_stage', False):
                        self._save_combined_plots()
                    else:
                        self._save_combined_csv()
                    ref_tag = f'ref{int(ref_main_idx):06d}' if ref_main_idx is not None else 'refXXXXXX'
                    stage_token = f'stage_{stage_label}' if stage_label else 'stage_sub'
                    overlay_name = f'overlay_{stage_token}_{ref_tag}_it_{it + 1:06d}.png'
                    overlay_path = self.logs_dir / overlay_name
                    depth_vis_full = scaled_depth_full
                    self._render_twist_overlay(it + 1, linpoints, intrinsic_prev, depth_prev.squeeze(), T_rel_eval, rgb_prev, overlay_path, est_depth_prev=depth_vis_full, est_trans_scale=float(getattr(self, '_cur_depth_scale', 1.0)))
                    self._render_stage_montage(overlay_path, rgb_prev, rgb_cur, depth_vis_full, depth_prev)
                self.global_iter_count += 1
                if self._check_global_iter_limit():
                    break
        except KeyboardInterrupt:
            print(f'[interrupt] Caught KeyboardInterrupt during stage={stage_label}. Saving partial logs...', flush=True)
            self.killer.stop = True
        finally:
            if not _entry_metrics and len(est_twist_trace) == 0:  # no logged iteration: the stage-entry scale, as without the gate
                (_, self._cur_depth_scale) = _compute_depth_and_scale(np.array(jax.device_get(_entry_lp)), np.asarray(depth_prev).squeeze())
            if _EXACT_READOUT:
                linpoints_final = jax.vmap(_mu_exact)(self.vars.belief)
                if _READOUT_GUARD and (_n_rg := _readout_guard_count(self.vars.belief)):
                    print(f'[READOUT-GUARD] stage={stage_label} at=end fallback={_n_rg}', flush=True)
            else:
                linpoints_final = jax.vmap(lambda g: g.mu())(self.vars.belief)
            _phase4_eval_stage = int('150')
            pose_mode = None
            depth_mode = 'keep'
            is_main_stage = stage_label is not None and 'main' in stage_label
            if do_prior_reinit_after:
                pose_mode = 'current'
                depth_mode = 'current'
            elif is_main_stage:
                pose_mode = 'zero'
                depth_mode = 'current'
                self._reset_pose_estimation_random(linpoints_final)
            if pose_mode is not None:
                self._update_prior_measurements_from_linpoints(linpoints_final, pose_mode=pose_mode, depth_mode=depth_mode)
            if pose_mode == 'current' and do_prior_reinit_after and ('0' == '1') and (self.prior_facs is not None):
                try:
                    _cq_p = float('1e-3')
                    _cq_d = float('3e-3')
                    _Lp = np.asarray(jax.device_get(self.vars.belief.Lam), dtype=np.float64)
                    _np_prior = int(self.prior_facs.z_Lam.shape[0])
                    _Lp = _Lp[:_np_prior]
                    _D = _Lp.shape[1]
                    _Qd = np.zeros((_D,), np.float64)
                    _Qd[:6] = _cq_p * _cq_p
                    if _D >= 7:
                        _Qd[6] = _cq_d * _cq_d
                    _Sig = np.linalg.inv(_Lp + 1e-08 * np.eye(_D)[None])
                    _Lc = np.linalg.inv(_Sig + np.diag(_Qd)[None])
                    self._carry_zlam = jnp.array(_Lc.astype(np.float32))
                    self.prior_facs.z_Lam = self._carry_zlam
                    self._carry_calls = getattr(self, '_carry_calls', 0) + 1
                    if self._carry_calls == 1 or self._carry_calls % 50 == 0:
                        _dg = np.diagonal(_Lc[:int(self.N)], axis1=1, axis2=2)
                        print(f'[CARRY] call {self._carry_calls}: leaf prec median rot {np.median(_dg[:, :3]):.2e} tr {np.median(_dg[:, 3:6]):.2e} dep {(np.median(_dg[:, 6]) if _D >= 7 else 0):.2e}', flush=True)
                except Exception as _ce:
                    print(f'[CARRY] failed: {_ce}', flush=True)
            _cc_adv = int(num_iters)
            if len(est_twist_trace) > 0:
                if not _DIAG_FETCH:  # never read
                    est_trace_np = np.array([jax.device_get(w) for w in est_twist_trace])
                try:
                    self._comb_iter_counter += int(_cc_adv)
                except Exception:
                    self._comb_iter_counter = int(self._comb_iter_counter) + int(_cc_adv)
        if return_linpoints:
            return linpoints_final
        return None

class DatasetPairRunner(TrajectoryPairRunner):
    """Run the same SE3D+depth pipeline on a pre-rendered dataset without Blender.

    Required config keys (top-level or nested under 'dataset'):
      - images_dir: folder containing frame%06d.jpg, depth%06d.png, normal%06d.png
      - use_dsine_normal (optional): if true, use dsine_normal%06d.png instead of GT normal
      - trajectory_path: text file, each line = 16 floats (row-major 4x4) for T_wc
      - intrinsic / intrinsic_matrix / K: 9 floats (row-major) or 3x3 list
      - start_frame: integer starting frame index (zero-based)
    """

    def __init__(self, blend_path, cfg, config_path=None):
        super().__init__(blend_path, cfg, config_path=config_path)
        self._covariant_execution_mode = 'dataset'
        self._parse_dataset_cfg()

    def _cfg_get(self, *names, default=None):
        for n in names:
            if n in self.cfg:
                return self.cfg[n]
        ds = self.cfg.get('dataset', {}) if isinstance(self.cfg.get('dataset', {}), dict) else {}
        for n in names:
            if n in ds:
                return ds[n]
        return default

    def _parse_dataset_cfg(self) -> None:
        img_dir = self._cfg_get('images_dir', 'images_root', 'images_path')
        traj_path = self._cfg_get('trajectory_path', 'traj_path', 'trajectory')
        intrinsic_any = self._cfg_get('intrinsic', 'intrinsic_matrix', 'K')
        start_f = self._cfg_get('start_frame', 'start', 'start_index', default=0)
        self.use_dsine_normal = bool(self._cfg_get('use_dsine_normal', 'dsine_normal', 'use_normal_dsine', default=False))
        if img_dir is None or traj_path is None or intrinsic_any is None:
            raise ValueError("Dataset mode requires 'images_dir', 'trajectory_path', and 'intrinsic' in config (top-level or under 'dataset').")
        self.ds_images_dir = Path(img_dir).resolve()
        self.ds_traj_path = Path(traj_path).resolve()
        self.ds_start_frame = int(start_f)
        K = None
        if isinstance(intrinsic_any, (list, tuple)):
            flat = []
            for x in intrinsic_any:
                if isinstance(x, (list, tuple)):
                    flat.extend(list(x))
                else:
                    flat.append(x)
            if len(flat) == 9:
                K = np.array(flat, dtype=np.float32).reshape(3, 3)
        if K is None:
            raise ValueError("Config 'intrinsic' must be 9 numbers (row-major) or 3x3 list.")
        self.ds_intrinsic = K.astype(np.float32)
        if not self.ds_traj_path.exists():
            raise FileNotFoundError(f'trajectory_path not found: {self.ds_traj_path}')
        mats = []
        with open(self.ds_traj_path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                arr = np.fromstring(s.replace(',', ' '), sep=' ', dtype=np.float64)
                if arr.size != 16:
                    continue
                Twc = arr.reshape(4, 4)
                try:
                    Ecv = np.linalg.inv(Twc)
                except Exception:
                    continue
                mats.append(Ecv.astype(np.float64))
        if len(mats) == 0:
            raise ValueError('No valid 4x4 T_wc found in trajectory file.')
        self.ds_extrinsics_cv = mats

    def _rgb_path(self, idx: int) -> Path:
        return (self.ds_images_dir / f'frame{idx:06d}.jpg').resolve()

    def _depth_path(self, idx: int) -> Path:
        return (self.ds_images_dir / f'depth{idx:06d}.png').resolve()

    def _normal_path(self, idx: int) -> Path:
        return (self.ds_images_dir / f'normal{idx:06d}.png').resolve()

    def _dsine_normal_path(self, idx: int) -> Path:
        return (self.ds_images_dir / f'dsine_normal{idx:06d}.png').resolve()

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'RGB not found or unreadable: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(rgb, 0.0, 1.0)

    @staticmethod
    def _load_depth_m(path: Path) -> np.ndarray:
        d16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d16 is None:
            raise FileNotFoundError(f'Depth not found or unreadable: {path}')
        if d16.dtype != np.uint16:
            raise ValueError(f'Depth must be 16-bit PNG: {path}')
        depth_m = d16.astype(np.float32) / 6553.5
        return depth_m[..., None]

    @staticmethod
    def _load_normal_cam(path: Path) -> np.ndarray:
        bgr16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr16 is None:
            raise FileNotFoundError(f'Normal not found or unreadable: {path}')
        if bgr16.dtype != np.uint16:
            raise ValueError(f'Normal must be 16-bit PNG: {path}')
        rgb16 = cv2.cvtColor(bgr16, cv2.COLOR_BGR2RGB).astype(np.float32)
        n = 2.0 * (rgb16 / 65535.0) - 1.0
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(norm, 1e-12)
        return n.astype(np.float32)

    @staticmethod
    def _load_normal_generic(path: Path) -> np.ndarray:
        """
        Load a normal map that may be stored as:
        - 16-bit PNG encoded in [0, 65535] -> map to [-1, 1]
        - 8-bit PNG encoded in [0, 255] -> map to [-1, 1]
        - Already float in [-1, 1] after OpenCV read (unlikely) -> clamp and renormalize
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Normal not found or unreadable: {path}')
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f'Normal must have 3 channels: {path}')
        if img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = img
        if rgb.dtype == np.uint16:
            n = 2.0 * (rgb.astype(np.float32) / 65535.0) - 1.0
        elif rgb.dtype == np.uint8:
            n = 2.0 * (rgb.astype(np.float32) / 255.0) - 1.0
        else:
            n = rgb.astype(np.float32)
            n = np.clip(n, -1.0, 1.0)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(norm, 1e-12)
        return n.astype(np.float32)

    def _load_frame_triplet(self, idx: int):
        rgb = self._load_rgb(self._rgb_path(idx))
        depth = self._load_depth_m(self._depth_path(idx))
        normal = None
        if getattr(self, 'use_dsine_normal', False):
            dsine_path = self._dsine_normal_path(idx)
            if dsine_path.exists():
                normal = self._load_normal_generic(dsine_path)
        if normal is None:
            normal = self._load_normal_cam(self._normal_path(idx))
        return (rgb, depth, normal)

    def _run_dataset_loop(self):
        reference = int(self.ds_start_frame)
        targets = int(self.cfg['target_frames'])
        per_target = int(self.cfg['iterations_per_target'])
        if targets <= 0 or per_target <= 0:
            raise ValueError('target_frames and iterations_per_target must be positive')
        final_target = reference + targets
        if reference < 0 or final_target >= len(self.ds_extrinsics_cv):
            raise IndexError('Requested target frames exceed trajectory length')
        (rgb_prev, depth_prev, normal_prev) = self._load_frame_triplet(reference)
        (self.H, self.W) = (int(rgb_prev.shape[0]), int(rgb_prev.shape[1]))
        self.N = self.H * self.W
        self.prev_intrinsic = np.asarray(self.ds_intrinsic, dtype=np.float32)
        (self.prev_depth, self.prev_normal) = (depth_prev, normal_prev)
        self._maybe_apply_normal_noise_and_visualize(stage_tag=f'ds_key_{reference:06d}')
        self.prev_normal = np.asarray(self.prev_normal, dtype=np.float32)
        if _RERUN:  # PIXVOD_RERUN: viewer setup with the reference frame and the GT trajectory of this run
            import rerun_live
            self._rerun = rerun_live.RerunLive(f'pixvod_{self.experiment_name}', self.ds_intrinsic, self.H, self.W, self.ds_extrinsics_cv[reference:final_target + 1],
                                               self.ds_extrinsics_cv[reference], rgb_prev, depth_prev, self.prev_normal)
        self._run_pre_warmup()
        Ecv0 = self.ds_extrinsics_cv[reference].astype(np.float64)
        Eeval = self.ds_extrinsics_cv[final_target].astype(np.float64)
        Ecur = None
        for stage in range(1, targets + 1):
            if getattr(self.killer, 'stop', False):
                break
            index = reference + stage
            rgb_current = self._load_rgb(self._rgb_path(index))
            Ecur = self.ds_extrinsics_cv[index].astype(np.float64)
            print(f' Stage {stage}/{targets}: frame {index}, {per_target} iterations (reference={reference})', flush=True)
            self._final_stage = stage == targets
            self._stage_optimize(per_target, rgb_prev, depth_prev, rgb_current, self.ds_intrinsic, Ecv0, Ecur, Eeval, do_prior_reinit_after=True, stage_label=f'sub{stage:02d}', ref_main_idx=reference)
        if Ecur is None:
            return
        linpoints = self._get_current_linpoints()
        est_final = jax.device_get(jnp.mean(linpoints[:self.N].reshape(self.H, self.W, -1)[self.cut:self.H - self.cut, self.cut:self.W - self.cut, :].reshape(-1, self.D_op), axis=0))
        scale = float(getattr(self, '_cur_depth_scale', 1.0))
        T_est = se3_Exp(jnp.array(est_final)[:6])
        T_est_scaled = T_est.at[:3, 3].set(T_est[:3, 3] * scale)
        self.traj_est_twist.append(jax.device_get(SE3_Log(T_est_scaled)))
        self.traj_gt_twist.append(jax.device_get(SE3_Log(jnp.array(Ecur @ np.linalg.inv(Ecv0)))))
        print(f'Finished {self.global_iter_count} target iterations; initial depth preparation is separate.', flush=True)

    def run_dataset(self) -> None:
        self._set_deterministic()
        self.killer = self.killer or type('K', (), {'stop': False})()
        try:
            self._run_dataset_loop()
        finally:
            try:  # PIXVOD_ASYNC_PLOT: the final figure wait must run even if a step below raises
                save_plot_twist(np.array(self.traj_gt_twist), np.array(self.traj_est_twist), self.logs_dir / 'trajectory_error_plot.png')
                if len(self.traj_gt_twist) > 0 and len(self.traj_est_twist) > 0:
                    gt_arr = np.asarray(self.traj_gt_twist)
                    est_arr = np.asarray(self.traj_est_twist)
                    num_pairs = min(gt_arr.shape[0], est_arr.shape[0])
                    gt_arr = gt_arr[:num_pairs]
                    est_arr = est_arr[:num_pairs]
                    norm_errs = []
                    for i in range(num_pairs):
                        est_i = jnp.array(est_arr[i])
                        gt_i = jnp.array(gt_arr[i])
                        (disp, _) = se3_geo_dist_explicit(est_i, gt_i)
                        (disp0, _) = se3_geo_dist_explicit(jnp.zeros_like(est_i), gt_i)
                        num = float(np.linalg.norm(np.array(jax.device_get(disp))))
                        den = float(np.linalg.norm(np.array(jax.device_get(disp0))))
                        den = den if den > 1e-12 else 1e-12
                        norm_errs.append(num / den)
                    pair_idx_col = np.arange(num_pairs).reshape(-1, 1)
                    norm_err_col = np.asarray(norm_errs, dtype=np.float32).reshape(-1, 1)
                    table = np.column_stack([pair_idx_col, norm_err_col, est_arr[:, :6], gt_arr[:, :6]])
                    header = 'pair_index,norm_geo_err,est_wx,est_wy,est_wz,est_tx,est_ty,est_tz,gt_wx,gt_wy,gt_wz,gt_tx,gt_ty,gt_tz'
                    out_csv = self.logs_dir / 'summary_pairs.csv'
                    np.savetxt(out_csv, table, delimiter=',', header=header, comments='')
                self._export_overlay_video(fps=12.0, out_filename='overlays_all_12fps.mp4')
            finally:
                self._close_trend_async()

def main(argv=None):
    parser = argparse.ArgumentParser(description='PixVOD: 150 target frames, 100 GBP iterations each (gates: gbp_engine/defaults.py).')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs' / 'replica.yaml', help='Parameter config (default: configs/replica.yaml beside this script)')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override a config key with KEY=VALUE; repeat for data paths or output paths.')
    args = parser.parse_args(argv)
    cfg_path = args.config.resolve()
    cfg = _apply_cli_overrides(dict(read_yaml(cfg_path)), args.overrides)
    # PIXVOD_ASYNC_PLOT: start the figure worker (fork+exec of a fresh interpreter) before the GPU is initialised
    ta = _trend_plots.AsyncTrendPlots() if _ASYNC_PLOT else None
    if _DONATE:
        import signal as _signal
        _signal.signal(_signal.SIGINT, _sigint_deferred)
    runner = DatasetPairRunner(Path.cwd(), cfg, config_path=cfg_path)
    runner._trend_async = ta
    runner.run_dataset()
if __name__ == '__main__':
    main()
