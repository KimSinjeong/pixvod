"""
Per-pixel camera SE(3) motion + depth estimation via GBP.

Module imported by run_pixvod.py: provides the geometry primitives
(unproject/project/se3_move/bilinear interpolation), the GBP factor/graph
construction, and the standalone runner base class.
"""
import os
import sys
import signal
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import random
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import jax
import jax.numpy as jnp
from functools import partial
from gbp_engine.utils import build_quadtree_edges
from gbp_engine.gbp import Gaussian, Variable, Factor, l2, update_factor_manifold_hJ, update_variable_twist_SE3, compute_ftov_msg
from gbp_engine.lie import se3_Exp, se3_geo_dist_explicit, _skew, se3_Jr
from gbp_engine.gbp import _DON  # PIXVOD_DONATE
_IPD = '1' == '1'
# PIXVOD_OOB_MSG: the stored photometric message of a row the photometric update does not write
# (projection outside the target image, or one of the 4 corners).  unset: keep the stored message (legacy).
# zero: store an exactly zero message instead.  rebase: keep it but re-express it at every change of the
# linearisation point (run_pixvod.py, update_variable_SE3D path only).
_OOB_MSG = os.environ.get('PIXVOD_OOB_MSG', '')
if _OOB_MSG not in ('', 'zero', 'rebase'):
    raise SystemExit(f'PIXVOD_OOB_MSG must be unset, zero or rebase; got {_OOB_MSG!r}')

# PIXVOD_PIXEL_CENTER: the reference ray of array pixel (i, j) is cast through (i + 0.5, j + 0.5) with
# cx = cy = 63.5 (the Replica/Pangolin pixel-corner convention), so projected coordinates are corner coordinates, while the
# bilinear sampler reads array index = coordinate.  0 (also when unset): no shift (at zero motion pixel x samples x + 0.5).
# 1: the sampler reads at projected - 0.5 (array index of the corner coordinate) in the photometric factor and in the
# robust-weight residual, which both sample through this function; rays, K, geometry gates and z are unchanged.
# The champion default sets 1 (gbp_engine/defaults.py); PIXVOD_BASELINE=legacy leaves it unset.  The robust-weight residual
# samples through this function on the host and in the device refresh (PIXVOD_DEV_REFRESH) alike.
_PIXEL_CENTER = os.environ.get('PIXVOD_PIXEL_CENTER', '0')
if _PIXEL_CENTER not in ('0', '1'):
    raise SystemExit(f'PIXVOD_PIXEL_CENTER must be 0 or 1; got {_PIXEL_CENTER!r}')
_PIXEL_CENTER = _PIXEL_CENTER == '1'
if _PIXEL_CENTER:
    print('[PIXEL-CENTER] photometric sampling at projected - 0.5 (corner -> array coordinates)', flush=True)

def photometric_array_xy(projected_xy):
    """Array coordinates at which the bilinear sampler reads a projected (corner) coordinate: shifted by -0.5 when PIXVOD_PIXEL_CENTER=1."""
    if _PIXEL_CENTER:
        return projected_xy - 0.5
    return projected_xy
_CUT = 2

def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

@partial(jax.jit, static_argnames=['jacobian'])
def remove_homo_from_point(p: jnp.ndarray, jacobian: bool=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    p: (4,) or (3,)
    returns: (3,) or (2,) and optionally d(p)/d p[:-1] as (3,4) or (2,3)
    """
    if jacobian:
        return (p[:-1] / p[-1], jnp.concatenate([jnp.eye(p.size - 1), -p[:-1, None] / p[-1]], axis=1) / p[-1])
    return p[:-1] / p[-1]

@partial(jax.jit, static_argnames=['jacobian'])
def se3_move_point(se3: jnp.ndarray, p: jnp.ndarray, jacobian: bool=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    se3: (6,), p: (4,)
    returns Tp_3d (3,) and optionally d(Tp_3d)/d se3 as (3,6) and d(Tp_3d)/d p as (3,4)
    """
    T = se3_Exp(se3)
    Tp = T @ p
    if jacobian:
        (Tp_3d, j_remove_homo) = remove_homo_from_point(Tp, jacobian=True)
        jac_bare = jnp.concatenate([jnp.concatenate([-T[:3, :3] @ _skew(p[:3] / p[3]), T[:3, :3]], axis=1), jnp.zeros((1, 6), dtype=Tp.dtype)], axis=0)
        jac = jac_bare @ se3_Jr(se3)
        return (Tp_3d, j_remove_homo @ jac, j_remove_homo @ T)
    else:
        return remove_homo_from_point(Tp, jacobian=False)

@partial(jax.jit, static_argnames=['jacobian'])
def homo2cart(p: jnp.ndarray, jacobian: bool=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    p: (3,) with last component != 0
    returns: (2,) and optionally (2,3) Jacobian
    """
    p_last_inv = 1.0 / p[-1]
    value = p[:-1] * p_last_inv
    if jacobian:
        jac = jnp.zeros((2, 3), dtype=p.dtype)
        jac = jac.at[0, 0].set(p_last_inv)
        jac = jac.at[1, 1].set(p_last_inv)
        p_last_inv_sqr = p_last_inv * p_last_inv
        jac = jac.at[0, 2].set(-p[0] * p_last_inv_sqr)
        jac = jac.at[1, 2].set(-p[1] * p_last_inv_sqr)
        return (value, jac)
    return value

@partial(jax.jit, static_argnames=['jacobian'])
def project_point(point_3d: jnp.ndarray, camera_matrix: jnp.ndarray, jacobian: bool=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    point_3d: (3,), camera_matrix: (3,3)
    returns pixel (2,)
    """
    if jacobian:
        (point_2d, jac_point_2d) = homo2cart(camera_matrix @ point_3d, jacobian=True)
        jac_projection = jac_point_2d @ camera_matrix
        return (point_2d, jac_projection)
    else:
        return homo2cart(camera_matrix @ point_3d, jacobian=False)

@partial(jax.jit, static_argnames=['jacobian'])
def unproject_point(d: jnp.ndarray, p: jnp.ndarray, K: jnp.ndarray, jacobian: bool=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    d: (), p: (2,), K: (3,3)
    returns: (4,) and optionally d(4)/d d as (4,)
    """
    p_px_homo = jnp.linalg.inv(K) @ jnp.concatenate([p, jnp.array([1.0])])
    p_3d_homo = jnp.concatenate([p_px_homo * d, jnp.array([1.0])])
    if jacobian:
        jac = jnp.zeros((4, 1), dtype=p_3d_homo.dtype)
        jac = jac.at[0, 0].set(p_px_homo[0])
        jac = jac.at[1, 0].set(p_px_homo[1])
        jac = jac.at[2, 0].set(1.0)
        return (p_3d_homo, jac)
    return p_3d_homo

def photometric_flow_inbounds(xi, p_i, K, height, width, ipd, rho_floor=1e-06):
    """Exact per-row geometry gate used by the photometric factor updater.

    Under IPD this intentionally uses the measurement model's existing
    ``max(rho, rho_floor)`` clamp.  ``rho_floor`` defaults to the historical
    1e-6 exactly; the covariant-gauge variant passes q*1e-6 explicitly so the
    physical validity boundary is invariant under rho'=q*rho.  Validity of the
    *unclamped* centre is a separate IRLS proposal-acceptance condition.
    """
    d_i = 1.0 / jnp.maximum(xi[6], rho_floor) if ipd else jnp.exp(xi[6])
    p_3d = unproject_point(d_i, p_i[:2], K, jacobian=False)
    p3 = se3_move_point(xi[:6], p_3d, jacobian=False)
    uv = project_point(p3, K, jacobian=False)
    u = uv[0]
    v = uv[1]
    Wf = jnp.array(width, dtype=p3.dtype)
    Hf = jnp.array(height, dtype=p3.dtype)
    return jnp.isfinite(u) & jnp.isfinite(v) & jnp.isfinite(p3[2]) & (p3[2] > 0.0) & (u >= 0.0) & (u < Wf) & (v >= 0.0) & (v < Hf)

@partial(jax.jit, static_argnames=['jacobian'])
def bilinear_interpolate_jax_xy(im: jnp.ndarray, xy: jnp.ndarray, jacobian: bool=False):
    """
    im: (H, W) or (H, W, C). If color, samples channel 0.
    xy: (2,)  (x, y)
    returns: scalar value or (value, jac) where jac is dI/d(x,y)
    """
    if im.ndim == 3:
        im0 = im[..., 0]
    elif im.ndim == 2:
        im0 = im
    else:
        raise ValueError('im must be (H, W) or (H, W, C)')
    (H, W) = im0.shape
    x = xy[0]
    y = xy[1]
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)
    Ia = im0[y0c, x0c]
    Ib = im0[y1c, x0c]
    Ic = im0[y0c, x1c]
    Id = im0[y1c, x1c]
    x_dtype = im0.dtype
    y_dtype = im0.dtype
    wa = (x1.astype(x_dtype) - x.astype(x_dtype)) * (y1.astype(y_dtype) - y.astype(y_dtype))
    wb = (x1.astype(x_dtype) - x.astype(x_dtype)) * (y.astype(y_dtype) - y0.astype(y_dtype))
    wc = (x.astype(x_dtype) - x0.astype(x_dtype)) * (y1.astype(y_dtype) - y.astype(y_dtype))
    wd = (x.astype(x_dtype) - x0.astype(x_dtype)) * (y.astype(y_dtype) - y0.astype(y_dtype))
    value = Ia * wa + Ib * wb + Ic * wc + Id * wd
    if not jacobian:
        return value
    else:
        fx = x.astype(x_dtype) - x0.astype(x_dtype)
        fy = y.astype(y_dtype) - y0.astype(y_dtype)
        dI_dx = (Ic - Ia) * (1.0 - fy) + (Id - Ib) * fy
        dI_dy = (Ib - Ia) * (1.0 - fx) + (Id - Ic) * fx
        jac = jnp.stack([dI_dx, dI_dy], axis=-1).astype(im0.dtype)
        return (value, jac)
_H_INTERP = 'bilinear'.strip().lower()

@partial(jax.jit, static_argnames=['jacobian'])
def bicubic_interpolate_jax_xy(im: jnp.ndarray, xy: jnp.ndarray, jacobian: bool=False):
    """Catmull-Rom bicubic. im: (H,W) or (H,W,C) (channel 0). xy: (2,) = (x, y).
    Returns value or (value, dI/d(x,y)). Border: clamped taps (matches bilinear's clip)."""
    im0 = im[..., 0] if im.ndim == 3 else im
    (H, W) = im0.shape
    (x, y) = (xy[0], xy[1])
    dt = im0.dtype
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    sx = x.astype(dt) - x0.astype(dt)
    sy = y.astype(dt) - y0.astype(dt)

    def _w(s):
        s2 = s * s
        s3 = s2 * s
        return jnp.stack([0.5 * (-s3 + 2.0 * s2 - s), 0.5 * (3.0 * s3 - 5.0 * s2 + 2.0), 0.5 * (-3.0 * s3 + 4.0 * s2 + s), 0.5 * (s3 - s2)])

    def _dw(s):
        s2 = s * s
        return jnp.stack([0.5 * (-3.0 * s2 + 4.0 * s - 1.0), 0.5 * (9.0 * s2 - 10.0 * s), 0.5 * (-9.0 * s2 + 8.0 * s + 1.0), 0.5 * (3.0 * s2 - 2.0 * s)])
    offs = jnp.arange(-1, 3, dtype=jnp.int32)
    xs = jnp.clip(x0 + offs, 0, W - 1)
    ys = jnp.clip(y0 + offs, 0, H - 1)
    patch = im0[ys[:, None], xs[None, :]]
    (wx, wy) = (_w(sx), _w(sy))
    value = wy @ patch @ wx
    if not jacobian:
        return value
    dI_dx = wy @ patch @ _dw(sx)
    dI_dy = _dw(sy) @ patch @ wx
    return (value, jnp.stack([dI_dx, dI_dy], axis=-1).astype(dt))

def h_J_fn_general(x, p, K, img, rho_floor=1e-06):
    """
    x: (7,), p: (2,), K: (3,3), img: (H,W,C)
    returns: (1,) and (7,) Jacobian
    """
    (twist, d) = (x[:6], 1.0 / jnp.maximum(x[6], rho_floor))
    (p_3d_homo, j_depth) = unproject_point(d, p, K, jacobian=True)
    (rays_est, j_move, j_p) = se3_move_point(twist, p_3d_homo, jacobian=True)
    (pixels_xy_est, j_proj) = project_point(rays_est, K, jacobian=True)
    _interp = bicubic_interpolate_jax_xy if _H_INTERP == 'bicubic' else mitchell_interpolate_jax_xy if _H_INTERP == 'mitchell' else bilinear_interpolate_jax_xy
    _sample_xy = photometric_array_xy(pixels_xy_est)
    (r, j_value) = _interp(img, _sample_xy, jacobian=True)
    J_common = j_value @ j_proj
    J_se3 = J_common @ j_move
    _dchain = -(d * d) * jnp.where(x[6] > rho_floor, 1.0, 0.0)
    J_depth = J_common @ j_p @ j_depth * _dchain
    return (r, jnp.concatenate([J_se3, J_depth], axis=0))

@partial(jax.jit, static_argnames=['jacobian'])
def mitchell_interpolate_jax_xy(im: jnp.ndarray, xy: jnp.ndarray, jacobian: bool=False):
    """Mitchell-Netravali (B=C=1/3 by default via H_CUBIC_B/C) separable cubic filter.
    Approximating (not interpolating) kernel: mild built-in blur keeps the photometric basin
    cushioned (bilinear-like tracking stability) with less resampling bias than bilinear."""
    B = float(0.3333333333333333)
    C = float(0.3333333333333333)
    (a3, a2, a0) = ((12 - 9 * B - 6 * C) / 6.0, (-18 + 12 * B + 6 * C) / 6.0, (6 - 2 * B) / 6.0)
    (b3, b2, b1, b0) = ((-B - 6 * C) / 6.0, (6 * B + 30 * C) / 6.0, (-12 * B - 48 * C) / 6.0, (8 * B + 24 * C) / 6.0)
    im0 = im[..., 0] if im.ndim == 3 else im
    (H, W) = im0.shape
    (x, y) = (xy[0], xy[1])
    dt = im0.dtype
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    sx = x.astype(dt) - x0.astype(dt)
    sy = y.astype(dt) - y0.astype(dt)

    def _k(t):
        inner = (a3 * t + a2) * t * t + a0
        outer = ((b3 * t + b2) * t + b1) * t + b0
        return jnp.where(t < 1.0, inner, jnp.where(t < 2.0, outer, 0.0))

    def _dk(t):
        inner = (3.0 * a3 * t + 2.0 * a2) * t
        outer = (3.0 * b3 * t + 2.0 * b2) * t + b1
        return jnp.where(t < 1.0, inner, jnp.where(t < 2.0, outer, 0.0))

    def _w(s):
        return jnp.stack([_k(1.0 + s), _k(s), _k(1.0 - s), _k(2.0 - s)])

    def _dw(s):
        return jnp.stack([_dk(1.0 + s), _dk(s), -_dk(1.0 - s), -_dk(2.0 - s)])
    offs = jnp.arange(-1, 3, dtype=jnp.int32)
    xs = jnp.clip(x0 + offs, 0, W - 1)
    ys = jnp.clip(y0 + offs, 0, H - 1)
    patch = im0[ys[:, None], xs[None, :]]
    (wx, wy) = (_w(sx), _w(sy))
    value = wy @ patch @ wx
    if not jacobian:
        return value
    dI_dx = wy @ patch @ _dw(sx)
    dI_dy = _dw(sy) @ patch @ wx
    return (value, jnp.stack([dI_dx, dI_dy], axis=-1).astype(dt))

def h_J_fn_normal_ipd(x, nz, rho_floor=1e-06):
    """IPD (inverse-depth parameterization) variant of h_J_fn_normal: the three depth variables are inverse
    depths rho_i; with log d_i = -log rho_i the SAME exact perspective-normal PDE
    discretization becomes h in log(rho) differences (measurement z unchanged):
        h = [ nz*(log rho_0 - log rho_1),  -nz*(log rho_0 - log rho_2) ]
    Nonlinear in rho (J depends on the linpoint) — relinearized like any factor;
    the baseline log-depth form is exactly linear (constant J), which is the honest
    cost of this parameterization. rho is clamped at
    ``rho_floor`` (historical default 1e-6; q*1e-6 in the covariant variant)."""
    r0 = jnp.maximum(x[0], rho_floor)
    r1 = jnp.maximum(x[1], rho_floor)
    r2 = jnp.maximum(x[2], rho_floor)
    h1 = nz[0] * (jnp.log(r0) - jnp.log(r1))
    h2 = -(nz[0] * (jnp.log(r0) - jnp.log(r2)))
    h = jnp.stack([h1, h2]).reshape(-1)
    g0 = jnp.where(x[0] > rho_floor, 1.0, 0.0)
    g1 = jnp.where(x[1] > rho_floor, 1.0, 0.0)
    g2 = jnp.where(x[2] > rho_floor, 1.0, 0.0)
    J = jnp.array([[nz[0] / r0 * g0, -nz[0] / r1 * g1, 0.0], [-nz[0] / r0 * g0, 0.0, nz[0] / r2 * g2]])
    return (h, J)

def h2_fn(x):
    return se3_geo_dist_explicit(x[0], x[1])

def h3_fn(x):
    return (x, jnp.eye(x.size, dtype=x.dtype))

def build_twist_graph(H, W, D_feat, D_op=6):
    N = H * W
    Ni_v = 7 + 1
    (M_edges, edge_ids_qt, edge_vars_qt, edge_pos_qt) = build_quadtree_edges(H, assign_range=1, backend='numpy')
    N_total = int(edge_vars_qt.max()) + 1
    var_ids = np.arange(0, N_total, dtype=np.int32)
    belief = Gaussian(np.zeros((N_total, D_op), dtype=np.float32), np.repeat(np.eye(D_op, dtype=np.float32)[None, :, :], N_total, axis=0))
    means = np.random.randn(N_total, D_op)
    belief.eta = 0.0001 * np.pi / 180.0 * means / np.linalg.norm(means, axis=-1, keepdims=True)
    msgs = Gaussian(np.zeros((N_total, Ni_v, D_op)), np.zeros((N_total, Ni_v, D_op, D_op)))
    adj_factor_idx = -np.ones((N_total, Ni_v), dtype=np.int32)
    vars = Variable(var_ids, belief, msgs, adj_factor_idx)
    z_Lam = np.eye(D_feat)
    factor_ids = np.arange(0, (H - 2 * _CUT) * (W - 2 * _CUT), dtype=np.int32)
    z_Lams = np.tile(z_Lam * 0.2, ((H - 2 * _CUT) * (W - 2 * _CUT), 1, 1))
    threshold = np.ones((H - 2 * _CUT) * (W - 2 * _CUT)).reshape(-1, 1)
    potential = None
    zs = np.zeros(((H - 2 * _CUT) * (W - 2 * _CUT), D_feat))
    adj_var_id = np.arange(0, N, dtype=np.int32).reshape(H, W)[_CUT:H - _CUT, _CUT:W - _CUT].reshape(-1, 1)
    adj_var_idx = np.zeros((H - 2 * _CUT) * (W - 2 * _CUT), dtype=np.int32).reshape(-1, 1)
    dataterm_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
    bias_factor_id = factor_ids.size
    factor_ids = edge_ids_qt + bias_factor_id
    adj_var_id = edge_vars_qt.astype(np.int32)
    adj_var_idx = edge_pos_qt.astype(np.int32)
    zs = np.zeros((factor_ids.size, D_op), dtype=np.float32)
    z_Lams = np.tile(np.eye(D_op, dtype=np.float32), (factor_ids.size, 1, 1))
    threshold = np.ones((factor_ids.size, 1), dtype=np.float32)
    potential = None
    between_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
    vars.adj_factor_idx[between_facs.adj_var_id, between_facs.adj_var_idx] = between_facs.factor_id.reshape(-1, 1)
    bias_factor_id += factor_ids.size
    prior_factor_ids = np.arange(bias_factor_id, bias_factor_id + N_total, dtype=np.int32)
    prior_adj_var_id = np.arange(0, N_total, dtype=np.int32).reshape(-1, 1)
    prior_adj_var_idx = np.full((N_total, 1), 6, dtype=np.int32)
    prior_zs = np.zeros((N_total, D_op), dtype=np.float32)
    prior_z_Lams = np.tile(np.eye(D_op, dtype=np.float32), (N_total, 1, 1))
    prior_threshold = np.ones((N_total, 1), dtype=np.float32)
    prior_facs = Factor(prior_factor_ids, prior_zs, prior_z_Lams, prior_threshold, None, prior_adj_var_id, prior_adj_var_idx)
    vars.adj_factor_idx[prior_facs.adj_var_id, prior_facs.adj_var_idx] = prior_facs.factor_id.reshape(-1, 1)
    bias_factor_id += prior_factor_ids.size
    return (vars, dataterm_facs, between_facs, prior_facs)

def build_se3d_graph(H, W, D_feat, N_xyz, D_op=7, z_prior=1.0, intrinsic=None, graph_type: str='quadtree'):
    N = H * W
    graph_type = str(graph_type).strip().lower() if graph_type is not None else 'quadtree'
    if graph_type not in ('quadtree', 'planar'):
        raise ValueError(f"Unsupported graph_type={graph_type}. Use 'quadtree' or 'planar'.")
    if graph_type == 'quadtree':
        Ni_v = 13 + 1
        (M_edges, edge_ids, edge_vars, edge_pos) = build_quadtree_edges(H, assign_range=1, backend='numpy')
        edge_vars = edge_vars.astype(np.int32)
        edge_pos = edge_pos.astype(np.int32)
        edge_ids = edge_ids.astype(np.int32)
        N_total = int(edge_vars.max()) + 1
    else:
        PORT_SMOOTH_RIGHT = 14
        PORT_SMOOTH_LEFT = 15
        PORT_SMOOTH_DOWN = 16
        PORT_SMOOTH_UP = 17
        PORT_INVALID = 18
        Ni_v = PORT_INVALID + 1
        N_total = int(N)
        if W > 1:
            (ii_h, jj_h) = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W - 1, dtype=np.int32), indexing='ij')
            v0_h = (ii_h * W + jj_h).reshape(-1)
            v1_h = (v0_h + 1).reshape(-1)
            adj_h = np.stack([v0_h, v1_h], axis=1)
            idx_h = np.stack([np.full_like(v0_h, PORT_SMOOTH_RIGHT, dtype=np.int32), np.full_like(v1_h, PORT_SMOOTH_LEFT, dtype=np.int32)], axis=1)
        else:
            adj_h = np.zeros((0, 2), dtype=np.int32)
            idx_h = np.zeros((0, 2), dtype=np.int32)
        if H > 1:
            (ii_v, jj_v) = np.meshgrid(np.arange(H - 1, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
            v0_v = (ii_v * W + jj_v).reshape(-1)
            v1_v = (v0_v + W).reshape(-1)
            adj_v = np.stack([v0_v, v1_v], axis=1)
            idx_v = np.stack([np.full_like(v0_v, PORT_SMOOTH_DOWN, dtype=np.int32), np.full_like(v1_v, PORT_SMOOTH_UP, dtype=np.int32)], axis=1)
        else:
            adj_v = np.zeros((0, 2), dtype=np.int32)
            idx_v = np.zeros((0, 2), dtype=np.int32)
        edge_vars = np.concatenate([adj_h, adj_v], axis=0).astype(np.int32)
        edge_pos = np.concatenate([idx_h, idx_v], axis=0).astype(np.int32)
        M_edges = int(edge_vars.shape[0])
        edge_ids = np.arange(M_edges, dtype=np.int32)
    var_ids = np.arange(0, N_total, dtype=np.int32)
    belief = Gaussian(np.zeros((N_total, D_op), dtype=np.float32), np.repeat(np.eye(D_op, dtype=np.float32)[None, :, :], N_total, axis=0))
    belief.eta[:, -1] = np.log(z_prior)
    means = np.random.randn(N_total, D_op)
    belief.eta = 0.0001 * np.pi / 180.0 * means / np.linalg.norm(means, axis=-1, keepdims=True)
    belief.eta[:, -1] += np.float32(1.0 / z_prior)
    msgs = Gaussian(np.zeros((N_total, Ni_v, D_op)), np.zeros((N_total, Ni_v, D_op, D_op)))
    adj_factor_idx = -np.ones((N_total, Ni_v), dtype=np.int32)
    vars = Variable(var_ids, belief, msgs, adj_factor_idx)
    z_Lam = np.eye(D_feat)
    factor_ids = np.arange(0, (H - 2 * _CUT) * (W - 2 * _CUT), dtype=np.int32)
    z_Lams = np.tile(z_Lam * 0.2, ((H - 2 * _CUT) * (W - 2 * _CUT), 1, 1))
    threshold = np.ones((H - 2 * _CUT) * (W - 2 * _CUT)).reshape(-1, 1)
    potential = None
    zs = np.zeros(((H - 2 * _CUT) * (W - 2 * _CUT), D_feat))
    adj_var_id = np.arange(0, N, dtype=np.int32).reshape(H, W)[_CUT:H - _CUT, _CUT:W - _CUT].reshape(-1, 1)
    adj_var_idx = np.zeros((H - 2 * _CUT) * (W - 2 * _CUT), dtype=np.int32).reshape(-1, 1)
    dataterm_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
    bias_factor_id = factor_ids.size
    factor_ids = edge_ids + bias_factor_id
    adj_var_id = edge_vars.astype(np.int32)
    adj_var_idx = edge_pos.astype(np.int32)
    zs = np.zeros((factor_ids.size, D_op), dtype=np.float32)
    z_Lams = np.tile(np.eye(D_op, dtype=np.float32), (factor_ids.size, 1, 1))
    threshold = np.ones((factor_ids.size, 1), dtype=np.float32)
    potential = None
    between_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
    vars.adj_factor_idx[between_facs.adj_var_id, between_facs.adj_var_idx] = between_facs.factor_id.reshape(-1, 1)
    bias_factor_id += factor_ids.size
    prior_factor_ids = np.arange(bias_factor_id, bias_factor_id + N_total, dtype=np.int32)
    prior_adj_var_id = np.arange(0, N_total, dtype=np.int32).reshape(-1, 1)
    prior_adj_var_idx = np.full((N_total, 1), 6, dtype=np.int32)
    prior_zs = np.zeros((N_total, D_op), dtype=np.float32)
    prior_zs[:, -1] = 1.0 / z_prior
    prior_z_Lams = np.tile(np.eye(D_op, dtype=np.float32), (N_total, 1, 1))
    prior_threshold = np.ones((N_total, 1), dtype=np.float32)
    prior_facs = Factor(prior_factor_ids, prior_zs, prior_z_Lams, prior_threshold, None, prior_adj_var_id, prior_adj_var_idx)
    vars.adj_factor_idx[prior_facs.adj_var_id, prior_facs.adj_var_idx] = prior_facs.factor_id.reshape(-1, 1)
    bias_factor_id += prior_factor_ids.size

    def cross(i, j, H, W, B, N_xyz):
        factor_id = jnp.array([B + 2 * (i * W + j), B + 2 * (i * W + j) + 1])
        z_xyz = N_xyz[i, j]
        z = jnp.array([[z_xyz[0] / intrinsic[0, 0], z_xyz[0] / intrinsic[0, 0]], [z_xyz[1] / intrinsic[1, 1], z_xyz[1] / intrinsic[1, 1]]])
        v0_id = jnp.array([i * W + j, i * W + j])
        v1_id = jnp.array([i * W + j - 1, (i - 1) * W + j])
        v2_id = jnp.array([i * W + j + 1, (i + 1) * W + j])
        idx_0s = jnp.array([7, 10], dtype=np.int32)
        idx_1s = jnp.array([8, 11], dtype=np.int32)
        idx_2s = jnp.array([9, 12], dtype=np.int32)
        z_Lams = jnp.array([jnp.eye(2), jnp.eye(2)])
        return (factor_id, v0_id, v1_id, v2_id, idx_0s, idx_1s, idx_2s, z, z_Lams)
    (ii, jj) = np.meshgrid(np.arange(1, H - 1), np.arange(1, W - 1), indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()
    (factor_ids, v0_ids, v1_ids, v2_ids, idx_0s, idx_1s, idx_2s, zs, z_Lams) = jax.vmap(cross, in_axes=(0, 0, None, None, None, None))(ii, jj, H, W, bias_factor_id, jnp.asarray(N_xyz))
    factor_ids = factor_ids.reshape(-1)
    adj_var_id = np.stack([v0_ids.reshape(-1), v1_ids.reshape(-1), v2_ids.reshape(-1)], axis=1)
    adj_var_idx = np.stack([idx_0s.reshape(-1), idx_1s.reshape(-1), idx_2s.reshape(-1)], axis=1)
    z_Lams = z_Lams.reshape(-1, 2, 2)
    zs = zs.reshape(-1, 2)
    threshold = np.ones((factor_ids.size, 1))
    potential = None
    normal_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
    vars.adj_factor_idx[normal_facs.adj_var_id, normal_facs.adj_var_idx] = normal_facs.factor_id.reshape(-1, 1)
    bias_factor_id += factor_ids.size
    return (vars, dataterm_facs, between_facs, prior_facs, normal_facs)

def save_plot_twist(gt_twist, est_twist, out_path: Path):
    """
    Save per-axis twist trajectories (rotation+translation). Handles the single-sample case by
    plotting markers and setting explicit x-limits so something is visible
    even before multiple pairs are processed.
    Accepts (N,6) arrays for GT and estimate. If (N,3) is given, only rotation is plotted.
    """
    if gt_twist is None or est_twist is None:
        return
    if gt_twist.shape[0] == 0 or est_twist.shape[0] == 0:
        return
    n = int(min(gt_twist.shape[0], est_twist.shape[0]))
    gt_twist = np.asarray(gt_twist)[:n]
    est_twist = np.asarray(est_twist)[:n]
    x = np.arange(n)
    dim = gt_twist.shape[1] if gt_twist.ndim == 2 else 3
    dim = 6 if dim >= 6 else 3
    if dim == 6:
        (fig, ax) = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
        ax = np.array(ax)
        rot_labels = ['wx', 'wy', 'wz']
        trans_labels = ['tx', 'ty', 'tz']
        for i in range(3):
            a = ax[0, i]
            a.plot(x, gt_twist[:, i], color='g', linestyle='-', marker='o', label=f'GT {rot_labels[i]}')
            a.plot(x, est_twist[:, i], color='r', linestyle='--', marker='x', label=f'Est {rot_labels[i]}')
            if n == 1:
                a.set_xlim(-0.5, 0.5)
            a.set_ylabel(rot_labels[i])
            a.legend()
        for i in range(3):
            a = ax[1, i]
            a.plot(x, gt_twist[:, 3 + i], color='g', linestyle='-', marker='o', label=f'GT {trans_labels[i]}')
            a.plot(x, est_twist[:, 3 + i], color='r', linestyle='--', marker='x', label=f'Est {trans_labels[i]}')
            if n == 1:
                a.set_xlim(-0.5, 0.5)
            a.set_ylabel(trans_labels[i])
            a.legend()
        ax[-1, 0].set_xlabel('Frame Pair')
        ax[-1, 1].set_xlabel('Frame Pair')
        ax[-1, 2].set_xlabel('Frame Pair')
    else:
        (fig, ax) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['wx', 'wy', 'wz']
        for i in range(3):
            ax[i].plot(x, gt_twist[:, i], color='g', linestyle='-', marker='o', label=f'GT {labels[i]}')
            ax[i].plot(x, est_twist[:, i], color='r', linestyle='--', marker='x', label=f'Est {labels[i]}')
            if n == 1:
                ax[i].set_xlim(-0.5, 0.5)
            ax[i].set_ylabel(labels[i])
            ax[i].legend()
        ax[-1].set_xlabel('Frame Pair')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_plot_twist_iter(gt_twist_const, est_twist_trace, iter_step: int, out_path: Path, alt_out_path: Path | None=None):
    """
    Save per-iteration twist traces (x: iterations, y: wx/wy/wz and tx/ty/tz).
    gt_twist_const: shape (6,) or (T,6) constant GT over iterations
    est_twist_trace: shape (T,6) estimated values sampled every iter_step iterations
    iter_step: sampling interval in iterations
    out_path: primary output path (e.g., rotation_error.png)
    alt_out_path: optional secondary path to also save (e.g., per-pair file)
    """
    if est_twist_trace is None or len(est_twist_trace) == 0:
        return
    est_twist_trace = np.asarray(est_twist_trace)
    T = est_twist_trace.shape[0]
    if gt_twist_const is None:
        gt = np.zeros_like(est_twist_trace)
    else:
        gt_twist_const = np.asarray(gt_twist_const)
        if gt_twist_const.ndim == 1:
            gt = np.tile(gt_twist_const.reshape(1, -1), (T, 1))
        else:
            gt = gt_twist_const[:T]
    x = (np.arange(T) + 1) * int(iter_step)
    dim = est_twist_trace.shape[1] if est_twist_trace.ndim == 2 else 3
    dim = 6 if dim >= 6 else 3
    if dim == 6:
        (fig, ax) = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
        ax = np.array(ax)
        rot_labels = ['wx', 'wy', 'wz']
        trans_labels = ['tx', 'ty', 'tz']
        for i in range(3):
            a = ax[0, i]
            a.plot(x, gt[:, i], color='g', linestyle='-', label=f'GT {rot_labels[i]}')
            a.plot(x, est_twist_trace[:, i], color='r', linestyle='--', label=f'Est {rot_labels[i]}')
            a.set_ylabel(rot_labels[i])
            a.legend()
        for i in range(3):
            a = ax[1, i]
            a.plot(x, gt[:, 3 + i], color='g', linestyle='-', label=f'GT {trans_labels[i]}')
            a.plot(x, est_twist_trace[:, 3 + i], color='r', linestyle='--', label=f'Est {trans_labels[i]}')
            a.set_ylabel(trans_labels[i])
            a.legend()
        ax[-1, 0].set_xlabel('Iteration')
        ax[-1, 1].set_xlabel('Iteration')
        ax[-1, 2].set_xlabel('Iteration')
    else:
        (fig, ax) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['wx', 'wy', 'wz']
        for i in range(3):
            ax[i].plot(x, gt[:, i], color='g', linestyle='-', label=f'GT {labels[i]}')
            ax[i].plot(x, est_twist_trace[:, i], color='r', linestyle='--', label=f'Est {labels[i]}')
            ax[i].set_ylabel(labels[i])
            ax[i].legend()
        ax[-1].set_xlabel('Iteration')
    fig.tight_layout()
    fig.savefig(out_path)
    if alt_out_path is not None:
        fig.savefig(alt_out_path)
    plt.close(fig)

def factor_update_flow_hJ(factor, xs, p, K, img, h_fn, w, rho_floor=1e-06, dynamic_rho_floor=False):
    if dynamic_rho_floor:
        (h, J) = h_fn(xs.reshape(-1), p, K, img, rho_floor)
    else:
        (h, J) = h_fn(xs.reshape(-1), p, K, img)
    if J.ndim == 1:
        J = J[jnp.newaxis, :]
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h.reshape(-1)
    s = w(r.T @ z_Lam @ r, factor.threshold)
    Lam = s * J.T @ z_Lam @ J
    eta = s * J.T @ z_Lam @ r
    return Gaussian(eta, Lam)

@partial(jax.jit, static_argnames=['f', 'w', 'dynamic_rho_floor'], donate_argnames=_DON)
def update_factor_flow_hJ(facs, vars, vtof_msgs, linpoints, p, K, img, f, w, mask_ftov=None, rho_floor=1e-06, dynamic_rho_floor=False):
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    facs.potential = jax.vmap(factor_update_flow_hJ, in_axes=(0, 0, 0, None, None, None, None, None, None))(facs, linpoints_reordered, p, K, img, f, w, rho_floor, dynamic_rho_floor)
    try:
        H = img.shape[0]
        W = img.shape[1]
        xs_cur = linpoints_reordered[:, 0]

        def _inbounds(xi, p_i):
            if dynamic_rho_floor:
                return photometric_flow_inbounds(xi, p_i, K, H, W, _IPD, rho_floor)
            return photometric_flow_inbounds(xi, p_i, K, H, W, _IPD)
        mask_inbounds = jax.vmap(_inbounds, in_axes=(0, 0))(xs_cur, p)
        mask_inbounds = mask_inbounds.reshape(-1, 1)
    except Exception:
        mask_inbounds = jnp.ones((facs.factor_id.shape[0], 1), dtype=bool)
    ftov_msgs = jax.vmap(compute_ftov_msg)(facs, vtof_msgs_reordered)
    if mask_ftov is not None:
        mask_to_use = jnp.logical_and(mask_inbounds, mask_ftov.reshape(-1, 1))
    else:
        mask_to_use = mask_inbounds
    target_eta = vars.msgs.eta[facs.adj_var_id, facs.adj_var_idx]
    target_Lam = vars.msgs.Lam[facs.adj_var_id, facs.adj_var_idx]
    if _OOB_MSG == 'zero':
        target_eta = jnp.zeros_like(target_eta)
        target_Lam = jnp.zeros_like(target_Lam)
    mask_eta = mask_to_use[..., None]
    mask_Lam = mask_to_use[..., None, None]
    blended_eta = jnp.where(mask_eta, ftov_msgs.eta, target_eta)
    blended_Lam = jnp.where(mask_Lam, ftov_msgs.Lam, target_Lam)
    vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(blended_eta)
    vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(blended_Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)

def _depth_block_indices_se3d(num_vars: int=3, d_op: int=7, depth_idx_in_block: int=6):
    base = np.arange(num_vars, dtype=np.int32) * d_op
    return jnp.array(base + depth_idx_in_block, dtype=jnp.int32)

def factor_update_normal(factor, xs_depth, nz, h_J_fn, w, rho_floor=1e-06, dynamic_rho_floor=False):
    if dynamic_rho_floor:
        (h, J) = h_J_fn(xs_depth.reshape(-1), nz, rho_floor)
    else:
        (h, J) = h_J_fn(xs_depth.reshape(-1), nz)
    if J.ndim == 1:
        J = J[jnp.newaxis, :]
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h.reshape(-1)
    s = w(r.T @ z_Lam @ r, factor.threshold)
    Lam_small = s * J.T @ z_Lam @ J
    eta_small = s * J.T @ z_Lam @ r
    dim_full = 21
    eta_full = jnp.zeros((dim_full,), dtype=eta_small.dtype)
    Lam_full = jnp.zeros((dim_full, dim_full), dtype=Lam_small.dtype)
    depth_idx = _depth_block_indices_se3d()
    eta_full = eta_full.at[depth_idx].set(eta_small)
    Lam_full = Lam_full.at[depth_idx[:, None], depth_idx[None, :]].set(Lam_small)
    return Gaussian(eta_full, Lam_full)

@partial(jax.jit, static_argnames=['f', 'w', 'dynamic_rho_floor'])
def update_factor_normal(facs, vars, vtof_msgs, linpoints, nz, f, w, rho_floor=1e-06, dynamic_rho_floor=False):
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    xs_depth = linpoints_reordered[..., 6]
    facs.potential = jax.vmap(factor_update_normal, in_axes=(0, 0, 0, None, None, None, None))(facs, xs_depth, jnp.stack([nz, nz], axis=1).reshape(-1, 1), f, w, rho_floor, dynamic_rho_floor)
    ftov_msgs = jax.vmap(compute_ftov_msg)(facs, vtof_msgs_reordered)
    vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.eta)
    vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)

class GracefulKiller:

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        self.stop = True

class RotationStandaloneRunner:

    def __init__(self, blend_path: Path, cfg: Dict[str, Any], config_path: Path | None=None):
        self.blend_path = blend_path
        self.cfg = cfg
        self.config_path = config_path
        self.experiment_name: str = cfg.get('experiment_name', 'experiment')
        self.out_root = Path(cfg.get('output_root', REPO_ROOT / 'experiments')) / self.experiment_name
        (self.frames_dir, self.data_dir, self.logs_dir) = (self.out_root / 'frames', self.out_root / 'data', self.out_root / 'logs')
        ensure_dir(self.frames_dir)
        ensure_dir(self.data_dir)
        ensure_dir(self.logs_dir)
        self.num_frames: int = int(cfg.get('num_frames', 100))
        self.iterations_per_pair: int = int(cfg.get('iterations_per_pair', 1000))
        self.iteration_log_interval: int = int(cfg.get('iteration_log_interval', 100))
        self.resolution_x: int = int(cfg.get('resolution_x', 256))
        self.resolution_y: int = int(cfg.get('resolution_y', 256))
        try:
            fov_val = cfg.get('fov_deg', None)
            self.fov_deg = None if fov_val is None else float(fov_val)
        except Exception:
            self.fov_deg = None
        try:
            self.fov_fit = str(cfg.get('fov_fit', 'auto')).lower()
        except Exception:
            self.fov_fit = 'auto'
        self.retract_alpha: float = float(cfg.get('alpha', 1.0))
        global _CUT
        try:
            default_cut = int(_CUT)
        except Exception:
            default_cut = 2
        try:
            self.cut = int(cfg.get('cut', cfg.get('border_cut', cfg.get('image_border_cut', default_cut))))
        except Exception:
            self.cut = int(default_cut)
        try:
            _CUT = int(self.cut)
        except Exception:
            pass
        self.hp_schedule: List[Dict[str, Any]] = self._parse_hp_schedule()
        self.hp_schedule_idx: int = -1
        self.dataterm_s = float(cfg.get('dataterm_s', 0.05))
        self.smoothness_s_rot = float(cfg.get('smoothness_s_rot', 0.001))
        self.smoothness_s_tr = float(cfg.get('smoothness_s_tr', 0.001))
        self.prior_s = float(cfg.get('prior_s', 0.02))
        self.prior_boundary_scale: float = float(cfg.get('prior_boundary_scale', 0.5))
        try:
            self.prior_boundary_s: float = float(cfg.get('prior_boundary_s', self.prior_s * self.prior_boundary_scale))
        except Exception:
            self.prior_boundary_s = self.prior_s * self.prior_boundary_scale
        self.use_robust = bool(cfg.get('use_robust', False))
        self.robust_thres = float(cfg.get('robust_thres', 0.5))
        self.camera = None
        (self.traj_gt_twist, self.traj_est_twist) = ([], [])
        self.vars = self.dataterm_facs = self.between_facs = self.grids = None
        (self.H, self.W, self.N, self.D, self.D_op) = (self.resolution_y, self.resolution_x, self.resolution_y * self.resolution_x, 1, 6)
        self.killer = None
        self.global_iter_count = 0
        self.photometric_stride: int = int(cfg.get('photometric_stride', 1))
        self.photo_warmup: int = int(cfg.get('photo_warmup', 50))
        self.disable_photometric: bool = bool(cfg.get('disable_photometric', False))
        self.viz_stride: int = int(cfg.get('viz_stride', 16))
        self.num_iter_prior_reinit: int = int(cfg.get('iterations_per_target', 100))
        self._did_prior_reinit: bool = False
        self.trajectory_type: int = int(cfg.get('trajectory_type', 0))
        self._traj_y_start: float = float(cfg.get('traj_y_start', 4.0))
        self._traj_y_end: float = float(cfg.get('traj_y_end', -4.0))
        self._traj_x_amplitude: float = float(cfg.get('traj_x_amplitude', 4.0))
        self._traj_z_center: float = float(cfg.get('traj_z_center', 4.5))
        self._traj_z_amplitude: float = float(cfg.get('traj_z_amplitude', 1.0))
        self._traj_num_segments: int = int(cfg.get('traj_num_segments', 4))
        self._rot_x_amplitude: float = float(cfg.get('rot_x_amplitude', 0.2))
        self._rot_y_amplitude: float = float(cfg.get('rot_y_amplitude', 0.2))
        self._rot_z_amplitude: float = float(cfg.get('rot_z_amplitude', 0.2))
        prior_axis = cfg.get('prior_init_axis_angle', [0.0, 0.0, 0.0])
        if isinstance(prior_axis, (list, tuple)) and len(prior_axis) == self.D_op:
            self.prior_init_axis = jnp.array(prior_axis, dtype=jnp.float32)
        else:
            self.prior_init_axis = jnp.zeros((self.D_op,), dtype=jnp.float32)
        self.prior_init_noise_std: float = float(cfg.get('prior_init_noise_std', 0.0))
        self.normal_s: float = float(cfg.get('normal_s', 0.001))
        self.use_robust_normal: bool = bool(cfg.get('use_robust_normal', False))
        self.normal_robust_thres: float = float(cfg.get('normal_robust_thres', 0.1))
        self.prior_s_rot: float = float(cfg.get('prior_s_rot', cfg.get('prior_s', 0.02)))
        self.prior_s_tr: float = float(cfg.get('prior_s_tr', cfg.get('prior_s', 0.02)))
        self.prior_s_depth: float = float(cfg.get('prior_s_depth', cfg.get('prior_s', 0.02)))
        try:
            self.gt_with_noise_init: bool = bool(cfg.get('gt_with_noise_init', cfg.get('gt_init', False)))
        except Exception:
            self.gt_with_noise_init = bool(cfg.get('gt_init', False))
        self.seed: int = int(cfg.get('seed', 0))
        self.jax_key = None
        self.first_bad_found: bool = False
        self.first_bad_iter: int | None = None
        self.first_bad_var: int | None = None
        self._prev_iter_snapshot = None
        self.adj_factor_idx_host = None
        self.N_total: int | None = None
        try:
            if self.config_path is not None and Path(self.config_path).exists():
                dst = self.out_root / Path(self.config_path).name
                if str(Path(self.config_path).resolve()) != str(dst.resolve()):
                    with open(dst, 'w') as f:
                        yaml.safe_dump(self.cfg, f, sort_keys=False)
                else:
                    pass
        except Exception as e:
            print(f"[warn] Failed to write config to '{self.out_root}': {e}")

    def _ensure_twist_graph(self) -> None:
        """Initialize twist graph once (D_op=6) so we can run pre-warmup before pair optimization."""
        if self.vars is not None:
            return
        (self.vars, self.dataterm_facs, self.between_facs, self.prior_facs) = build_twist_graph(self.H, self.W, self.D, self.D_op)
        try:
            self.N_total = int(self.vars.belief.eta.shape[0])
        except Exception:
            self.N_total = None
        self._set_prior_measurements(self.prior_init_noise_std)
        self.prior_facs.threshold = jnp.ones_like(self.prior_facs.threshold) * self.robust_thres
        self.set_hyperparameters()

    def _run_pre_warmup(self) -> None:
        """Run photo_warmup iterations using only smoothness and prior before rendering frame 2.
        Does not increment global_iter_count and is excluded from iterations_per_pair.
        """
        try:
            num_pre = int(self.photo_warmup)
        except Exception:
            num_pre = 0
        if num_pre <= 0:
            return
        self._ensure_twist_graph()
        linpoints = jax.vmap(lambda x: x.mu())(self.vars.belief)
        est_twist_trace = []
        for i in range(num_pre):
            (self.vars, vtof_msgs, linpoints) = update_variable_twist_SE3(self.vars, linpoints, alpha=self.retract_alpha)
            (self.between_facs, self.vars) = update_factor_manifold_hJ(self.between_facs, self.vars, vtof_msgs, linpoints, h2_fn, l2)
            (self.prior_facs, self.vars) = update_factor_manifold_hJ(self.prior_facs, self.vars, vtof_msgs, linpoints, h3_fn, l2)
            if (i + 1) % max(1, int(self.iteration_log_interval)) == 0 or i + 1 == num_pre:
                est_twist = jnp.mean(linpoints, axis=0)
                est_twist_trace.append(est_twist)
                out_path = self.logs_dir / 'prewarm_rotation_error.png'
                save_plot_twist_iter(None, np.array([jax.device_get(w) for w in est_twist_trace]), self.iteration_log_interval, out_path)
                print(f'[prewarm] it={i + 1}/{num_pre}: saved twist trace -> {out_path}', flush=True)
        self.photo_warmup = 0

    def _parse_hp_schedule(self) -> List[Dict[str, Any]]:
        """
        Parse root-level hyperparameters schedule from config.

        Expected YAML format:
          hyperparameters:
            - at: 0
              dataterm_s: 0.05
              smoothness_s_rot: 0.001
              smoothness_s_tr: 0.001
              prior_s: 0.02
              use_robust: true
              robust_thres: 0.5
              iteration_log_interval: 50
        """
        schedule_cfg = self.cfg.get('hyperparameters', [])
        parsed: List[Dict[str, Any]] = []
        if isinstance(schedule_cfg, list):
            for item in schedule_cfg:
                if not isinstance(item, dict):
                    continue
                at_val = item.get('at', item.get('iteration'))
                if at_val is None:
                    continue
                try:
                    at_iter = int(at_val)
                except Exception:
                    continue
                updates = {k: v for (k, v) in item.items() if k not in ('at', 'iteration')}
                parsed.append({'at': at_iter, 'updates': updates})
        parsed.sort(key=lambda d: d['at'])
        return parsed

    def _apply_hp_updates(self, updates: Dict[str, Any]) -> None:
        """Apply supported hyperparameter updates to self.* fields."""
        supported_keys = {'dataterm_s', 'smoothness_s_rot', 'smoothness_s_tr', 'prior_s', 'use_robust', 'robust_thres', 'iteration_log_interval', 'photometric_stride', 'photo_warmup', 'viz_stride', 'prior_init_axis_angle', 'prior_init_noise_std', 'prior_boundary_scale', 'prior_boundary_s'}
        if 'prior_init_axis_angle' in updates:
            val = updates['prior_init_axis_angle']
            try:
                if isinstance(val, (list, tuple)) and len(val) == self.D_op:
                    self.prior_init_axis = jnp.array(val, dtype=jnp.float32)
            except Exception:
                pass
        if 'prior_init_noise_std' in updates:
            try:
                self.prior_init_noise_std = float(updates['prior_init_noise_std'])
            except Exception:
                pass
        for k in supported_keys:
            if k in ('prior_init_axis_angle', 'prior_init_noise_std'):
                continue
            if k in updates:
                current = getattr(self, k, None)
                val = updates[k]
                if isinstance(current, bool):
                    setattr(self, k, bool(val))
                elif isinstance(current, int):
                    setattr(self, k, int(val))
                elif isinstance(current, float):
                    setattr(self, k, float(val))
                else:
                    setattr(self, k, val)
        try:
            if ('prior_init_axis_angle' in updates or 'prior_init_noise_std' in updates) and self.prior_facs is not None:
                self._set_prior_measurements(self.prior_init_noise_std)
        except Exception:
            pass
        if updates:
            try:
                keys_applied = {k: updates[k] for k in updates.keys() if k in supported_keys}
                if keys_applied:
                    print(f'[HP] Applied hyperparameters at iter {self.global_iter_count}: {keys_applied}')
            except Exception:
                pass

    def _maybe_apply_scheduled_hyperparameters(self) -> bool:
        """
        Apply any schedule entries whose 'at' <= current global iteration.
        Returns True if any update was applied this call.
        """
        changed = False
        while self.hp_schedule_idx + 1 < len(self.hp_schedule) and self.hp_schedule[self.hp_schedule_idx + 1]['at'] <= self.global_iter_count:
            self.hp_schedule_idx += 1
            self._apply_hp_updates(self.hp_schedule[self.hp_schedule_idx]['updates'])
            changed = True
        return changed

    def _set_prior_measurements(self, noise_std: float | None=None) -> None:
        """
        (Re)apply prior measurements `z` for SO(3)/SE(3) prior factors using
        the current `self.prior_init_axis` and optional Gaussian noise.
        """
        if self.prior_facs is None:
            return
        try:
            base = np.array(jax.device_get(self.prior_init_axis), dtype=np.float32).reshape(1, -1)
        except Exception:
            base = np.array(self.prior_init_axis, dtype=np.float32).reshape(1, -1)
        if noise_std is None:
            noise_std = 0.0
        if float(noise_std) > 0.0:
            noise = np.random.randn(self.prior_facs.z.shape[0], self.D_op).astype(np.float32) * float(noise_std)
            z = base.repeat(self.prior_facs.z.shape[0], axis=0) + noise
        else:
            z = np.repeat(base, self.prior_facs.z.shape[0], axis=0)
        if _IPD and z.shape[1] >= 7:
            z[:, 6] += np.float32(1.0)
        self.prior_facs.z = jnp.array(z)

    def _set_deterministic(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import jax.random as jrandom
            self.jax_key = jrandom.PRNGKey(self.seed)
        except Exception:
            self.jax_key = None

    def _render_twist_overlay(self, it_count: int, linpoints: jnp.ndarray, K_prev: np.ndarray, depth_prev: np.ndarray, T_rel: np.ndarray, rgb_prev: np.ndarray, out_path: Path, est_depth_prev: np.ndarray | None=None, est_trans_scale: float | None=None) -> None:
        """
        Draw SE(3) projections on the previous image plane:
          - Green: per-pixel estimated SE(3) (from leaf variables)
          - Red: ground-truth relative SE(3) (extrinsic_cur @ inv(extrinsic_prev))
        Uses depth from previous frame for unprojection.
        """
        try:
            (H, W) = (self.H, self.W)
            base = (np.clip(rgb_prev, 0.0, 1.0) * 255.0).astype(np.uint8)
            vis = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            N = self.N
            if linpoints.shape[0] < N:
                return
            est_twists = np.array(jax.device_get(linpoints[:N, :6]))
            K = np.array(K_prev, dtype=np.float64)
            K_inv = np.linalg.inv(K)
            T_gt = np.array(T_rel, dtype=np.float64)

            def _so3_exp_np(omega: np.ndarray) -> np.ndarray:
                theta = np.linalg.norm(omega)
                if theta < 1e-12:
                    return np.eye(3, dtype=np.float64)
                k = omega / theta
                Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
                s = np.sin(theta)
                c = np.cos(theta)
                return np.eye(3, dtype=np.float64) + s * Kx + (1.0 - c) * (Kx @ Kx)

            def _se3_exp_np(xi: np.ndarray) -> np.ndarray:
                omega = xi[:3]
                v = xi[3:]
                R = _so3_exp_np(omega)
                theta = np.linalg.norm(omega)
                if theta < 1e-12:
                    V = np.eye(3, dtype=np.float64)
                else:
                    Kx = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]], dtype=np.float64) / theta
                    s = np.sin(theta)
                    c = np.cos(theta)
                    V = np.eye(3, dtype=np.float64) + (1 - c) / theta ** 2 * Kx + (theta - s) / theta ** 3 * (Kx @ Kx)
                t = V @ v
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t
                return T
            s = max(1, int(self.viz_stride))
            mu = min(8, W - 1)
            mv = min(8, H - 1)
            color_est = (0, 255, 0)
            color_gt = (0, 0, 255)
            thickness = 1
            tip_length = 0.3
            depth_gt = np.asarray(depth_prev).reshape(H, W)
            depth_est = np.asarray(est_depth_prev).reshape(H, W) if est_depth_prev is not None else depth_gt
            for v in range(mv, H, s):
                for u in range(mu, W, s):
                    u0 = float(u) + 0.5
                    v0 = float(v) + 0.5
                    p_cam = K_inv @ np.array([u0, v0, 1.0], dtype=np.float64)
                    d_gt = float(depth_gt[v, u])
                    if np.isfinite(d_gt) and d_gt > 0.0:
                        X_prev_gt = np.array([p_cam[0] * d_gt, p_cam[1] * d_gt, d_gt, 1.0], dtype=np.float64)
                        X_gt = T_gt @ X_prev_gt
                        if abs(X_gt[3]) >= 1e-12 and abs(X_gt[2]) >= 1e-12:
                            q_gt = K @ (X_gt[:3] / X_gt[3])
                            (ug, vg) = (q_gt[0] / q_gt[2], q_gt[1] / q_gt[2])
                            (x0, y0) = (int(round(u0)), int(round(v0)))
                            (xg, yg) = (int(round(ug)), int(round(vg)))
                            cv2.arrowedLine(vis, (x0, y0), (xg, yg), color_gt, thickness=thickness, tipLength=tip_length)
                    d_est = float(depth_est[v, u])
                    if np.isfinite(d_est) and d_est > 0.0:
                        X_prev_est = np.array([p_cam[0] * d_est, p_cam[1] * d_est, d_est, 1.0], dtype=np.float64)
                        idx = v * W + u
                    T_est = _se3_exp_np(est_twists[idx])
                    if est_trans_scale is not None:
                        T_est[:3, 3] = T_est[:3, 3] * float(est_trans_scale)
                        X_est = T_est @ X_prev_est
                        if abs(X_est[3]) >= 1e-12 and abs(X_est[2]) >= 1e-12:
                            q_est = K @ (X_est[:3] / X_est[3])
                            (ue, ve) = (q_est[0] / q_est[2], q_est[1] / q_est[2])
                            (x0, y0) = (int(round(u0)), int(round(v0)))
                            (xe, ye) = (int(round(ue)), int(round(ve)))
                            cv2.arrowedLine(vis, (x0, y0), (xe, ye), color_est, thickness=thickness, tipLength=tip_length)
            cv2.imwrite(str(out_path), vis)
        except Exception as e:
            print(f'[viz] SE3 overlay render failed at iter={it_count}: {e}')

    def _save_gt_flow_histograms(self, K_prev: np.ndarray, depth_prev: np.ndarray, T_rel: np.ndarray, tag: str) -> None:
        """
        Compute ground-truth optical flow (image-plane displacement) induced by T_rel using
        previous frame depth and intrinsics. Save length histogram (10 bins from min-0.5 to max+0.5)
        and 8-direction polar arrow plot. Also save min/median/max lengths to a txt file.
        Histograms are normalized to sum to 1.
        """
        try:
            (H, W) = (int(self.H), int(self.W))
            depth = np.asarray(depth_prev).reshape(H, W)
            K = np.asarray(K_prev, dtype=np.float64)
            K_inv = np.linalg.inv(K)
            (yy, xx) = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
            u0 = xx + 0.5
            v0 = yy + 0.5
            coords = np.stack([u0, v0, np.ones_like(u0)], axis=-1).reshape(-1, 3)
            rays = (K_inv @ coords.T).T
            d = depth.reshape(-1)
            valid = np.isfinite(d) & (d > 0.0)
            if not np.any(valid):
                print('[flow] No valid depths for GT flow stats')
                return
            X_prev = np.empty((d.size, 4), dtype=np.float64)
            X_prev[:, :3] = rays * d[:, None]
            X_prev[:, 3] = 1.0
            X_cur = (np.asarray(T_rel, dtype=np.float64) @ X_prev.T).T
            valid &= np.isfinite(X_cur[:, 2]) & (np.abs(X_cur[:, 3]) > 1e-12) & (X_cur[:, 2] != 0.0)
            if not np.any(valid):
                print('[flow] No valid projections for GT flow stats')
                return
            X = X_cur[valid]
            U0 = u0.reshape(-1)[valid]
            V0 = v0.reshape(-1)[valid]
            q = (K @ (X[:, :3].T / X[:, 3])).T
            u1 = q[:, 0] / q[:, 2]
            v1 = q[:, 1] / q[:, 2]
            du = u1 - U0
            dv = v1 - V0
            L = np.sqrt(du * du + dv * dv)
            L_min = float(np.min(L))
            L_max = float(np.max(L))
            L_med = float(np.median(L))
            bin_edges = np.linspace(L_min - 0.5, L_max + 0.5, 11)
            (hist_counts, _) = np.histogram(L, bins=bin_edges)
            hist_sum = float(hist_counts.sum())
            length_hist = hist_counts.astype(np.float64) / (hist_sum if hist_sum > 0 else 1.0)
            angles = np.mod(np.arctan2(dv, du), 2.0 * np.pi)
            dir_edges = np.linspace(0.0, 2.0 * np.pi, 9)
            (dir_counts, _) = np.histogram(angles, bins=dir_edges)
            dir_sum = float(dir_counts.sum())
            dir_hist = dir_counts.astype(np.float64) / (dir_sum if dir_sum > 0 else 1.0)
            stats_path = self.logs_dir / f'gt_flow_stats_{tag}.txt'
            with open(stats_path, 'w') as f:
                f.write(f'min_length_px: {L_min:.6f}\n')
                f.write(f'median_length_px: {L_med:.6f}\n')
                f.write(f'max_length_px: {L_max:.6f}\n')
                f.write(f'num_valid: {L.size}\n')
            print(f'[flow] GT flow stats (tag={tag}) -> min={L_min:.3f}, med={L_med:.3f}, max={L_max:.3f}; saved {stats_path}')
            try:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                (fig, ax) = plt.subplots(1, 1, figsize=(8, 4))
                ax.bar(centers, length_hist, width=(bin_edges[1] - bin_edges[0]) * 0.9, color='C0', edgecolor='k')
                ax.set_xlabel('Flow length (pixels)')
                ax.set_ylabel('Normalized frequency')
                ax.set_title('GT optical flow length histogram')
                fig.tight_layout()
                fig.savefig(self.logs_dir / f'gt_flow_length_hist_{tag}.png')
                plt.close(fig)
            except Exception as e:
                print(f'[flow] Failed to save length histogram: {e}')
            try:
                dir_centers = 0.5 * (dir_edges[:-1] + dir_edges[1:])
                fig2 = plt.figure(figsize=(5, 5))
                axp = fig2.add_subplot(111, projection='polar')
                axp.set_theta_zero_location('E')
                axp.set_theta_direction(-1)
                axp.set_rlim(0.0, 1.0)
                for (th, r) in zip(dir_centers, dir_hist):
                    axp.annotate('', xy=(th, r), xytext=(th, 0.0), arrowprops=dict(arrowstyle='->', color='C1', lw=2))
                axp.set_title('GT optical flow direction (8 bins, normalized)')
                fig2.tight_layout()
                fig2.savefig(self.logs_dir / f'gt_flow_dir_hist_{tag}.png')
                plt.close(fig2)
            except Exception as e:
                print(f'[flow] Failed to save direction histogram: {e}')
        except Exception as e:
            print(f'[flow] Failed GT flow histogram computation (tag={tag}): {e}')

    def set_hyperparameters(self):
        self.dataterm_facs.z_Lam = jnp.tile(jnp.eye(self.D) * (1.0 / self.dataterm_s ** 2), (self.dataterm_facs.z_Lam.shape[0], 1, 1))
        try:
            (H, W) = (int(self.H), int(self.W))
            cut = int(_CUT)
            (yy, xx) = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
            interior_mask_np = (yy >= cut) & (yy < H - cut) & (xx >= cut) & (xx < W - cut)
            interior_mask_leaves = jnp.array(interior_mask_np.reshape(-1))
        except Exception:
            interior_mask_leaves = jnp.ones((self.N,), dtype=bool)
        num_prior = int(self.prior_facs.z_Lam.shape[0])
        leaf_count = min(self.N, num_prior)
        stronger_mask = jnp.zeros((num_prior,), dtype=bool)
        boundary_flags = (~interior_mask_leaves)[:leaf_count]
        stronger_mask = stronger_mask.at[:leaf_count].set(boundary_flags)
        scale_interior = 1.0 / self.prior_s ** 2
        scale_boundary = 1.0 / self.prior_boundary_s ** 2
        scale_vec = jnp.where(stronger_mask, scale_boundary, scale_interior).reshape(-1, 1, 1)
        self.prior_facs.z_Lam = jnp.eye(self.D_op)[None, :, :] * scale_vec
        self.between_facs.z_Lam = jnp.tile(jnp.diag(jnp.array([1.0 / self.smoothness_s_rot ** 2, 1.0 / self.smoothness_s_rot ** 2, 1.0 / self.smoothness_s_rot ** 2, 1.0 / self.smoothness_s_tr ** 2, 1.0 / self.smoothness_s_tr ** 2, 1.0 / self.smoothness_s_tr ** 2])), (self.between_facs.z_Lam.shape[0], 1, 1))
        self.between_facs.threshold = jnp.ones_like(self.between_facs.threshold) * self.robust_thres
        self.dataterm_facs.threshold = jnp.ones_like(self.dataterm_facs.threshold) * self.robust_thres
