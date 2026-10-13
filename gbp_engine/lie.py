import jax.numpy as jnp
from jax import config as _jax_config
# PIXVOD_FP32: unset or 0 -> float64 everywhere on the device (jax_enable_x64, unchanged).  1 -> x64 off: every
# JAX array and operation is float32 (explicit float64 requests and float64 numpy inputs are converted to float32 by
# JAX); host-side numpy setup and diagnostics keep their own dtypes and are converted when uploaded.
_FP32 = __import__('os').environ.get('PIXVOD_FP32', '0')
if _FP32 not in ('0', '1'):
    raise SystemExit(f'PIXVOD_FP32 must be 0 or 1; got {_FP32!r}')
_jax_config.update('jax_enable_x64', _FP32 != '1')
if _FP32 == '1':
    print('[FP32] jax_enable_x64 off: device computation in float32', flush=True)
from jax import jit, lax
from functools import partial
_EPS = 0.001
_EPS_PI = 0.0001

_FAST_MM = __import__('os').environ.get('PIXVOD_FAST_MATMUL', '1') != '0'

def _mm(a, b):
    """Product of two small dense matrices (batched under vmap).  The standard path emits the
    contraction as a broadcast multiply plus a reduction, which XLA fuses into the surrounding
    elementwise work; that is far faster than the cuBLAS batched gemm a plain dot lowers to,
    because these matrices are 6x6/7x7 and the batch is large.  It rounds differently in the
    last bit than cuBLAS, so it changes the trajectory without changing the mathematics; it
    became the standard after a 42-sequence evaluation left the passing and divergent counts
    unchanged.  PIXVOD_FAST_MATMUL=0 restores the cuBLAS path exactly."""
    if not _FAST_MM:
        return a @ b
    return (a[..., :, :, None] * b[..., None, :, :]).sum(axis=-2)

def _mv(A, v):
    """Matrix-vector form of _mm, with the same gate and the same caveat."""
    if not _FAST_MM:
        return A @ v
    return (A * v[..., None, :]).sum(axis=-1)

def _skew(w):
    (wx, wy, wz) = w
    return jnp.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])

def _vee(S):
    return jnp.array([S[2, 1], S[0, 2], S[1, 0]])

def _axis_angle(w):
    theta = jnp.linalg.norm(w)

    def small(_):
        return w * 0.0

    def large(_):
        return w / jnp.where(theta == 0.0, 1.0, theta)
    axis = lax.cond(theta < _EPS, small, large, operand=None)
    return (axis, theta)

def _A(theta):

    def small(_):
        return 1.0 - theta ** 2 / 6.0 + theta ** 4 / 120.0

    def general(_):
        return jnp.sin(theta) / theta
    return lax.cond(theta < _EPS, small, general, operand=None)

def _B(theta):

    def small(_):
        return 0.5 - theta ** 2 / 24.0 + theta ** 4 / 720.0

    def general(_):
        return (1.0 - jnp.cos(theta)) / theta ** 2
    return lax.cond(theta < _EPS, small, general, operand=None)

def _C(theta):

    def small(_):
        return 1.0 / 6.0 - theta ** 2 / 120.0 + theta ** 4 / 5040.0

    def general(_):
        return (theta - jnp.sin(theta)) / theta ** 3
    return lax.cond(theta < _EPS, small, general, operand=None)

def so3_hat(w):
    return _skew(w)

@jit
def so3_Exp(w):
    th = jnp.linalg.norm(w)
    W = _skew(w)
    (A, B) = (_A(th), _B(th))
    return jnp.eye(3) + A * W + B * (W @ W)

@jit
def SO3_Log(R):
    Wvee = _vee(0.5 * (R - R.T))
    s = jnp.linalg.norm(Wvee)
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
def so3_Jl(w):
    """
    Left Jacobian on SO(3).
    J_l(w) = I + (1 - cos θ)/θ^2 [w]_x + (θ - sin θ)/θ^3 [w]_x^2
    (same second coefficient as J_r, sign flip on the first term).
    Small-angle series identical to geometry.py for value agreement.
    """
    th = jnp.linalg.norm(w)
    W = _skew(w)
    I = jnp.eye(3, dtype=w.dtype)
    th2 = th * th
    eps = jnp.finfo(w.dtype).eps
    valid = th > eps
    coeff0 = jnp.where(valid, +(1.0 - jnp.cos(th)) / (th2 + 1e-300), +(0.5 - th2 / 24.0))
    coeff1 = jnp.where(valid, (th - jnp.sin(th)) / (th2 * th + jnp.array(1e-300, dtype=w.dtype)), 1.0 / 6.0 - th2 / 120.0)
    return I + coeff0 * W + coeff1 * (W @ W)

@jit
def so3_Jr(w):
    """
    Right Jacobian on SO(3).
    Matches geometry.py small-angle handling and coefficients for value-level agreement.
    J_r(w) = I - (1 - cos θ)/θ^2 [w]_x + (θ - sin θ)/θ^3 [w]_x^2
    with small-angle series:
      (1 - cos θ)/θ^2 ≈ 0.5 - θ^2/24
      (θ - sin θ)/θ^3 ≈ 1/6 - θ^2/120
    """
    th = jnp.linalg.norm(w)
    W = _skew(w)
    I = jnp.eye(3, dtype=w.dtype)
    th2 = th * th
    eps = jnp.finfo(w.dtype).eps
    valid = th > eps
    coeff0 = jnp.where(valid, -(1.0 - jnp.cos(th)) / (th2 + 1e-300), -(0.5 - th2 / 24.0))
    coeff1 = jnp.where(valid, (th - jnp.sin(th)) / (th2 * th + jnp.array(1e-300, dtype=w.dtype)), 1.0 / 6.0 - th2 / 120.0)
    return I + coeff0 * W + coeff1 * (W @ W)

@jit
def so3_Jl_inv(w):
    th = jnp.linalg.norm(w)
    W = _skew(w)
    I = jnp.eye(3)

    def small(_):
        return I - 0.5 * W + 1.0 / 12.0 * (W @ W)

    def general(_):
        th_safe = jnp.minimum(th, jnp.pi - _EPS_PI)
        s = jnp.sin(th_safe)
        s = jnp.where(jnp.abs(s) < _EPS, _EPS, s)
        coef = 1.0 / (th_safe * th_safe) - (1.0 + jnp.cos(th_safe)) / (2.0 * th_safe * s)
        return I - 0.5 * W + coef * (W @ W)
    return lax.cond(th < _EPS, small, general, operand=None)

@jit
def so3_Jr_inv(w):
    return so3_Jl_inv(-w)

def se3_hat(xi):
    w = xi[:3]
    v = xi[3:]
    W = _skew(w)
    Xi = jnp.zeros((4, 4))
    Xi = Xi.at[:3, :3].set(W)
    Xi = Xi.at[:3, 3].set(v)
    return Xi

@jit
def _ad_se3(xi):
    w = xi[:3]
    v = xi[3:]
    W = _skew(w)
    V = _skew(v)
    upper = jnp.concatenate([W, jnp.zeros((3, 3))], axis=1)
    lower = jnp.concatenate([V, W], axis=1)
    return jnp.concatenate([upper, lower], axis=0)

@jit
def Ad_from_RT(R, t):
    """Ad_T with T = (R,t). Shape: (6,6)"""
    zero = jnp.zeros((3, 3))
    Rt = _skew(t) @ R
    top = jnp.concatenate([R, zero], axis=1)
    bot = jnp.concatenate([Rt, R], axis=1)
    return jnp.concatenate([top, bot], axis=0)

@jit
def Ad_from_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return Ad_from_RT(R, t)

@jit
def Ad_from_se3(xi):
    """Ad_{Exp(xi)}."""
    T = se3_Exp(xi)
    return Ad_from_T(T)

@jit
def se3_Exp(xi):
    w = xi[:3]
    v = xi[3:]
    R = so3_Exp(w)
    th = jnp.linalg.norm(w)
    W = _skew(w)
    (B, C) = (_B(th), _C(th))
    Vmat = jnp.eye(3) + B * W + C * (W @ W)
    t = Vmat @ v
    T = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(t)
    return T

@jit
def SE3_Log(T):
    R = T[:3, :3]
    t = T[:3, 3]
    w = SO3_Log(R)
    V_inv = so3_Jl_inv(w)
    v = V_inv @ t
    return jnp.concatenate([w, v])
_BERNOULLI = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0]
import os as _os
_LEGACY_JR_SIGN = '0' == '1'
_SMOOTH_FULL_J = '0' == '1'
_DANCERS_JR = '0' == '1'

@partial(jit, static_argnames='terms')
def _power_series_left_jacobian(ad, terms: int=10):
    I = jnp.eye(6)
    J = I
    ad_pow = I
    sign = -1.0
    denom = 2.0
    for n in range(1, terms + 1):
        ad_pow = _mm(ad_pow, ad)
        coef = 1.0 / denom
        J = J + coef * ad_pow
        sign = -sign
        denom *= n + 2
    return J

@partial(jit, static_argnames='terms')
def _power_series_left_jacobian_inv(ad, terms: int=10):
    I = jnp.eye(6)
    Jinv = jnp.zeros((6, 6))
    ad_pow = I
    fact = 1.0
    for n in range(0, min(terms, len(_BERNOULLI)) + 1):
        coef = _BERNOULLI[n] / fact
        Jinv = Jinv + coef * ad_pow
        ad_pow = _mm(ad_pow, ad)
        fact *= n + 1 if n + 1 > 0 else 1
    return Jinv
_N_TERMS = 6

def _se3_Q(xi):
    w = xi[:3]
    v = xi[3:]
    V = _skew(v)
    W = _skew(w)
    th2 = jnp.dot(w, w)
    small = th2 < 0.25
    th2_safe = jnp.where(small, 1.0, th2)
    th = jnp.sqrt(th2_safe)
    sin_th = jnp.sin(th)
    cos_th = jnp.cos(th)
    th4 = th2 * th2
    th6 = th4 * th2
    B = jnp.where(small, 1.0 / 6.0 - th2 / 120.0 + th4 / 5040.0 - th6 / 362880.0, (th - sin_th) / (th2_safe * th))
    C = jnp.where(small, -1.0 / 24.0 + th2 / 720.0 - th4 / 40320.0 + th6 / 3628800.0, (1.0 - th2_safe / 2.0 - cos_th) / (th2_safe * th2_safe))
    D = jnp.where(small, -1.0 / 60.0 + th2 / 1260.0 - th4 / 60480.0 + th6 / 4989600.0, C - 3.0 * (th - sin_th - th2_safe * th / 6.0) / (th2_safe * th2_safe * th))
    VW = V @ W
    WV = VW.T
    WVW = WV @ W
    VWW = VW @ W
    return 0.5 * V + B * (WV + VW + WVW) - C * (VWW - VWW.T - 3.0 * WVW) - D * (WVW @ W)

def _se3_Jl_closed(xi):
    J3 = so3_Jl(xi[:3])
    Q = _se3_Q(xi)
    Z = jnp.zeros((3, 3), dtype=xi.dtype)
    return jnp.concatenate([jnp.concatenate([J3, Z], axis=1), jnp.concatenate([Q, J3], axis=1)], axis=0)

def _se3_Jl_inv_closed(xi):
    Ji = so3_Jl_inv(xi[:3])
    Q = _se3_Q(xi)
    Z = jnp.zeros((3, 3), dtype=xi.dtype)
    return jnp.concatenate([jnp.concatenate([Ji, Z], axis=1), jnp.concatenate([-Ji @ Q @ Ji, Ji], axis=1)], axis=0)

@jit
def se3_Jl(xi):
    return _power_series_left_jacobian(_ad_se3(xi), terms=_N_TERMS)

@jit
def se3_Jr(xi):
    return _power_series_left_jacobian(_ad_se3(-xi), terms=_N_TERMS)

@jit
def se3_Jl_inv(xi):
    return _power_series_left_jacobian_inv(_ad_se3(xi), terms=_N_TERMS)

@jit
def se3_Jr_inv(xi):
    return _power_series_left_jacobian_inv(_ad_se3(-xi), terms=_N_TERMS)

def _bracket(a, b, ad_fn):
    return ad_fn(a) @ b

@partial(jit, static_argnames=('ad_fn', 'order'))
def bch_se3_displacement(x, y, ad_fn, order=4):
    xm = -x
    z = y + xm
    c1 = _bracket(xm, y, ad_fn)
    z = z + 0.5 * c1 * (1.0 if order >= 2 else 0.0)
    c2 = _bracket(xm, c1, ad_fn)
    c3 = _bracket(y, _bracket(y, xm, ad_fn), ad_fn)
    z = z + 1.0 / 12.0 * (c2 + c3) * (1.0 if order >= 3 else 0.0)
    c4 = _bracket(y, c2, ad_fn)
    z = z + -1.0 / 24.0 * c4 * (1.0 if order >= 4 else 0.0)
    return z

def _so3_bracket(a, b):
    return jnp.cross(a, b)

@partial(jit, static_argnames='order')
def bch_so3_displacement(x, y, order=4):
    xm = -x
    z = y + xm
    c1 = _so3_bracket(xm, y)
    z = z + 0.5 * c1 * (1.0 if order >= 2 else 0.0)
    c2 = _so3_bracket(xm, c1)
    c3 = _so3_bracket(y, _so3_bracket(y, xm))
    z = z + 1.0 / 12.0 * (c2 + c3) * (1.0 if order >= 3 else 0.0)
    c4 = _so3_bracket(y, c2)
    z = z + -1.0 / 24.0 * c4 * (1.0 if order >= 4 else 0.0)
    return z

@jit
def so3_geo_dist_explicit(x, y):
    R_rel = so3_Exp(-x) @ so3_Exp(y)
    c = jnp.clip((jnp.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    th = jnp.arccos(c)

    def small(_):
        return bch_so3_displacement(x, y, order=4)

    def general(_):
        return SO3_Log(R_rel)
    r = lax.cond(th < 0.01, small, general, operand=None)
    Jrinv = so3_Jr_inv(r)
    Jr_Jx = -Jrinv.mT
    Jr_Jy = Jrinv
    return (r, jnp.concatenate([Jr_Jx, Jr_Jy], axis=-1))

def so3r3_geo_dist_explicit(x, y):
    (rR, JR) = so3_geo_dist_explicit(x[:3], y[:3])
    (rT, JT_Jx, JT_Jy) = (y[3:] - x[3:], -jnp.eye(3, dtype=x.dtype), jnp.eye(3, dtype=x.dtype))
    r = jnp.concatenate([rR, rT], axis=0)
    Jr_Jx = jnp.concatenate([jnp.concatenate([JR[:, :3], jnp.zeros((3, 3))], axis=-1), jnp.concatenate([jnp.zeros((3, 3)), JT_Jx], axis=-1)], axis=0)
    Jr_Jy = jnp.concatenate([jnp.concatenate([JR[:, 3:], jnp.zeros((3, 3))], axis=-1), jnp.concatenate([jnp.zeros((3, 3)), JT_Jy], axis=-1)], axis=0)
    return (r, jnp.concatenate([Jr_Jx, Jr_Jy], axis=-1))

@jit
def so3_geo_dist(x, y):
    R_rel = so3_Exp(-x) @ so3_Exp(y)
    c = jnp.clip((jnp.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    th = jnp.arccos(c)

    def small(_):
        return bch_so3_displacement(x, y, order=4)

    def general(_):
        return SO3_Log(R_rel)
    return lax.cond(th < 0.01, small, general, operand=None)

@jit
def se3_geo_dist_explicit(x, y):
    T_rel = se3_Exp(-x) @ se3_Exp(y)
    r = SE3_Log(T_rel)
    Jr_r_inv = se3_Jr_inv(r)
    Ad_Tinv = Ad_from_se3(-r)
    Jx_param = -Jr_r_inv @ Ad_Tinv
    Jy_param = Jr_r_inv
    return (r, jnp.concatenate([Jx_param, Jy_param], axis=-1))

@jit
def se3_residual_wrt_reference(x, y, T):
    T_rel = jnp.linalg.inv(T) @ se3_Exp(-x) @ T @ se3_Exp(y)
    r = SE3_Log(T_rel)
    t = SE3_Log(T)
    Jr_r_inv = se3_Jr_inv(r)
    Ad_Trelinv = Ad_from_se3(-r)
    Ad_Tinv = Ad_from_se3(-t)
    Jx_param = -Jr_r_inv @ Ad_Trelinv @ Ad_Tinv
    Jy_param = Jr_r_inv
    return (r, jnp.concatenate([Jx_param, Jy_param], axis=-1))

def se3_geo_dist(x, y):
    return SE3_Log(se3_Exp(-x) @ se3_Exp(y))
