import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial
import numpy as np
import time
from .defaults import gate as _accel_gate
# PIXVOD_DONATE: the variable store and the factor set are donated to the jitted updates of the default
# path, so XLA updates them in place instead of copying them first (values unchanged).
_DON = ('facs', 'vars') if _accel_gate('PIXVOD_DONATE') else ()
_DON_V = ('vars',) if _accel_gate('PIXVOD_DONATE') else ()
from .lie import so3_Jr, so3_Jr_inv, so3_geo_dist, se3_Jr, se3_Jr_inv, se3_geo_dist, _EPS, _mm, _mv

_FP32_RIDGE = __import__('os').environ.get('PIXVOD_FP32_RIDGE', '0') == '1'
if _FP32_RIDGE:
    print('[FP32-RIDGE] safe_mu_ref dense ridge >= 32 ulp of max diag', flush=True)
# PIXVOD_UNARY_EXACT: 1 -> compute_ftov_msg returns a unary factor's potential as its message instead of
# (potential + vtof) - vtof, which equals it in exact arithmetic but in float32 drops potential entries below the
# rounding step of the variable's belief (the depth prior's 100 against a belief depth precision of ~1e10).
_UNARY_EXACT = __import__('os').environ.get('PIXVOD_UNARY_EXACT', '0')
if _UNARY_EXACT not in ('0', '1'):
    raise SystemExit(f'PIXVOD_UNARY_EXACT must be 0 or 1; got {_UNARY_EXACT!r}')
_UNARY_EXACT = _UNARY_EXACT == '1'
if _UNARY_EXACT:
    print('[UNARY-EXACT] unary factor message = potential', flush=True)


@jax.jit
def update_variable_twist_SO3(vars, old_linpoints):
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SO3, in_axes=(0, 0))(old_linpoints, old_posterior)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(RebaseSafe_SO3, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@partial(jax.jit, static_argnames=['alpha'])
def update_variable_SE3D(vars, old_linpoints, alpha: float=1.0, msg_slots=None):
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SE3_plus_depth, in_axes=(0, 0, None))(old_linpoints, old_posterior, alpha)
    if msg_slots is None:

        def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
            return jax.vmap(Rebase_SE3_plus_depth, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
        new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    else:
        # msg_slots lists the message slots some factor writes.  Every other slot holds an
        # exactly zero Gaussian for the whole run, and the rebase of a zero message depends
        # only on the variable's two linearisation points, not on the port.  So rebase each
        # occupied slot individually and fill the rest with that one per-variable result --
        # the same value the port-wise rebase produces, down to the sign of its zeros.
        (slot_idx, slot_var) = msg_slots
        (n_var, n_port, d_op) = vars.msgs.eta.shape
        n_slot = n_var * n_port
        occupied = Gaussian(vars.msgs.eta.reshape(n_slot, d_op)[slot_idx], vars.msgs.Lam.reshape(n_slot, d_op, d_op)[slot_idx])
        rebased = jax.vmap(Rebase_SE3_plus_depth, in_axes=(0, 0, 0))(new_linpoints[slot_var], old_linpoints[slot_var], occupied)
        empty = Gaussian(jnp.zeros((n_var, d_op), dtype=vars.msgs.eta.dtype), jnp.zeros((n_var, d_op, d_op), dtype=vars.msgs.Lam.dtype))
        filler = jax.vmap(Rebase_SE3_plus_depth, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, empty)
        new_eta = jnp.broadcast_to(filler.eta[:, None, :], (n_var, n_port, d_op)).reshape(n_slot, d_op).at[slot_idx].set(rebased.eta).reshape(n_var, n_port, d_op)
        new_Lam = jnp.broadcast_to(filler.Lam[:, None, :, :], (n_var, n_port, d_op, d_op)).reshape(n_slot, d_op, d_op).at[slot_idx].set(rebased.Lam).reshape(n_var, n_port, d_op, d_op)
        new_msgs = Gaussian(new_eta, new_Lam)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@partial(jax.jit, static_argnames=['alpha'])
def update_variable_SE3D_keepDepth(vars, old_linpoints, alpha: float=1.0):
    """
    Retract/rebase SE(3) only, keeping the depth chart (center) fixed.
    """
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SE3_keepDepth, in_axes=(0, 0, None))(old_linpoints, old_posterior, alpha)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(Rebase_SE3_keepDepth, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@partial(jax.jit, static_argnames=['alpha'])
def update_variable_D(vars, old_linpoints, alpha: float=1.0):
    """
    Depth only update.
    """
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_D, in_axes=(0, 0, None))(old_linpoints, old_posterior, alpha)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(Rebase_D, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@partial(jax.jit, static_argnames=['alpha'])
def update_variable_twist_SE3(vars, old_linpoints, alpha: float=1.0):
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SE3, in_axes=(0, 0, None))(old_linpoints, old_posterior, alpha)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(Rebase_SE3, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@jax.jit
def update_variable_translation_SE3(vars, old_linpoints):
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    old_linpoints = jnp.concatenate([jnp.zeros(old_linpoints.shape, dtype=old_linpoints.dtype), old_linpoints], axis=-1)
    old_posterior.eta = jnp.concatenate([jnp.zeros(old_posterior.eta.shape, dtype=old_posterior.eta.dtype), old_posterior.eta], axis=-1)
    Lam_eye_repeated = jnp.tile(jnp.eye(old_posterior.Lam.shape[-1], dtype=old_posterior.Lam.dtype)[None, ...], (old_posterior.Lam.shape[0], 1, 1))
    Lam_zero_repeated = jnp.tile(jnp.zeros(old_posterior.Lam.shape[-2:], dtype=old_posterior.Lam.dtype)[None, ...], (old_posterior.Lam.shape[0], 1, 1))
    old_posterior.Lam = jnp.concatenate([jnp.concatenate([Lam_eye_repeated, Lam_zero_repeated], axis=-1), jnp.concatenate([Lam_zero_repeated, old_posterior.Lam], axis=-1)], axis=-2)
    (new_linpoints, new_belief) = jax.vmap(Retract_SE3, in_axes=(0, 0))(old_linpoints, old_posterior)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(RebaseSafe_SE3, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    Lam_eye_repeated = jnp.tile(Lam_eye_repeated[:, None, ...], (1, vars.msgs.Lam.shape[1], 1, 1))
    Lam_zero_repeated = jnp.tile(Lam_zero_repeated[:, None, ...], (1, vars.msgs.Lam.shape[1], 1, 1))
    vars.msgs.eta = jnp.concatenate([jnp.zeros(vars.msgs.eta.shape, dtype=vars.msgs.eta.dtype), vars.msgs.eta], axis=-1)
    vars.msgs.Lam = jnp.concatenate([jnp.concatenate([Lam_eye_repeated, Lam_zero_repeated], axis=-1), jnp.concatenate([Lam_zero_repeated, vars.msgs.Lam], axis=-1)], axis=-2)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.msgs.eta = vars.msgs.eta[..., 3:6]
    vars.msgs.Lam = vars.msgs.Lam[..., 3:6, 3:6]
    new_msgs.eta = new_msgs.eta[..., 3:6]
    new_msgs.Lam = new_msgs.Lam[..., 3:6, 3:6]
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief.eta = new_belief.eta[..., 3:6]
    vars.belief.Lam = new_belief.Lam[..., 3:6, 3:6]
    return (vars, vtof_msgs, new_linpoints[..., 3:6])

@partial(jax.jit, static_argnames=['alpha'])
def update_variable_SO3R3(vars, old_linpoints, alpha: float=1.0):
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SO3R3, in_axes=(0, 0, None))(old_linpoints, old_posterior, alpha)

    def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
        return jax.vmap(RebaseSafe_SO3R3, in_axes=(None, None, 0))(new_lin, old_lin, msgs_g)
    new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

@register_pytree_node_class
class Gaussian:

    def __init__(self, eta, Lam):
        self.eta = eta
        self.Lam = Lam

    def tree_flatten(self):
        return ((self.eta, self.Lam), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def mu(self):
        is_zero = jnp.allclose(self.Lam, 0)

        def zero_branch(_):
            return jnp.asarray(self.eta, dtype=self.Lam.dtype)

        def solve_branch(_):
            diag = jnp.diag(self.Lam)
            scale = jnp.maximum(jnp.mean(jnp.abs(diag)), jnp.asarray(1e-12, dtype=self.Lam.dtype))
            lam_damp = jnp.maximum(jnp.asarray(1e-08, dtype=self.Lam.dtype), jnp.asarray(0.001, dtype=self.Lam.dtype) * scale)
            n = self.Lam.shape[-1]
            A = self.Lam + lam_damp * jnp.eye(n, dtype=self.Lam.dtype)
            b = jnp.asarray(self.eta, dtype=self.Lam.dtype)
            return jnp.linalg.solve(A, b)
        return jax.lax.cond(is_zero, zero_branch, solve_branch, operand=None)

    def sigma(self):
        diag = jnp.diag(self.Lam)
        scale = jnp.maximum(jnp.mean(jnp.abs(diag)), jnp.asarray(1e-12, dtype=self.Lam.dtype))
        lam_damp = jnp.maximum(jnp.asarray(1e-08, dtype=self.Lam.dtype), jnp.asarray(0.001, dtype=self.Lam.dtype) * scale)
        n = self.Lam.shape[-1]
        return jnp.linalg.inv(self.Lam + lam_damp * jnp.eye(n, dtype=self.Lam.dtype))

    def zero_like(self):
        return Gaussian(jnp.zeros_like(self.eta), jnp.zeros_like(self.Lam))

    def __repr__(self) -> str:
        return f'Gaussian(eta={self.eta}, lam={self.Lam})'

    def __mul__(self, other):
        return Gaussian(self.eta + other.eta, self.Lam + other.Lam)

    def __truediv__(self, other):
        return Gaussian(self.eta - other.eta, self.Lam - other.Lam)

    def copy(self):
        return Gaussian(self.eta.copy(), self.Lam.copy())

@register_pytree_node_class
class Variable:
    var_id: int
    belief: Gaussian
    msgs: Gaussian
    adj_factor_idx: jnp.array

    def __init__(self, var_id, belief, msgs, adj_factor_idx):
        self.var_id = var_id
        self.belief = belief
        self.msgs = msgs
        self.adj_factor_idx = adj_factor_idx

    def tree_flatten(self):
        return ((self.var_id, self.belief, self.msgs, self.adj_factor_idx), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Factor:
    factor_id: jnp.array
    z: jnp.ndarray
    z_Lam: jnp.ndarray
    threshold: jnp.ndarray
    potential: Gaussian
    adj_var_id: jnp.array
    adj_var_idx: jnp.array

    def __init__(self, factor_id, z, z_Lam, threshold, potential, adj_var_id, adj_var_idx):
        self.factor_id = factor_id
        self.z = z
        self.z_Lam = z_Lam
        self.threshold = threshold
        self.potential = potential
        self.adj_var_id = adj_var_id
        self.adj_var_idx = adj_var_idx

    def tree_flatten(self):
        return ((self.factor_id, self.z, self.z_Lam, self.threshold, self.potential, self.adj_var_id, self.adj_var_idx), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@partial(jax.jit, static_argnames=['i', 'j'])
def marginalize(gaussians: Gaussian, i, j):
    eta = gaussians.eta
    Lam = gaussians.Lam
    k = eta.size
    idx = jnp.arange(0, k)
    aa = idx[i:j]
    bb = jnp.concatenate([idx[:i], idx[j:]])
    aa_eta = eta[aa]
    bb_eta = eta[bb]
    aa_Lam = Lam[aa[:, None], aa]
    ab_Lam = Lam[aa[:, None], bb]
    bb_Lam = Lam[bb][:, bb]
    if bb_Lam.size == 0:
        return Gaussian(aa_eta, aa_Lam)
    diag = jnp.diag(bb_Lam)
    scale = jnp.maximum(jnp.mean(jnp.abs(diag)), 1e-12)
    lam_damp = jnp.maximum(1e-08, 0.001 * scale)
    bb_reg = bb_Lam + lam_damp * jnp.eye(bb_Lam.shape[0], dtype=bb_Lam.dtype)
    X = jnp.linalg.solve(bb_reg, ab_Lam.T)
    y = jnp.linalg.solve(bb_reg, bb_eta)
    eta = aa_eta - ab_Lam @ y
    Lam = aa_Lam - ab_Lam @ X
    return Gaussian(eta, Lam)

def tree_stack(tree, axis=0, use_np=True):
    if use_np:
        return jax.tree.map(lambda *v: jnp.array(np.stack(v, axis=axis)), *tree)
    return jax.tree.map(lambda *v: jnp.stack(v, axis=axis), *tree)

def h_fn(x):
    return x

def h2_fn(x):
    x1 = x[0]
    x2 = x[1]
    return x2 - x1

def update_belief(var: Variable, ftov_msgs):
    belief = var.belief.zero_like()
    for i in range(ftov_msgs.eta.shape[0]):
        belief = belief * Gaussian(ftov_msgs.eta[i], ftov_msgs.Lam[i])
    return belief

def compute_vtof_msgs(var: Variable, ftov_msgs):
    vtof_msgs = []
    for (i, idx) in enumerate(var.adj_factor_idx):
        msg = var.belief / Gaussian(ftov_msgs.eta[i], ftov_msgs.Lam[i])
        eta = jnp.where(idx < 0, msg.zero_like().eta, msg.eta)
        Lam = jnp.where(idx < 0, msg.zero_like().Lam, msg.Lam)
        vtof_msgs.append(Gaussian(eta, Lam))
    return tree_stack(vtof_msgs, use_np=False)

@partial(jax.jit, static_argnames=['h_fn'])
def factor_energy(factor, xs, h_fn):
    h = h_fn(xs)
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h
    return 0.5 * r @ z_Lam @ r.T

@partial(jax.jit, static_argnames=['h_fn'])
def factor_mahalanobis(factor, xs, h_fn):
    h = h_fn(xs)
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h
    return (r @ z_Lam @ r.T) ** 0.5

@partial(jax.jit, static_argnames=['h_fn'])
def factor_mahalanobis_split(factor, xs, h_fn, split_index: int=3):
    """
    Compute Mahalanobis distances for two partitions of the residual vector.

    Args:
        factor: Factor carrying z and z_Lam.
        xs: Variable(s) the factor connects to, concatenated as expected by h_fn.
        h_fn: Factor function mapping xs -> measurement space.
        split_index: Dimension at which to split residual r into r[:k] and r[k:].

    Returns:
        A pair (d_front, d_back) of Mahalanobis distances corresponding to the
        front block [0:split_index) and the back block [split_index:]).
    """
    h = h_fn(xs)
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h
    k = split_index
    r_front = r[:k]
    r_back = r[k:]
    Lam_ff = z_Lam[:k, :k]
    Lam_bb = z_Lam[k:, k:]
    d_front = jnp.sqrt(jnp.maximum(r_front @ Lam_ff @ r_front.T, 0.0))
    d_back = jnp.sqrt(jnp.maximum(r_back @ Lam_bb @ r_back.T, 0.0))
    return (d_front, d_back)

def factor_update_manifold_hJ(factor, xs, h_fn, w):
    (h, J) = h_fn(xs)
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h.reshape(-1)
    if factor.threshold.size == 1:
        s = w(r.T @ z_Lam @ r, factor.threshold)
        Lam = s * J.T @ z_Lam @ J
        eta = s * J.T @ z_Lam @ r
    else:
        z_Lam_diag = jnp.diag(z_Lam)
        errors = z_Lam_diag * r ** 2
        s_vec = w(errors, factor.threshold.reshape(-1))
        S = jnp.diag(s_vec)
        Lam = J.T @ S @ z_Lam @ J
        eta = J.T @ S @ z_Lam @ r
    return Gaussian(eta, Lam)

def factor_update_prior_manifold(factor, xs, w):
    """Ablation (DANCERS_PRIOR=1): dancers-style manifold prior potential.

    dancers Variable.cpp applies its built-in prior as residual tau = prior_mu (-) X with
    J = -Jl^{-1}(tau); the identical Gauss-Newton written right-handed is
      rho = X (-) prior = Log(Exp(-z) Exp(x)),  J = Jr^{-1}(rho)
    (equivalence verified in dancers_equiv_tests.py T7). The pose block (first 6 dofs) is the
    manifold residual; remaining dofs (depth) stay linear, exactly like h3_fn's identity model.
    """
    x = xs.reshape(-1)
    n = x.shape[0]
    z = factor.z
    z_Lam = factor.z_Lam
    rho6 = se3_geo_dist(z[:6], x[:6])
    J6 = se3_Jr_inv(rho6)
    if n > 6:
        rho = jnp.concatenate([rho6, x[6:] - z[6:]], axis=0)
        J = jnp.eye(n, dtype=x.dtype).at[:6, :6].set(J6)
    else:
        rho = rho6
        J = J6
    if factor.threshold.size == 1:
        s = w(rho.T @ z_Lam @ rho, factor.threshold)
        JtL = J.T @ z_Lam
        Lam = s * (JtL @ J)
        eta = -s * (JtL @ rho)
    else:
        z_Lam_diag = jnp.diag(z_Lam)
        errors = z_Lam_diag * rho ** 2
        s_vec = w(errors, factor.threshold.reshape(-1))
        S = jnp.diag(s_vec)
        Lam = J.T @ S @ z_Lam @ J
        eta = -(J.T @ S @ z_Lam @ rho)
    return Gaussian(eta, Lam)

@partial(jax.jit, static_argnames=['w'])
def update_factor_prior_manifold(facs, vars, vtof_msgs, linpoints, w):
    """Prior-factor sweep with the dancers-style manifold potential (unary factors; the
    message/factor dims match, so no padding path is needed)."""
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    facs.potential = jax.vmap(factor_update_prior_manifold, in_axes=(0, 0, None))(facs, linpoints_reordered, w)
    ftov_msgs = jax.vmap(compute_ftov_msg)(facs, vtof_msgs_reordered)
    vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.eta)
    vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)

def compute_ftov_msg(factor, vtof_msgs):
    (N_adj, dim) = vtof_msgs.eta.shape
    if _UNARY_EXACT and N_adj == 1 and factor.potential.eta.shape[-1] == dim:
        return tree_stack([Gaussian(factor.potential.eta, factor.potential.Lam)], use_np=False)
    pot = factor.potential.copy()
    i = 0
    for n in range(N_adj):
        j = i + dim
        pot.eta = pot.eta.at[i:j].add(vtof_msgs.eta[n])
        pot.Lam = pot.Lam.at[i:j, i:j].add(vtof_msgs.Lam[n])
        i = j
    ftov_msgs = []
    i = 0
    for n in range(N_adj):
        j = i + dim
        pot_m_1 = pot.copy()
        pot_m_1.eta = pot_m_1.eta.at[i:j].add(-vtof_msgs.eta[n])
        pot_m_1.Lam = pot_m_1.Lam.at[i:j, i:j].add(-vtof_msgs.Lam[n])
        msg = marginalize(pot_m_1, i, j)
        ftov_msgs.append(msg)
        i = j
    return tree_stack(ftov_msgs, use_np=False)

@jax.jit
def update_variable(vars):
    vars.belief = jax.vmap(update_belief)(vars, vars.msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs)(vars, vars.msgs)
    linpoints = jax.vmap(lambda x: x.mu())(vars.belief)
    return (vars, vtof_msgs, linpoints)

@jax.jit
def update_variable_twist(vars):
    old_f2v_msgs = jax.vmap(update_belief)(vars, vars.msgs)
    old_linpoints = jax.vmap(lambda x: x.mu())(vars.belief)
    new_belief = jax.vmap(Retract_SE3, in_axes=(0, 0))(old_linpoints, old_f2v_msgs)
    new_eta = jax.vmap(lambda Lam, mu: Lam @ mu)(new_belief.Lam, new_belief.eta)
    vars.belief = Gaussian(new_eta, new_belief.Lam)
    new_linpoints = jax.vmap(lambda x: x.mu())(new_belief)
    vars.msgs = jax.vmap(Rebase_SE3, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, old_f2v_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, vars.msgs)
    return (vars, vtof_msgs, new_linpoints)

def safe_mu(eta, Lam):
    diag = jnp.diag(Lam)
    off = Lam - jnp.diag(diag)
    is_zero = jnp.all(Lam == 0)
    is_diag = jnp.all(jnp.abs(off) <= _EPS)

    def zero_branch(_):
        return eta

    def nonzero_branch(_):

        def diag_branch(__):
            denom = jnp.where(jnp.abs(diag) > _EPS, diag, 1.0)
            x = eta / denom
            x = jnp.where(jnp.abs(diag) > _EPS, x, 0.0)
            return x

        def dense_branch(__):
            n = Lam.shape[-1]
            return jnp.linalg.solve(Lam + _EPS * jnp.eye(n, dtype=Lam.dtype), eta)
        return jax.lax.cond(is_diag, diag_branch, dense_branch, operand=None)
    return jax.lax.cond(is_zero, zero_branch, nonzero_branch, operand=None)

@jax.jit
def Retract_SO3(linpoint: jnp.ndarray, m, alpha: float=1.0):
    """
    Retract SO(3)
    linpoint : jnp.array(3,)
    m : Gaussian
    alpha : float
    """
    tau = alpha * safe_mu(m.eta, m.Lam)
    mean = so3_geo_dist(-linpoint, tau)
    Jrinv = so3_Jr_inv(tau)
    Lam = Jrinv.T @ m.Lam @ Jrinv
    belief = Gaussian(Lam @ mean, Lam)
    return (mean, belief)

@jax.jit
def Retract_SE3(linpoint: jnp.ndarray, m: Gaussian, alpha: float=1.0) -> Gaussian:
    """
    Retract SE(3)
    linpoint : jnp.array(6,)    -> [rotation, translation]
    tau : jnp.array(6,)    -> [rotation, translation]
    alpha : float    -> alpha
    """
    tau = alpha * safe_mu(m.eta, m.Lam)
    mean = se3_geo_dist(-linpoint, tau)
    Jrinv = se3_Jr_inv(tau)
    Lam = Jrinv.T @ m.Lam @ Jrinv
    belief = Gaussian(Lam @ mean, Lam)
    return (mean, belief)

@partial(jax.jit, static_argnames=['alpha'])
def Retract_D(linpoint: jnp.ndarray, m: Gaussian, alpha: float=1.0) -> Gaussian:
    """
    Retract Depth only
    linpoint : jnp.array(1,)
    m : Gaussian
    alpha : float
    """
    tau = alpha * safe_mu(m.eta, m.Lam)
    mean = linpoint + tau
    Lam = m.Lam
    belief = Gaussian(Lam @ mean, Lam)
    return (mean, belief)

@partial(jax.jit, static_argnames=['alpha'])
def Retract_SO3R3(linpoint: jnp.ndarray, m: Gaussian, alpha: float=1.0) -> Gaussian:
    """
    Retract SO(3), R(3)
    linpoint : jnp.array(6,)    -> [so3(3), r(3)]
    m : Gaussian
    alpha : float
    """
    tau = alpha * safe_mu(m.eta, m.Lam)
    mean_rot = so3_geo_dist(-linpoint[:3], tau[:3])
    Jrinv = so3_Jr_inv(tau[:3])
    Lam_rot = Jrinv.T @ m.Lam[:3, :3] @ Jrinv
    mean_tr = linpoint[3:] + tau[3:]
    mean = jnp.concatenate([mean_rot, mean_tr], axis=-1)
    Lam = jnp.block([[Lam_rot, jnp.zeros((3, 3))], [jnp.zeros((3, 3)), m.Lam[3:, 3:]]])
    belief = Gaussian(Lam @ mean, Lam)
    return (mean, belief)

@jax.jit
def Rebase_SO3(new_linpoint, old_linpoint, old_m):
    tau = safe_mu(old_m.eta, old_m.Lam)
    global_mean = so3_geo_dist(-old_linpoint, tau)
    Jretract = so3_Jr_inv(tau)
    delta = so3_geo_dist(new_linpoint, global_mean)
    Jdelta = so3_Jr(delta)
    J = Jretract @ Jdelta
    Lam = J.T @ old_m.Lam @ J
    return Gaussian(Lam @ delta, Lam)

@jax.jit
def RebaseSafe_SO3(new_linpoint, old_linpoint, old_m, std=1.0):
    P = Gaussian(jnp.zeros_like(old_m.eta), jnp.eye(old_m.Lam.shape[-1]) / std)
    return Rebase_SO3(new_linpoint, old_linpoint, old_m * P)

@jax.jit
def Rebase_SO3R3(new_linpoint, old_linpoint, old_m):
    tau = safe_mu(old_m.eta, old_m.Lam)
    global_mean_rot = so3_geo_dist(-old_linpoint[:3], tau[:3])
    Jretract = so3_Jr_inv(tau[:3])
    delta_rot = so3_geo_dist(new_linpoint[:3], global_mean_rot)
    Jdelta = so3_Jr(delta_rot)
    J_rot = Jretract @ Jdelta
    global_mean_tr = old_linpoint[3:] + tau[3:]
    delta_tr = global_mean_tr - new_linpoint[3:]
    delta = jnp.concatenate([delta_rot, delta_tr], axis=-1)
    Lam = jnp.block([[J_rot.T @ old_m.Lam[:3, :3] @ J_rot, jnp.zeros((3, 3))], [jnp.zeros((3, 3)), old_m.Lam[3:, 3:]]])
    return Gaussian(Lam @ delta, Lam)

@jax.jit
def RebaseSafe_SO3R3(new_linpoint, old_linpoint, old_m, std=1.0):
    P = Gaussian(jnp.zeros_like(old_m.eta), jnp.eye(old_m.Lam.shape[-1]) / std)
    return Rebase_SO3R3(new_linpoint, old_linpoint, old_m * P)

@jax.jit
def Rebase_SE3(new_linpoint, old_linpoint, old_m):
    tau = safe_mu(old_m.eta, old_m.Lam)
    global_mean = se3_geo_dist(-old_linpoint, tau)
    Jretract = se3_Jr_inv(tau)
    delta = se3_geo_dist(new_linpoint, global_mean)
    Jdelta = se3_Jr(delta)
    J = Jretract @ Jdelta
    Lam = J.T @ old_m.Lam @ J
    return Gaussian(Lam @ delta, Lam)

@jax.jit
def RebaseSafe_SE3(new_linpoint, old_linpoint, old_m, std=1.0):
    P = Gaussian(jnp.zeros_like(old_m.eta), jnp.eye(old_m.Lam.shape[-1]) / std)
    return Rebase_SE3(new_linpoint, old_linpoint, old_m * P)

@jax.jit
def Rebase_D(new_linpoint, old_linpoint, old_m):
    tau = safe_mu(old_m.eta, old_m.Lam)
    global_mean = old_linpoint + tau
    Jretract = jnp.eye(1)
    delta = -new_linpoint + global_mean
    Jdelta = jnp.eye(1)
    J = Jretract @ Jdelta
    Lam = J.T @ old_m.Lam @ J
    return Gaussian(Lam @ delta, Lam)

def _split_se3_depth(x7):
    return (x7[:6], x7[6])

def _stack_se3_depth(xi6, d):
    return jnp.concatenate([xi6, jnp.array([d])], axis=0)

def _blkdiag_se3_scalar(A6, s11=1.0):
    return jnp.block([[A6, jnp.zeros((6, 1))], [jnp.zeros((1, 6)), jnp.array([[s11]])]])

def _safe_mu(eta: jnp.ndarray, Lam: jnp.ndarray) -> jnp.ndarray:
    """Numerically safe mean extraction for canonical Gaussian (eta, Lam).
    - If Lam is (near-)diagonal, use element-wise division with eps guards.
    - Else, solve with small Tikhonov regularization.
    """
    diag = jnp.diag(Lam)
    off = Lam - jnp.diag(diag)
    is_diag = jnp.all(jnp.abs(off) <= 1e-12)

    def diag_branch(_):
        eps = 1e-12
        denom = jnp.where(jnp.abs(diag) > eps, diag, 1.0)
        x = eta / denom
        x = jnp.where(jnp.abs(diag) > eps, x, 0.0)
        return x

    def dense_branch(_):
        eps = 1e-06
        n = Lam.shape[-1]
        return jnp.linalg.solve(Lam + eps * jnp.eye(n, dtype=Lam.dtype), eta)
    return jax.lax.cond(is_diag, diag_branch, dense_branch, operand=None)

def _safe_mu_np(eta: np.ndarray, Lam: np.ndarray) -> np.ndarray:
    diag = np.diag(Lam)
    off = Lam - np.diag(diag)
    is_diag = np.all(np.abs(off) <= 1e-12)
    if is_diag:
        eps = 1e-12
        denom = np.where(np.abs(diag) > eps, diag, 1.0)
        x = eta / denom
        x = np.where(np.abs(diag) > eps, x, 0.0)
        return x
    else:
        eps = 1e-06
        n = Lam.shape[-1]
        return np.linalg.solve(Lam + eps * np.eye(n, dtype=Lam.dtype), eta)

@jax.jit
def Retract_SE3_plus_depth(linpoint7: jnp.ndarray, m: Gaussian, alpha: float=1.0):
    """
    linpoint7: (7,) = [so3(3), t(3), depth(1)]  (= [rotation, translation, depth])
    m: Gaussian or PyTree of Gaussians s.t. vmap(lambda x: x.mu())(m) -> (7,)
    """
    tau7 = alpha * safe_mu(m.eta, m.Lam)
    (tau6, tau_d) = _split_se3_depth(tau7)
    (lin6, lin_d) = _split_se3_depth(linpoint7)
    mean6 = se3_geo_dist(-lin6, tau6)
    mean_d = lin_d + tau_d
    mean7 = _stack_se3_depth(mean6, mean_d)
    Jrinv6 = se3_Jr_inv(tau6)
    Jrinv7 = _blkdiag_se3_scalar(Jrinv6, 1.0)
    Lam_new = _mm(_mm(Jrinv7.T, m.Lam), Jrinv7)
    belief = Gaussian(_mv(Lam_new, mean7), Lam_new)
    return (mean7, belief)

@jax.jit
def Retract_SE3_keepDepth(linpoint7: jnp.ndarray, m: Gaussian, alpha: float=1.0):
    """
    Re-linearize (retract) SE(3) only, keeping the depth chart (center) fixed;
    that is, mean_d stays equal to lin_d.
    """
    tau7 = alpha * safe_mu(m.eta, m.Lam)
    (tau6, _tau_d) = _split_se3_depth(tau7)
    (lin6, lin_d) = _split_se3_depth(linpoint7)
    mean6 = se3_geo_dist(-lin6, tau6)
    mean_d = lin_d
    mean7 = _stack_se3_depth(mean6, mean_d)
    Jrinv6 = se3_Jr_inv(tau6)
    Jrinv7 = _blkdiag_se3_scalar(Jrinv6, 1.0)
    Lam_new = Jrinv7.T @ m.Lam @ Jrinv7
    belief = Gaussian(Lam_new @ mean7, Lam_new)
    return (mean7, belief)

@jax.jit
def Rebase_SE3_plus_depth(new_linpoint7: jnp.ndarray, old_linpoint7: jnp.ndarray, old_m: Gaussian):
    """
    Rebase from chart at old_linpoint7 to chart at new_linpoint7 on SE(3)×R.
    """
    tau7 = safe_mu(old_m.eta, old_m.Lam)
    (tau6, tau_d) = _split_se3_depth(tau7)
    (old_lin6, old_lin_d) = _split_se3_depth(old_linpoint7)
    (new_lin6, new_lin_d) = _split_se3_depth(new_linpoint7)
    global_se3 = se3_geo_dist(-old_lin6, tau6)
    global_d = old_lin_d + tau_d
    Jretract6 = se3_Jr_inv(tau6)
    Jretract7 = _blkdiag_se3_scalar(Jretract6, 1.0)
    delta6 = se3_geo_dist(new_lin6, global_se3)
    delta_d = global_d - new_lin_d
    delta7 = _stack_se3_depth(delta6, delta_d)
    Jdelta6 = se3_Jr(delta6)
    Jdelta7 = _blkdiag_se3_scalar(Jdelta6, 1.0)
    J = _mm(Jretract7, Jdelta7)
    Lam_new = _mm(_mm(J.T, old_m.Lam), J)
    return Gaussian(_mv(Lam_new, delta7), Lam_new)

@jax.jit
def Rebase_SE3_keepDepth(new_linpoint7: jnp.ndarray, old_linpoint7: jnp.ndarray, old_m: Gaussian):
    """
    Rebase SE(3) only, keeping the depth chart fixed.
    (Designed for the case new_lin_d == old_lin_d, but also works safely otherwise.)
    """
    tau7 = safe_mu(old_m.eta, old_m.Lam)
    (tau6, tau_d) = _split_se3_depth(tau7)
    (old_lin6, old_lin_d) = _split_se3_depth(old_linpoint7)
    (new_lin6, new_lin_d) = _split_se3_depth(new_linpoint7)
    global_se3 = se3_geo_dist(-old_lin6, tau6)
    global_d = old_lin_d + tau_d
    Jretract6 = se3_Jr_inv(tau6)
    Jretract7 = _blkdiag_se3_scalar(Jretract6, 1.0)
    delta6 = se3_geo_dist(new_lin6, global_se3)
    delta_d = global_d - new_lin_d
    delta7 = _stack_se3_depth(delta6, delta_d)
    Jdelta6 = se3_Jr(delta6)
    Jdelta7 = _blkdiag_se3_scalar(Jdelta6, 1.0)
    J = Jretract7 @ Jdelta7
    Lam_new = J.T @ old_m.Lam @ J
    return Gaussian(Lam_new @ delta7, Lam_new)

def RebaseSafe_SE3_plus_depth(new_linpoint7: jnp.ndarray, old_linpoint7: jnp.ndarray, old_m: Gaussian, std=1.0):
    P = Gaussian(jnp.zeros_like(old_m.eta), jnp.eye(old_m.Lam.shape[-1]) / std)
    return Rebase_SE3_plus_depth(new_linpoint7, old_linpoint7, old_m * P)

@partial(jax.jit, static_argnames=['f', 'w'], donate_argnames=_DON)
def update_factor_manifold_hJ(facs, vars, vtof_msgs, linpoints, f, w, mask_ftov=None):
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    msg_dim = vtof_msgs_reordered.eta.shape[-1]
    factor_dim = linpoints_reordered.shape[-1]
    vtof_msgs_for_compute = vtof_msgs_reordered
    if factor_dim != msg_dim:
        vtof_msgs_for_compute = Gaussian(eta=vtof_msgs_reordered.eta[..., :factor_dim], Lam=vtof_msgs_reordered.Lam[..., :factor_dim, :factor_dim])
    facs.potential = jax.vmap(factor_update_manifold_hJ, in_axes=(0, 0, None, None))(facs, linpoints_reordered, f, w)
    ftov_msgs = jax.vmap(compute_ftov_msg)(facs, vtof_msgs_for_compute)
    if factor_dim != msg_dim:
        target_dim = msg_dim
        padded_eta = jnp.zeros(ftov_msgs.eta.shape[:-1] + (target_dim,))
        padded_eta = padded_eta.at[..., :factor_dim].set(ftov_msgs.eta)
        padded_Lam = jnp.zeros(ftov_msgs.Lam.shape[:-2] + (target_dim, target_dim))
        padded_Lam = padded_Lam.at[..., :factor_dim, :factor_dim].set(ftov_msgs.Lam)
        ftov_msgs = Gaussian(padded_eta, padded_Lam)
    if mask_ftov is not None:
        target_eta = vars.msgs.eta[facs.adj_var_id, facs.adj_var_idx]
        target_Lam = vars.msgs.Lam[facs.adj_var_id, facs.adj_var_idx]
        mask_eta = mask_ftov[..., None]
        mask_Lam = mask_ftov[..., None, None]
        blended_eta = jnp.where(mask_eta, ftov_msgs.eta, target_eta)
        blended_Lam = jnp.where(mask_Lam, ftov_msgs.Lam, target_Lam)
        vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(blended_eta)
        vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(blended_Lam)
    else:
        vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.eta)
        vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)

@jax.jit
def huber(e, t):
    x = jnp.sqrt(e)
    return jnp.where(x <= t, 1.0, t / x)

@jax.jit
def tukey(e, t):
    """
    Tukey biweight weighting function for robust regression.
    
    This implements the weighting function w(r) for the Tukey biweight
    (also known as bisquare) M-estimator:
    
    w(r) = (1 - (r/c)^2)^2   if |r| <= c
    w(r) = 0                 if |r| > c
    
    where r is the residual and c is the tuning constant.
    
    References:
    - Beaton, A. E., & Tukey, J. W. (1974). The fitting of power series, 
      meaning polynomials, illustrated on band-spectroscopic data. 
      Technometrics, 16(2), 147-185.
    - Holland, P. W., & Welsch, R. E. (1977). Robust regression using 
      iteratively reweighted least-squares. Communications in Statistics-
      Theory and Methods, 6(9), 813-827.
    - Huber, P. J., & Ronchetti, E. M. (2009). Robust statistics. 
      John Wiley & Sons.
    
    Args:
        e: Squared residual value(s)
        t: Tuning constant (threshold parameter)
        
    Returns:
        Weight value(s) according to Tukey biweight function
    """
    x = jnp.sqrt(e)
    ratio = x / t
    ratio_sq = ratio * ratio
    weight = jnp.where(x <= t, (1.0 - ratio_sq) * (1.0 - ratio_sq), 0.0)
    return weight

@jax.jit
def l2(e, _):
    return 1.0

def build_default(img):
    (H, W, D) = img.shape
    N = H * W
    Ni_v = 6
    gpu_device = jax.devices('gpu')[0]
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        z_Lam = np.eye(D)
        tic = time.time()
        belief = Gaussian(np.ones(D), np.eye(D))
        msgs = Gaussian(np.zeros((Ni_v, D)), np.zeros((Ni_v, D, D)))
        adj_factor_idx = -np.ones(Ni_v, dtype=np.int32)
        vars = [Variable(i, belief, msgs, adj_factor_idx.copy()) for i in range(N)]
        tok = time.time()
        print('vars', tok - tic)
        tic = time.time()
        prior_facs = []
        for i in range(N):
            f_id = i
            v_id = i
            idx = np.argmax(vars[v_id].adj_factor_idx == -1)
            vars[v_id].adj_factor_idx[idx] = f_id
            adj_var_id = np.array([v_id], dtype=np.int32)
            adj_var_idx = np.array([idx], dtype=np.int32)
            z = img[i // W, i % W]
            factor = Factor(i, z, z_Lam * 0.2, 1, None, adj_var_id, adj_var_idx)
            prior_facs.append(factor)
        between_facs = []
        next_id = N
        for i in range(H):
            for j in range(W):
                if j < W - 1:
                    f_id = next_id
                    next_id += 1
                    v1_id = i * W + j
                    v2_id = i * W + j + 1
                    idx_1 = np.argmax(vars[v1_id].adj_factor_idx == -1)
                    vars[v1_id].adj_factor_idx[idx_1] = f_id
                    idx_2 = np.argmax(vars[v2_id].adj_factor_idx == -1)
                    vars[v2_id].adj_factor_idx[idx_2] = f_id
                    adj_var_id = np.array([v1_id, v2_id], dtype=np.int32)
                    adj_var_idx = np.array([idx_1, idx_2], dtype=np.int32)
                    factor = Factor(f_id, np.zeros((D,)), 1, z_Lam, None, adj_var_id, adj_var_idx)
                    between_facs.append(factor)
                if i < H - 1:
                    f_id = next_id
                    next_id += 1
                    v1_id = i * W + j
                    v2_id = (i + 1) * W + j
                    idx_1 = np.argmax(vars[v1_id].adj_factor_idx == -1)
                    vars[v1_id].adj_factor_idx[idx_1] = f_id
                    idx_2 = np.argmax(vars[v2_id].adj_factor_idx == -1)
                    vars[v2_id].adj_factor_idx[idx_2] = f_id
                    adj_var_id = np.array([v1_id, v2_id], dtype=np.int32)
                    adj_var_idx = np.array([idx_1, idx_2], dtype=np.int32)
                    factor = Factor(f_id, np.zeros((D,)), z_Lam, 1, None, adj_var_id, adj_var_idx)
                    between_facs.append(factor)
        tok = time.time()
        print('facs', tok - tic)
    tic = time.time()
    vars = tree_stack(vars)
    prior_facs = tree_stack(prior_facs)
    between_facs = tree_stack(between_facs)
    print('factor_id', between_facs.factor_id.shape)
    print('adj_var_id', between_facs.adj_var_id.shape)
    print('adj_var_idx', between_facs.adj_var_idx.shape)
    tok = time.time()
    print('stack', tok - tic)
    return (vars, prior_facs, between_facs)

def build_2D(img):
    (H, W, D) = img.shape
    N = H * W
    Ni_v = 5 + 1
    gpu_device = jax.devices('gpu')[0]
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        tic = time.time()
        var_ids = np.arange(0, N, dtype=np.int32)
        belief = Gaussian(np.zeros((N, D)), np.tile(np.eye(D), (N, 1, 1)))
        msgs = Gaussian(np.zeros((N, Ni_v, D)), np.zeros((N, Ni_v, D, D)))
        adj_factor_idx = -np.ones((N, Ni_v), dtype=np.int32)
        vars = Variable(var_ids, belief, msgs, adj_factor_idx)
        tok = time.time()
        print('vars', tok - tic)
        tic = time.time()
        prior_facs = []
        z_Lam = np.eye(D)
        factor_ids = np.arange(0, N, dtype=np.int32)
        zs = img.reshape(-1, D)
        z_Lams = np.tile(z_Lam * 0.2, (N, 1, 1))
        threshold = np.ones(N).reshape(-1, 1)
        potential = None
        adj_var_id = np.arange(0, N, dtype=np.int32).reshape(-1, 1)
        adj_var_idx = np.zeros(N, dtype=np.int32).reshape(-1, 1)
        prior_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)

        def horizontal(i, j, H, W, N):
            factor_id = N + i * W + j
            v1_id = i * W + j
            v2_id = i * W + j + 1
            idx_1 = 1
            idx_2 = 2
            return (factor_id, v1_id, v2_id, idx_1, idx_2)

        def verticle(i, j, H, W, N):
            factor_id = 2 * N + i * W + j
            v1_id = i * W + j
            v2_id = (i + 1) * W + j
            idx_1 = 3
            idx_2 = 4
            return (factor_id, v1_id, v2_id, idx_1, idx_2)
        (ii, jj) = np.meshgrid(np.arange(H), np.arange(W - 1), indexing='xy')
        ii = ii.flatten()
        jj = jj.flatten()
        (factor_ids, v1_ids, v2_ids, idx_1s, idx_2s) = jax.vmap(horizontal, in_axes=(0, 0, None, None, None))(ii, jj, H, W, N)
        adj_var_id = np.stack([v1_ids, v2_ids], axis=1)
        adj_var_idx = np.stack([idx_1s, idx_2s], axis=1)
        zs = np.zeros((factor_ids.size, D))
        z_Lams = np.tile(z_Lam, (factor_ids.size, 1, 1))
        threshold = np.ones((factor_ids.size, 1))
        potential = None
        h_between_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
        (ii, jj) = np.meshgrid(np.arange(H - 1), np.arange(W), indexing='ij')
        ii = ii.flatten()
        jj = jj.flatten()
        (factor_ids, v1_ids, v2_ids, idx_1s, idx_2s) = jax.vmap(verticle, in_axes=(0, 0, None, None, None))(ii, jj, H, W, N)
        adj_var_id = np.stack([v1_ids, v2_ids], axis=1)
        adj_var_idx = np.stack([idx_1s, idx_2s], axis=1)
        zs = np.zeros((factor_ids.size, D))
        z_Lams = np.tile(z_Lam, (factor_ids.size, 1, 1))
        threshold = np.ones((factor_ids.size, 1))
        potential = None
        v_between_facs = Factor(factor_ids, zs, z_Lams, threshold, potential, adj_var_id, adj_var_idx)
        between_facs = Factor(np.concatenate([h_between_facs.factor_id, v_between_facs.factor_id]), np.concatenate([h_between_facs.z, v_between_facs.z]), np.concatenate([h_between_facs.z_Lam, v_between_facs.z_Lam]), np.concatenate([h_between_facs.threshold, v_between_facs.threshold]), None, np.concatenate([h_between_facs.adj_var_id, v_between_facs.adj_var_id]), np.concatenate([h_between_facs.adj_var_idx, v_between_facs.adj_var_idx]))
        vars.adj_factor_idx[between_facs.adj_var_id, between_facs.adj_var_idx] = between_facs.factor_id.reshape(-1, 1)
    vars = jax.device_put(vars, gpu_device)
    prior_facs = jax.device_put(prior_facs, gpu_device)
    between_facs = jax.device_put(between_facs, gpu_device)
    return (vars, prior_facs, between_facs)

# --- Reference-unit (scale-invariant) variants.  Used only when run_pixvod's PIXVOD_REF_GAUGE
# gate selects them; the legacy configuration never calls them.  u2 is the per-coordinate square of the unit map u from reference
# units to current units (x = u * x_ref), so a precision in current units is Lam_ref / (u u^T).  Each function below is
# the legacy function with its fixed-unit constants evaluated in reference units; with u2 = 1 it computes the same
# values as the legacy function.

def safe_mu_ref(eta, Lam, u2):
    """safe_mu with its threshold and ridge _EPS read in reference units."""
    diag = jnp.diag(Lam)
    off = Lam - jnp.diag(diag)
    su = jnp.sqrt(u2)
    is_zero = jnp.all(Lam == 0)
    is_diag = jnp.all(jnp.abs(off) * (su[:, None] * su[None, :]) <= _EPS)

    def zero_branch(_):
        return eta

    def nonzero_branch(_):

        def diag_branch(__):
            ok = jnp.abs(diag) * u2 > _EPS
            denom = jnp.where(ok, diag, 1.0)
            x = eta / denom
            x = jnp.where(ok, x, 0.0)
            return x

        def dense_branch(__):
            if _FP32_RIDGE:
                # PIXVOD_FP32_RIDGE (diagnostic, FP32 only): ridge at least 32 ulp of the largest diagonal entry, so the
                # regularised matrix stays solvable in float32 (condition <= ~2.6e5); float64 path unchanged
                ridge = jnp.maximum(_EPS / u2, 32.0 * jnp.finfo(Lam.dtype).eps * jnp.max(jnp.abs(diag)))
                return jnp.linalg.solve(Lam + jnp.diag(ridge), eta)
            return jnp.linalg.solve(Lam + jnp.diag(_EPS / u2), eta)
        return jax.lax.cond(is_diag, diag_branch, dense_branch, operand=None)
    return jax.lax.cond(is_zero, zero_branch, nonzero_branch, operand=None)

@jax.jit
def Retract_SE3_plus_depth_ref(linpoint7: jnp.ndarray, m: Gaussian, alpha: float, u2):
    tau7 = alpha * safe_mu_ref(m.eta, m.Lam, u2)
    (tau6, tau_d) = _split_se3_depth(tau7)
    (lin6, lin_d) = _split_se3_depth(linpoint7)
    mean6 = se3_geo_dist(-lin6, tau6)
    mean_d = lin_d + tau_d
    mean7 = _stack_se3_depth(mean6, mean_d)
    Jrinv6 = se3_Jr_inv(tau6)
    Jrinv7 = _blkdiag_se3_scalar(Jrinv6, 1.0)
    Lam_new = _mm(_mm(Jrinv7.T, m.Lam), Jrinv7)
    belief = Gaussian(_mv(Lam_new, mean7), Lam_new)
    return (mean7, belief)

@jax.jit
def Rebase_SE3_plus_depth_ref(new_linpoint7: jnp.ndarray, old_linpoint7: jnp.ndarray, old_m: Gaussian, u2):
    tau7 = safe_mu_ref(old_m.eta, old_m.Lam, u2)
    (tau6, tau_d) = _split_se3_depth(tau7)
    (old_lin6, old_lin_d) = _split_se3_depth(old_linpoint7)
    (new_lin6, new_lin_d) = _split_se3_depth(new_linpoint7)
    global_se3 = se3_geo_dist(-old_lin6, tau6)
    global_d = old_lin_d + tau_d
    Jretract6 = se3_Jr_inv(tau6)
    Jretract7 = _blkdiag_se3_scalar(Jretract6, 1.0)
    delta6 = se3_geo_dist(new_lin6, global_se3)
    delta_d = global_d - new_lin_d
    delta7 = _stack_se3_depth(delta6, delta_d)
    Jdelta6 = se3_Jr(delta6)
    Jdelta7 = _blkdiag_se3_scalar(Jdelta6, 1.0)
    J = _mm(Jretract7, Jdelta7)
    Lam_new = _mm(_mm(J.T, old_m.Lam), J)
    return Gaussian(_mv(Lam_new, delta7), Lam_new)

def _safe_mu_ref_nodense(eta, Lam, u2):
    """PIXVOD_COND_SPLIT: safe_mu_ref for a Lam that is zero or diagonal in reference units -- the same select of its zero
    and diagonal branches that vmap makes of safe_mu_ref, without evaluating its dense branch."""
    diag = jnp.diag(Lam)
    is_zero = jnp.all(Lam == 0)
    ok = jnp.abs(diag) * u2 > _EPS
    denom = jnp.where(ok, diag, 1.0)
    x = eta / denom
    x = jnp.where(ok, x, 0.0)
    return jnp.where(is_zero, eta, x)


def _needs_dense_ref(Lam, u2):
    """PIXVOD_COND_SPLIT: True where safe_mu_ref takes its dense branch (Lam neither zero nor diagonal in reference units)."""
    diag = jnp.diag(Lam)
    off = Lam - jnp.diag(diag)
    su = jnp.sqrt(u2)
    is_zero = jnp.all(Lam == 0)
    is_diag = jnp.all(jnp.abs(off) * (su[:, None] * su[None, :]) <= _EPS)
    return jnp.logical_not(is_zero) & jnp.logical_not(is_diag)


def _rebase_ref_tail(new_linpoint7, old_linpoint7, old_m, tau7):
    """Rebase_SE3_plus_depth_ref after its safe_mu_ref (the same operations, with tau7 given)."""
    (tau6, tau_d) = _split_se3_depth(tau7)
    (old_lin6, old_lin_d) = _split_se3_depth(old_linpoint7)
    (new_lin6, new_lin_d) = _split_se3_depth(new_linpoint7)
    global_se3 = se3_geo_dist(-old_lin6, tau6)
    global_d = old_lin_d + tau_d
    Jretract6 = se3_Jr_inv(tau6)
    Jretract7 = _blkdiag_se3_scalar(Jretract6, 1.0)
    delta6 = se3_geo_dist(new_lin6, global_se3)
    delta_d = global_d - new_lin_d
    delta7 = _stack_se3_depth(delta6, delta_d)
    Jdelta6 = se3_Jr(delta6)
    Jdelta7 = _blkdiag_se3_scalar(Jdelta6, 1.0)
    J = _mm(Jretract7, Jdelta7)
    Lam_new = _mm(_mm(J.T, old_m.Lam), J)
    return Gaussian(_mv(Lam_new, delta7), Lam_new)


def _rebase_msgs_split(msgs, new_linpoints, old_linpoints, msg_slots, u2):
    """PIXVOD_COND_SPLIT: the occupied-slot rebase of update_variable_SE3D_ref with the slots split by factor class.
    vmap turns safe_mu_ref's lax.cond into a select, so its dense 7x7 solve runs for every slot.  The messages of the prior,
    normal and border-normal factors are zero or diagonal by construction (depth-only or identity-Jacobian potentials), so
    that class is rebased with the zero/diagonal branches only -- unless some slot of it needs the dense branch, which a
    scalar lax.cond checks (then the class runs the full rebase, as before).  The empty slots hold exact zero messages,
    for which safe_mu_ref returns eta itself.  Every slot gets the value of the select it had before."""
    (_, _, dg_idx, dg_var, dn_idx, dn_var) = msg_slots
    (n_var, n_port, d_op) = msgs.eta.shape
    n_slot = n_var * n_port
    eta_f = msgs.eta.reshape(n_slot, d_op)
    Lam_f = msgs.Lam.reshape(n_slot, d_op, d_op)
    full = jax.vmap(Rebase_SE3_plus_depth_ref, in_axes=(0, 0, 0, None))
    dn = Gaussian(eta_f[dn_idx], Lam_f[dn_idx])
    rebased_dn = full(new_linpoints[dn_var], old_linpoints[dn_var], dn, u2)
    dg = Gaussian(eta_f[dg_idx], Lam_f[dg_idx])
    need = jnp.any(jax.vmap(_needs_dense_ref, in_axes=(0, None))(dg.Lam, u2))
    rebased_dg = jax.lax.cond(need, lambda ops: full(ops[0], ops[1], ops[2], u2),
                              lambda ops: jax.vmap(lambda nl, ol, m: _rebase_ref_tail(nl, ol, m, _safe_mu_ref_nodense(m.eta, m.Lam, u2)))(*ops),
                              (new_linpoints[dg_var], old_linpoints[dg_var], dg))
    empty = Gaussian(jnp.zeros((n_var, d_op), dtype=msgs.eta.dtype), jnp.zeros((n_var, d_op, d_op), dtype=msgs.Lam.dtype))
    filler = jax.vmap(lambda nl, ol, m: _rebase_ref_tail(nl, ol, m, m.eta))(new_linpoints, old_linpoints, empty)
    new_eta = jnp.broadcast_to(filler.eta[:, None, :], (n_var, n_port, d_op)).reshape(n_slot, d_op).at[dn_idx].set(rebased_dn.eta).at[dg_idx].set(rebased_dg.eta).reshape(n_var, n_port, d_op)
    new_Lam = jnp.broadcast_to(filler.Lam[:, None, :, :], (n_var, n_port, d_op, d_op)).reshape(n_slot, d_op, d_op).at[dn_idx].set(rebased_dn.Lam).at[dg_idx].set(rebased_dg.Lam).reshape(n_var, n_port, d_op, d_op)
    return Gaussian(new_eta, new_Lam)


@partial(jax.jit, static_argnames=['alpha'], donate_argnames=_DON_V)
def update_variable_SE3D_ref(vars, old_linpoints, alpha: float=1.0, msg_slots=None, u2=None):
    """update_variable_SE3D with safe_mu in reference units (retract and rebase)."""
    old_posterior = jax.vmap(update_belief)(vars, vars.msgs)
    (new_linpoints, new_belief) = jax.vmap(Retract_SE3_plus_depth_ref, in_axes=(0, 0, None, None))(old_linpoints, old_posterior, alpha, u2)
    if msg_slots is None:

        def rebase_msgs_for_var(new_lin, old_lin, msgs_g):
            return jax.vmap(Rebase_SE3_plus_depth_ref, in_axes=(None, None, 0, None))(new_lin, old_lin, msgs_g, u2)
        new_msgs = jax.vmap(rebase_msgs_for_var, in_axes=(0, 0, 0))(new_linpoints, old_linpoints, vars.msgs)
    elif len(msg_slots) == 6:  # PIXVOD_COND_SPLIT slot layout (run_pixvod._build_msg_slots)
        new_msgs = _rebase_msgs_split(vars.msgs, new_linpoints, old_linpoints, msg_slots, u2)
    else:
        (slot_idx, slot_var) = msg_slots
        (n_var, n_port, d_op) = vars.msgs.eta.shape
        n_slot = n_var * n_port
        occupied = Gaussian(vars.msgs.eta.reshape(n_slot, d_op)[slot_idx], vars.msgs.Lam.reshape(n_slot, d_op, d_op)[slot_idx])
        rebased = jax.vmap(Rebase_SE3_plus_depth_ref, in_axes=(0, 0, 0, None))(new_linpoints[slot_var], old_linpoints[slot_var], occupied, u2)
        empty = Gaussian(jnp.zeros((n_var, d_op), dtype=vars.msgs.eta.dtype), jnp.zeros((n_var, d_op, d_op), dtype=vars.msgs.Lam.dtype))
        filler = jax.vmap(Rebase_SE3_plus_depth_ref, in_axes=(0, 0, 0, None))(new_linpoints, old_linpoints, empty, u2)
        new_eta = jnp.broadcast_to(filler.eta[:, None, :], (n_var, n_port, d_op)).reshape(n_slot, d_op).at[slot_idx].set(rebased.eta).reshape(n_var, n_port, d_op)
        new_Lam = jnp.broadcast_to(filler.Lam[:, None, :, :], (n_var, n_port, d_op, d_op)).reshape(n_slot, d_op, d_op).at[slot_idx].set(rebased.Lam).reshape(n_var, n_port, d_op, d_op)
        new_msgs = Gaussian(new_eta, new_Lam)
    vars.belief = jax.vmap(update_belief)(vars, new_msgs)
    vtof_msgs = jax.vmap(compute_vtof_msgs, in_axes=(0, 0))(vars, new_msgs)
    vars.belief = new_belief
    return (vars, vtof_msgs, new_linpoints)

def marginalize_ref(gaussians: Gaussian, i, j, u2_full):
    """marginalize with its shared loading computed in reference units (u2_full: one entry per factor dimension)."""
    eta = gaussians.eta
    Lam = gaussians.Lam
    k = eta.size
    idx = jnp.arange(0, k)
    aa = idx[i:j]
    bb = jnp.concatenate([idx[:i], idx[j:]])
    aa_eta = eta[aa]
    bb_eta = eta[bb]
    aa_Lam = Lam[aa[:, None], aa]
    ab_Lam = Lam[aa[:, None], bb]
    bb_Lam = Lam[bb][:, bb]
    if bb_Lam.size == 0:
        return Gaussian(aa_eta, aa_Lam)
    u2_bb = jnp.concatenate([u2_full[:i], u2_full[j:]])
    diag = jnp.diag(bb_Lam)
    scale = jnp.maximum(jnp.mean(jnp.abs(diag) * u2_bb), 1e-12)
    lam_damp = jnp.maximum(1e-08, 0.001 * scale)
    bb_reg = bb_Lam + jnp.diag(lam_damp / u2_bb)
    X = jnp.linalg.solve(bb_reg, ab_Lam.T)
    y = jnp.linalg.solve(bb_reg, bb_eta)
    eta = aa_eta - ab_Lam @ y
    Lam = aa_Lam - ab_Lam @ X
    return Gaussian(eta, Lam)

def compute_ftov_msg_ref(factor, vtof_msgs, u2_full):
    (N_adj, dim) = vtof_msgs.eta.shape
    pot = factor.potential.copy()
    i = 0
    for n in range(N_adj):
        j = i + dim
        pot.eta = pot.eta.at[i:j].add(vtof_msgs.eta[n])
        pot.Lam = pot.Lam.at[i:j, i:j].add(vtof_msgs.Lam[n])
        i = j
    ftov_msgs = []
    i = 0
    for n in range(N_adj):
        j = i + dim
        pot_m_1 = pot.copy()
        pot_m_1.eta = pot_m_1.eta.at[i:j].add(-vtof_msgs.eta[n])
        pot_m_1.Lam = pot_m_1.Lam.at[i:j, i:j].add(-vtof_msgs.Lam[n])
        msg = marginalize_ref(pot_m_1, i, j, u2_full)
        ftov_msgs.append(msg)
        i = j
    return tree_stack(ftov_msgs, use_np=False)

@partial(jax.jit, static_argnames=['f', 'w'], donate_argnames=_DON)
def update_factor_manifold_hJ_ref(facs, vars, vtof_msgs, linpoints, f, w, u2, mask_ftov=None):
    """update_factor_manifold_hJ with marginalisation loading in reference units (u2: per-variable 7-vector)."""
    vtof_msgs_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs)
    linpoints_reordered = jax.tree_util.tree_map(lambda x: x[facs.adj_var_id], linpoints)
    msg_dim = vtof_msgs_reordered.eta.shape[-1]
    factor_dim = linpoints_reordered.shape[-1]
    vtof_msgs_for_compute = vtof_msgs_reordered
    if factor_dim != msg_dim:
        vtof_msgs_for_compute = Gaussian(eta=vtof_msgs_reordered.eta[..., :factor_dim], Lam=vtof_msgs_reordered.Lam[..., :factor_dim, :factor_dim])
    facs.potential = jax.vmap(factor_update_manifold_hJ, in_axes=(0, 0, None, None))(facs, linpoints_reordered, f, w)
    u2_full = jnp.tile(u2[:factor_dim], vtof_msgs_for_compute.eta.shape[-2])
    ftov_msgs = jax.vmap(compute_ftov_msg_ref, in_axes=(0, 0, None))(facs, vtof_msgs_for_compute, u2_full)
    if factor_dim != msg_dim:
        target_dim = msg_dim
        padded_eta = jnp.zeros(ftov_msgs.eta.shape[:-1] + (target_dim,))
        padded_eta = padded_eta.at[..., :factor_dim].set(ftov_msgs.eta)
        padded_Lam = jnp.zeros(ftov_msgs.Lam.shape[:-2] + (target_dim, target_dim))
        padded_Lam = padded_Lam.at[..., :factor_dim, :factor_dim].set(ftov_msgs.Lam)
        ftov_msgs = Gaussian(padded_eta, padded_Lam)
    if mask_ftov is not None:
        target_eta = vars.msgs.eta[facs.adj_var_id, facs.adj_var_idx]
        target_Lam = vars.msgs.Lam[facs.adj_var_id, facs.adj_var_idx]
        mask_eta = mask_ftov[..., None]
        mask_Lam = mask_ftov[..., None, None]
        blended_eta = jnp.where(mask_eta, ftov_msgs.eta, target_eta)
        blended_Lam = jnp.where(mask_Lam, ftov_msgs.Lam, target_Lam)
        vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(blended_eta)
        vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(blended_Lam)
    else:
        vars.msgs.eta = vars.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.eta)
        vars.msgs.Lam = vars.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(ftov_msgs.Lam)
    vars.msgs.eta = vars.msgs.eta.at[:, -1].set(0)
    vars.msgs.Lam = vars.msgs.Lam.at[:, -1].set(0)
    return (facs, vars)
