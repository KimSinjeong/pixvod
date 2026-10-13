import numpy as np
import numexpr as ne

try:
    import jax.numpy as jnp
except Exception:
    jnp = None

# EXR-related Utils

def encode_to_srgb(x):
    a = 0.055
    return ne.evaluate("""where(
                            x <= 0.0031308,
                            x * 12.92,
                            (1 + a) * (x ** (1 / 2.4)) - a
                          )""")

# Factor Graph Building Utils

def build_quadtree_edges(H: int, assign_range: int = 12, backend: str = "numpy"):
    """
    Returns:
      edge_ids : (M,) int64
      edge_vars: (M,2) int64  # (child_idx, parent_idx)
      edge_pos : (M,2) int32  # (16, 12/13/14/15)
    """
    _TL = assign_range
    _BL = assign_range + 1
    _TR = assign_range + 2
    _BR = assign_range + 3
    _CHILD = assign_range + 4
    # --- Input check: must be a positive power of two (so we can recurse down to the 1x1 root) ---
    if not (isinstance(H, int) and H > 0 and (H & (H - 1)) == 0):
        raise ValueError("H must be a positive power of two (e.g. 2, 4, 8, ..., 256).")

    xp = np if backend == "numpy" else jnp
    if backend == "jax" and jnp is None:
        raise RuntimeError("JAX is not installed. Use backend='numpy'.")

    L = int(np.log2(H)) + 1  # levels 0..L-1 (the last, L-1, is the 1x1 root)
    sizes = xp.array([H // (2**l) for l in range(L)], dtype=xp.int64)  # side length of each level
    counts = sizes * sizes  # number of elements at each level
    offsets = xp.concatenate([xp.array([0], dtype=xp.int64), xp.cumsum(counts[:-1])])  # start offset per level

    edge_vars_list = []
    edge_pos_list  = []

    # 2x2 block at level l -> parent at level l+1 (l=0..L-2)
    for l in range(L - 1):
        n_child  = int(sizes[l])
        n_parent = int(sizes[l + 1])

        # parent coordinate (i,j) grid
        i = xp.arange(n_parent, dtype=xp.int64)
        j = xp.arange(n_parent, dtype=xp.int64)
        Ii, Jj = xp.meshgrid(i, j, indexing="ij")  # shape (n_parent, n_parent)
        parent_flat = offsets[l + 1] + (Ii * n_parent + Jj)  # (n_parent, n_parent)

        # child coordinates (TL, BL, TR, BR)
        # TL (top-left):     (2i,   2j  )
        TL_r, TL_c = 2 * Ii,     2 * Jj
        # BL (bottom-left):  (2i+1, 2j  )
        BL_r, BL_c = 2 * Ii + 1, 2 * Jj
        # TR (top-right):    (2i,   2j+1)
        TR_r, TR_c = 2 * Ii,     2 * Jj + 1
        # BR (bottom-right): (2i+1, 2j+1)
        BR_r, BR_c = 2 * Ii + 1, 2 * Jj + 1

        child_base = offsets[l]
        TL_flat = child_base + (TL_r * n_child + TL_c)
        BL_flat = child_base + (BL_r * n_child + BL_c)
        TR_flat = child_base + (TR_r * n_child + TR_c)
        BR_flat = child_base + (BR_r * n_child + BR_c)

        # Position codes: child fixed at 16, parent is TL=12, BL=13, TR=14, BR=15
        # stack order: TL, BL, TR, BR (order can be changed if desired)
        childs  = xp.stack([TL_flat, BL_flat, TR_flat, BR_flat], axis=0)           # (4, n_parent, n_parent)
        parents = xp.stack([parent_flat]*4, axis=0)                                 # (4, n_parent, n_parent)
        pair    = xp.stack([childs.reshape(-1), parents.reshape(-1)], axis=1)       # (4*n_parent^2, 2)

        pos_parent = xp.concatenate([
            xp.full((n_parent * n_parent, 1), _TL, dtype=xp.int32),  # TL
            xp.full((n_parent * n_parent, 1), _BL, dtype=xp.int32),  # BL
            xp.full((n_parent * n_parent, 1), _TR, dtype=xp.int32),  # TR
            xp.full((n_parent * n_parent, 1), _BR, dtype=xp.int32),  # BR
        ], axis=0)
        pos_child  = xp.full((pos_parent.shape[0], 1), _CHILD, dtype=xp.int32)
        pos_pair   = xp.concatenate([pos_child, pos_parent], axis=1)                # (4*n_parent^2, 2)

        edge_vars_list.append(pair)
        edge_pos_list.append(pos_pair)

    edge_vars = xp.concatenate(edge_vars_list, axis=0).astype(xp.int64)  # (M,2)
    edge_pos  = xp.concatenate(edge_pos_list, axis=0).astype(xp.int32)   # (M,2)
    M = int(edge_vars.shape[0])
    edge_ids = xp.arange(M, dtype=xp.int64)

    return M, edge_ids, edge_vars, edge_pos

# if __name__ == "__main__":
#     # Usecase:
#     H = 64
#     M, edge_ids, edge_vars, edge_pos = build_quadtree_edges(H, backend="numpy")
#     print(M, edge_ids.shape, edge_vars.shape, edge_pos.shape)
