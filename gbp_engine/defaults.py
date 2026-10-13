"""Default configuration of the pipeline (champion): the champion gates in float32 with TF32 matrix products, the
half-pixel correction and the accelerations.

Applied at import, before any gate is read (gbp_engine/__init__.py and run_pixvod.py import this module first), by
writing every gate variable that is not already in the environment into os.environ.  The values printed as
'[DEFAULTS] ...' are the effective ones.

  PIXVOD_PRECISION = tf32 (default)  float32 + safe_mu_ref ridge + NaN-path fix, default matmul precision (TF32 allowed)
                   | highest         the same with JAX_DEFAULT_MATMUL_PRECISION=highest (full float32 products)
                   | float64         float64; ridge and NaN-path fix off (champion gates, the half-pixel correction and the
                                     accelerations in float64; float64 with the accelerations is untested)
  PIXVOD_BASELINE  = champion (default) | legacy: set nothing -- the previous default configuration (float64, every
                     gate unset); gates set explicitly still apply.

Champion gates = PIXVOD_EXACT_READOUT=1 PIXVOD_REF_GAUGE=prior_rho,prior_t,between_t,marg,safe PIXVOD_NORMAL_BORDER=1
PIXVOD_BINI=1 PIXVOD_NORMAL_FLOOR=invalid PIXVOD_OOB_MSG=zero.  Float32 = PIXVOD_FP32=1 PIXVOD_FP32_RIDGE=1, NaN-path fix =
PIXVOD_UNARY_EXACT=1 PIXVOD_READOUT_GUARD=1.
Half-pixel correction (PIXEL) = PIXVOD_PIXEL_CENTER=1 (run_pixvo.py photometric_array_xy; 0 = the sampler before it).

Overrides and their limits:
- A gate set explicitly wins (e.g. PIXVOD_BINI=0, PIXVOD_NORMAL_FLOOR=anchor, PIXVOD_OOB_MSG= for the legacy handling).
  PIXVOD_REF_GAUGE has no 'off' value: switching it off needs PIXVOD_BASELINE=legacy and the other gates set by hand.
- PIXVOD_FP32 and JAX_DEFAULT_MATMUL_PRECISION define the precision and must agree with PIXVOD_PRECISION (else exit).
- The float32 ridge acts only inside safe_mu_ref, i.e. only with the token 'safe' in PIXVOD_REF_GAUGE; float32 without it
  is the configuration that failed every run (exit).
- Every process that imports gbp_engine gets these defaults, offline analysis included (float32 unless
  PIXVOD_PRECISION=float64 or PIXVOD_BASELINE=legacy), and child processes inherit them through os.environ; the
  legacy banner therefore lists every gate it finds already set.

Accelerations.  Each gate in ACCEL is 0 or 1 and is read at import by the module that uses it.  Under the
champion default every gate not set explicitly takes PIXVOD_ACCEL's value (default 1), so PIXVOD_ACCEL=0 turns all of them
off and gives the code path before the accelerations (with PIXVOD_PIXEL_CENTER=0 as well: the default before
both); a gate set explicitly wins (e.g. PIXVOD_ACCEL=0 PIXVOD_DONATE=1 is donation
alone).  PIXVOD_BASELINE=legacy sets none of them (legacy unchanged); a gate set
explicitly still applies there.  Printed as '[ACCEL] ...'."""
import os
import sys

# gate -> (changes numbers?, meaning)
ACCEL = {
    'PIXVOD_ASYNC_PLOT': (False, 'trend figures drawn by a separate worker process from a snapshot of the plotted data'),
    'PIXVOD_DONATE': (False, 'variable store and factor sets donated to the jitted updates (no device copies)'),
    'PIXVOD_DIAG_FETCH': (False, 'output-only work: dead device-to-host copies skipped, diagnostic fetches batched'),
    'PIXVOD_STAGE_CONST': (False, 'stage constants built once and kept on the device'),
    'PIXVOD_COND_SPLIT': (False, 'rebase: the dense solve is skipped for message slots whose class is zero/diagonal'),
    'PIXVOD_UNCOMMITTED': (False, 'the arrays placed explicitly on GPU 0 are put on the default device uncommitted, so the '
                                 'jitted updates are not compiled again when committed arguments reach them'),
    'PIXVOD_DEV_REFRESH': (True, 'Student-t weight refresh on the device (working precision, Cholesky Schur complement)'),
    'PIXVOD_SMALLSOLVE': (True, 'normal-factor messages (interior, nb3, nb1) by the closed-form Schur complement onto the depth '
                                'entries (unrolled Cholesky) instead of two batched 14x14 LU solves per target'),
}
ACCEL_NUMBER_CHANGING = sorted(k for k, (c, _) in ACCEL.items() if c)


def _accel_check():
    bad = {k: os.environ[k] for k in ACCEL if k in os.environ and os.environ[k] not in ('0', '1')}
    if bad:
        raise SystemExit(f'acceleration gates must be 0 or 1; got {bad}')

CHAMPION = {'PIXVOD_EXACT_READOUT': '1', 'PIXVOD_REF_GAUGE': 'prior_rho,prior_t,between_t,marg,safe', 'PIXVOD_NORMAL_BORDER': '1',
         'PIXVOD_BINI': '1', 'PIXVOD_NORMAL_FLOOR': 'invalid', 'PIXVOD_OOB_MSG': 'zero'}
FP32 = {'PIXVOD_FP32': '1', 'PIXVOD_FP32_RIDGE': '1', 'PIXVOD_UNARY_EXACT': '1', 'PIXVOD_READOUT_GUARD': '1'}
PIXEL = {'PIXVOD_PIXEL_CENTER': '1'}
PRECISION = {'tf32': FP32, 'highest': dict(FP32, JAX_DEFAULT_MATMUL_PRECISION='highest'),
             'float64': {'PIXVOD_FP32': '0', 'PIXVOD_FP32_RIDGE': '0', 'PIXVOD_UNARY_EXACT': '0', 'PIXVOD_READOUT_GUARD': '0'}}

BASELINE = os.environ.get('PIXVOD_BASELINE', 'champion')
if BASELINE not in ('champion', 'legacy'):
    raise SystemExit(f'PIXVOD_BASELINE must be champion or legacy; got {BASELINE!r}')
if BASELINE == 'legacy':
    if 'PIXVOD_PRECISION' in os.environ:
        raise SystemExit('PIXVOD_PRECISION applies to the champion default only, not to PIXVOD_BASELINE=legacy')
    if os.environ.get('PIXVOD_ACCEL', '0') != '0':
        raise SystemExit('PIXVOD_ACCEL applies to the champion default only, not to PIXVOD_BASELINE=legacy (set gates explicitly)')
    _accel_check()
    present = sorted(k for k in list(CHAMPION) + list(PIXEL) + list(PRECISION['highest']) if k in os.environ)
    print('[DEFAULTS] baseline=legacy: no defaults set' + (' (already set: ' + ' '.join(f'{k}={os.environ[k]}' for k in present) + ')'
                                                              if present else ' (previous default)'), flush=True)
    _acc_set = [k for k in ACCEL if k in os.environ]
    if _acc_set:
        print('[ACCEL] baseline=legacy, set explicitly: ' + ' '.join(f'{k}={os.environ[k]}' for k in _acc_set), flush=True)
else:
    PREC = os.environ.get('PIXVOD_PRECISION', 'tf32')
    if PREC not in PRECISION:
        raise SystemExit(f'PIXVOD_PRECISION must be tf32, highest or float64; got {PREC!r}')
    explicit = sorted(k for k in list(CHAMPION) + list(PIXEL) + list(PRECISION[PREC]) if k in os.environ)
    for k, v in {**CHAMPION, **PIXEL, **PRECISION[PREC]}.items():
        os.environ.setdefault(k, v)
    # the two variables that define the precision must agree with PIXVOD_PRECISION
    mm = os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')
    if os.environ['PIXVOD_FP32'] != ('0' if PREC == 'float64' else '1') or mm != ('highest' if PREC == 'highest' else None):
        raise SystemExit(f'PIXVOD_PRECISION={PREC} contradicts PIXVOD_FP32={os.environ["PIXVOD_FP32"]!r} / '
                         f'JAX_DEFAULT_MATMUL_PRECISION={mm!r}; set only PIXVOD_PRECISION')
    if os.environ['PIXVOD_FP32'] == '1' and os.environ.get('PIXVOD_FP32_RIDGE') == '1' and 'safe' not in os.environ['PIXVOD_REF_GAUGE'].split(','):
        raise SystemExit('the float32 ridge acts only through the token safe of PIXVOD_REF_GAUGE; keep it in float32')
    if PREC == 'highest' and 'jax' in sys.modules:  # jax read its environment at import already
        sys.modules['jax'].config.update('jax_default_matmul_precision', 'highest')
    print(f'[DEFAULTS] baseline=champion precision={PREC}: ' + ' '.join(f'{k}={os.environ[k]}' for k in {**CHAMPION, **PIXEL, **PRECISION[PREC]})
          + (f' (set explicitly: {",".join(explicit)})' if explicit else ''), flush=True)
    ACC = os.environ.get('PIXVOD_ACCEL', '1')
    if ACC not in ('0', '1'):
        raise SystemExit(f'PIXVOD_ACCEL must be 0 or 1; got {ACC!r}')
    _accel_check()
    _acc_explicit = [k for k in ACCEL if k in os.environ]
    for k in ACCEL:
        os.environ.setdefault(k, ACC)
    print(f'[ACCEL] PIXVOD_ACCEL={ACC}: ' + ' '.join(f'{k}={os.environ[k]}' for k in ACCEL)
          + (f' (set explicitly: {",".join(_acc_explicit)})' if _acc_explicit else ''), flush=True)


def gate(name):
    """Effective value of an acceleration gate (validated above; unset = off, e.g. under PIXVOD_BASELINE=legacy)."""
    assert name in ACCEL, name
    return os.environ.get(name, '0') == '1'
