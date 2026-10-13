"""gbp_engine — GBP core modules used by the PixVOD trajectory runner.

Author: Shinjeong Kim <shinjeong.kim.99@gmail.com>

This package ships only the pure-Python pieces required by ``run_pixvod.py`` /
``run_pixvo.py``:

- ``gbp``   — Gaussian Belief Propagation core (factors, Gaussian, message passing)
- ``lie``   — SO(3) / SE(3) Lie-group operations
- ``utils`` — helpers (e.g. quadtree edge construction)
- ``defaults`` — default gate settings (champion: champion gates, float32, TF32, half-pixel correction, accelerations), applied on import

The original Blender add-on (operators / overlays / panels / properties) is
intentionally not part of this release.
"""
from . import defaults  # before gbp / lie read their gates
