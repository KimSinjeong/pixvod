# /// script
# requires-python = "==3.10.*"
# dependencies = ["torch==2.5.1", "torchvision==0.20.1", "geffnet==1.0.2", "numpy==2.0.1", "pillow==11.1.0"]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
# ///
"""Predict surface normals with DSINE for frames of a rendered Replica sequence (dsine_normal{idx:06d}.png).

    uv run replica/dsine_normals.py --images-dir data/replica_office2_128 --frames 12000

Runs DSINE's own projects/dsine/test_minimal.py from the submodule, unmodified, in a temporary working folder: the
frames are copied to samples/img/ with one intrinsics file each (fx,fy,cx,cy), and samples/output/frame{idx}.png is
saved as dsine_normal{idx}.png next to the frames.  The only change to DSINE's behaviour is where get_args() creates its
experiment folders (the upstream path is in the authors' home directory).  Checkpoint: the link in
third_party/DSINE/README.md, saved as third_party/DSINE/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt.
The inline metadata above gives the script its own environment, because torch pins CUDA libraries that differ from the
pipeline's JAX pins."""
import argparse
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSINE_DIR = ROOT / 'third_party' / 'DSINE'
TEST_MINIMAL = DSINE_DIR / 'projects' / 'dsine' / 'test_minimal.py'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--images-dir', required=True, type=Path, help='folder with frame{idx:06d}.jpg; the PNGs are written here')
    ap.add_argument('--frames', required=True, type=int, nargs='+', help='frame indices, e.g. the start frames of the runs')
    ap.add_argument('--intrinsics', type=float, nargs=4, default=[109.985, 109.985, 63.5, 63.5], metavar=('FX', 'FY', 'CX', 'CY'))
    ap.add_argument('--ckpt', type=Path, default=DSINE_DIR / 'projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt')
    a = ap.parse_args()
    images_dir = a.images_dir.resolve()
    ckpt = a.ckpt.resolve()
    if not ckpt.exists():
        sys.exit(f'DSINE checkpoint not found: {ckpt} (see third_party/DSINE/README.md, "Step 1")')

    work = Path(tempfile.mkdtemp(prefix='dsine_'))
    (work / 'samples' / 'img').mkdir(parents=True)
    for idx in a.frames:
        shutil.copyfile(images_dir / f'frame{idx:06d}.jpg', work / 'samples' / 'img' / f'frame{idx:06d}.jpg')
        (work / 'samples' / 'img' / f'frame{idx:06d}.txt').write_text(','.join(f'{v:g}' for v in a.intrinsics) + '\n')
    (work / 'experiments').symlink_to(TEST_MINIMAL.parent / 'experiments')

    sys.dont_write_bytecode = True                   # keep the submodule free of __pycache__
    sys.path.insert(0, str(DSINE_DIR))
    os.chdir(work)                                   # test_minimal.py reads ./samples/img and ./experiments
    import projects.dsine.config as config
    config.EXPERIMENT_DIR = str(work / 'exp')        # where get_args() creates its experiment folders
    sys.argv = [str(TEST_MINIMAL), './experiments/exp001_cvpr2024/dsine.txt', '--ckpt_path', str(ckpt)]
    runpy.run_path(str(TEST_MINIMAL), run_name='__main__')

    for idx in a.frames:
        out = images_dir / f'dsine_normal{idx:06d}.png'
        shutil.copyfile(work / 'samples' / 'output' / f'frame{idx:06d}.png', out)
        print(f'frame{idx:06d}.jpg -> {out}', flush=True)
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
