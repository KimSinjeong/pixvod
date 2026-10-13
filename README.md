<div align="center">

# PixVOD: Pixel-Distributed Direct Visual Odometry and Depth Estimation

Visual odometry and depth estimation distributed across pixels, which agree on camera motion through Gaussian Belief Propagation.

[Paper](https://arxiv.org/abs/2606.03989) ·
[Project page](https://www.shinjeongkim.com/pixvod/)

</div>

PixVOD distributes direct visual odometry and depth estimation across the pixels of a focal-plane sensor-processor
array. Each pixel keeps its own state and exchanges messages with its neighbours through Gaussian Belief Propagation
(GBP), so the pixels reach consensus on the camera motion while inferring depth from per-pixel photometric observations
and a surface-normal prior. A keyframe-like anchoring mechanism regulates the effective baseline between frames.

This repository runs the method as a GPU simulation on Replica. It reproduces the upsampled camera trajectories, renders
RGB, depth and normal frames, and runs the estimator with an optional live [Rerun](https://rerun.io) view.

## Installation

Requirements: Linux, an NVIDIA GPU and driver, [uv](https://docs.astral.sh/uv/), and for the renderer CMake 3.x, a C++
compiler, and the OpenGL/EGL/GLEW, libjpeg and libpng development packages. Pangolin at the pinned commit does not
configure with CMake 4.

```bash
git clone --recursive https://github.com/KimSinjeong/pixvod.git
cd pixvod                  # existing clone: git submodule update --init --recursive

uv sync                    # Python 3.10 environment in .venv (about 5.5 GB including the CUDA libraries)
bash replica/build_renderer.sh
```

`uv sync` installs JAX 0.6.2 and the CUDA libraries at the versions that produced the reference results. Results depend
on the GPU model and the JAX version, so keep these pins when comparing runs byte for byte.

`build_renderer.sh` applies `replica/replica_renderer.patch` to the Replica-Dataset submodule and builds Pangolin and
`ReplicaRenderer`. Run it from a shell with no conda environment active, so that only system libraries are found. The
patch adds trajectory input, an intrinsics file, a normal-map output and a fixed exposure to the stock renderer.

## Data

```text
data/
├── replica_v1/office_2/                  # mesh.ply, textures/, glass.sur
├── Replica/office2/
│   ├── traj.txt                          # 2,000 poses (NICE-SLAM)
│   └── traj_sub.txt                      # (2,000 − 1)(<N> + 1) + 1 poses (upsampled)
└── replica_office2_128/                  # frame{idx}.jpg, depth{idx}.png, normal{idx}.png
                                          # (+ dsine_normal{idx}.png, optional)
```

- **Replica meshes and textures** (about 43 GB): `cd third_party/Replica-Dataset && ./download.sh /path/to/replica_v1`
  (needs `pigz`), then link or move the result to `data/replica_v1`. Keep the
  [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) terms and citation.
- **Camera trajectories**: `traj.txt` of each scene from the NICE-SLAM Replica archive
  (<https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip>), one pose per line as a row-major 4×4 camera-to-world
  matrix. Only `traj.txt` is used.
- Scene names: `office0`…`office4` and `room0`…`room2` on the trajectory side; `office_0`…`room_2` in `replica_v1`.
- `<N>`: the number of camera poses inserted between consecutive poses of `traj.txt` (the reference results use 9).

### Trajectory upsampling

Insert `<N>` poses between consecutive poses by screw linear interpolation (ScLERP), turning 2,000 poses into
(2,000 − 1)(`<N>` + 1) + 1:

```bash
uv run python replica/sclerp_upsample.py \
  --input data/Replica/office2/traj.txt \
  --output data/Replica/office2/traj_sub.txt \
  --n <N>
```

### Rendering

Render at 128×128 with a 60° horizontal field of view (fx = fy = 109.985, cx = cy = 63.5):

```bash
# whole scene: (2,000 − 1)(<N> + 1) + 1 frames, about 0.2 s each
bash replica/render_scene.sh office2 data/replica_v1 data/Replica/office2/traj_sub.txt data/replica_office2_128

# only the frames a run needs: start frame S with 150 target frames uses S..S+150 (151 frames, about 11 s)
bash replica/render_scene.sh office2 data/replica_v1 data/Replica/office2/traj_sub.txt data/replica_office2_128 12000 151
```

Each index `idx` is the line number in the full trajectory. The renderer writes three files per frame:

- `frame{idx:06d}.jpg`: RGB.
- `depth{idx:06d}.png`: uint16, metres × 6553.5.
- `normal{idx:06d}.png`: uint16 RGB.

Without a display the renderer uses EGL.

## Running

Run from a shell with no `PIXVOD_*`, `JAX_*` or `XLA_*` variables set. The defaults come from `gbp_engine/defaults.py`,
and any variable you set overrides them. The default run has 1,000 initial depth-preparation iterations, then 150
target frames with 100 GBP iterations each.

```bash
uv run python run_pixvod.py \
  --set images_dir=data/replica_office2_128 \
  --set trajectory_path=data/Replica/office2/traj_sub.txt \
  --set start_frame=12000 \
  --set output_root=outputs \
  --set experiment_name=office2_12000
```

Results go to `outputs/office2_12000/logs/iter_trend_combined.csv`. The evaluation value is `norm_geo_err_current` in
the last row (iteration 15000). A run takes about 100 s on an RTX 4090.

### Live visualization with Rerun

Prefix the same command with `PIXVOD_RERUN=1`:

```bash
PIXVOD_RERUN=1 uv run python run_pixvod.py \
  --set images_dir=data/replica_office2_128 \
  --set trajectory_path=data/Replica/office2/traj_sub.txt \
  --set start_frame=12000 \
  --set output_root=outputs \
  --set experiment_name=office2_12000
```

If no viewer is running, the run starts one: `.venv/bin/rerun --bind 127.0.0.1 --port 9876`, reachable only from this
machine. If a viewer is already running, the run streams into it as a new recording (the record of one run). This needs
a display.

The view updates every 100 depth-preparation iterations and at the end of every target stage. It shows:

- camera trajectories (GT red, estimate green) and the reference and current cameras;
- GT and estimated depth and point clouds of the reference frame;
- the warped current frame and the photometric residual;
- error plots.

The numbers are byte-identical to a run without the viewer. On an RTX 4090 the run takes about 5% longer than when without the viewer.

### Predicted surface normals (DSINE)

By default the surface-normal prior of the reference (start) frame is the rendered normal map. To use a normal map
predicted from the RGB frame by [DSINE](https://github.com/baegwangbin/DSINE) instead:

1. Download the DSINE weights (link in `third_party/DSINE/README.md`, "Step 1") and save `exp001_cvpr2024/dsine.pt` as
   `third_party/DSINE/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt`.
2. Predict the normal map of each start frame you will run. The script declares its own dependencies (PyTorch 2.5.1,
   geffnet), and `uv run` builds a separate environment for it on first use, because PyTorch pins CUDA libraries
   that differ from the JAX ones:

   ```bash
   uv run replica/dsine_normals.py --images-dir data/replica_office2_128 --frames 12000
   ```

   It runs DSINE's own `projects/dsine/test_minimal.py` from the submodule, unmodified, on copies of the frames with
   the intrinsics fx = fy = 109.985, cx = cy = 63.5. It saves each output as `dsine_normal012000.png` (uint8 RGB)
   next to the frames.
3. Add `--set use_dsine_normal=true` to the run command. Only the reference frame's normal is used. If its
   `dsine_normal{idx}.png` is missing, the run silently falls back to the rendered normal.

Normal convention: DSINE uses (x, y, z) = (right, down, front) and orients a surface facing the camera as (0, 0, +1),
the opposite sign of the usual outward normal in OpenCV camera coordinates. The rendered `normal{idx}.png` maps use the
same convention, so both files are read as they are, with no axis swap or sign flip. Do not negate DSINE's output.

## Repository layout

```text
run_pixvod.py                 entrypoint: dataset runner (stages, logging, figures)
run_pixvo.py                  factor graph construction, measurement models, base runner
gbp_engine/                   GBP primitives, Lie group utilities, default settings
configs/replica.yaml          parameter configuration (Replica)
trend_plots.py                error and pose trend figures
rerun_live.py                 live Rerun stream (PIXVOD_RERUN=1)
replica/                      trajectory upsampling, renderer patch, build and render scripts, DSINE normals
third_party/Replica-Dataset/  renderer source (git submodule)
third_party/DSINE/            surface-normal predictor (git submodule)
```

## License and citation

Original code is available under the [MIT License](LICENSE). Third-party components (Replica-Dataset, DSINE) retain
their own licenses, and the Replica dataset keeps its own terms. `replica/replica_renderer.patch` modifies the
Replica-Dataset SDK and is distributed under the
[Replica Dataset Research Terms](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE), not the MIT
License.

```bibtex
@InProceedings{kim2026pixvod,
  title     = {{PixVOD}: Pixel-Distributed Direct Visual Odometry and Depth Estimation},
  author    = {Kim, Shinjeong and Alzugaray, Ignacio and Rhodes, Callum and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

We thank [Riku Murai](https://rmurai.co.uk/). Much of this code builds on the early implementation of applying GBP to [bilateral normal integration](https://github.com/xucao-42/bilateral_normal_integration) by Callum Rhodes and Riku Murai.
