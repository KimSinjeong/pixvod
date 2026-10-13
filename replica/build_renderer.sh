#!/usr/bin/env bash
# Build ReplicaRenderer from the Replica-Dataset submodule with replica/replica_renderer.patch applied:
# trajectory input (one T_wc per line), intrinsics file (W H / horizontal|vertical FoV, or a 3x3 K), and a
# normal{idx}.png output next to frame{idx}.jpg and depth{idx}.png.
# Needs: cmake (<= 3.x; Pangolin of this vintage does not configure with CMake 4), a C++ compiler, OpenGL/EGL/GLEW,
# libjpeg and libpng development packages.  Run with no conda env active so that only system libraries are found.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="$ROOT/third_party/Replica-Dataset"
PATCH="$ROOT/replica/replica_renderer.patch"
cd "$SDK"
git submodule update --init --recursive
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
else
  echo "patch neither applies nor is applied; check $SDK" >&2; exit 1
fi
(cd 3rdparty/Pangolin && mkdir -p build && cd build \
  && cmake .. -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF && make -j"$(nproc)")
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
echo "built: $SDK/build/ReplicaSDK/ReplicaRenderer"
