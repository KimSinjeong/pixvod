#!/usr/bin/env bash
# Render one Replica scene along an upsampled trajectory at 128 x 128 (horizontal FoV 60 deg: fx = fy = 109.985,
# cx = cy = 63.5), writing frame{idx}.jpg, depth{idx}.png (uint16, metres x 6553.5) and normal{idx}.png (uint16 RGB).
#   usage: render_scene.sh <scene, e.g. office2> <replica_v1 dir> <trajectory> <out dir> [<first index> <count>]
# The renderer names files 0..K-1 by trajectory line; with <first index> <count> only that range of the trajectory is
# rendered and the files are renamed to their index in the full trajectory (the name the pipeline reads).
# A full scene (19,991 poses for N = 9) takes about 1.25 h.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RENDERER="$ROOT/third_party/Replica-Dataset/build/ReplicaSDK/ReplicaRenderer"
SCENE=$1; REPLICA=$2; TRAJ=$(realpath "$3"); OUT=$4; FIRST=${5:-}; COUNT=${6:-}
NAME="${SCENE//[0-9]/}_${SCENE//[^0-9]/}"   # office2 -> office_2, the replica_v1 folder name
SRC="$(realpath "$REPLICA")/$NAME"
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
if [ -n "$FIRST" ]; then
  TMP=$(mktemp -d); sed -n "$((FIRST + 1)),$((FIRST + COUNT))p" "$TRAJ" > "$TMP/traj.txt"
  (cd "$TMP" && "$RENDERER" "$SRC/mesh.ply" "$SRC/textures" traj.txt "$ROOT/replica/intrinsic_128.txt" "$SRC/glass.sur")
  for ((k = 0; k < COUNT; k++)); do
    a=$(printf %06d $k); b=$(printf %06d $((FIRST + k)))
    mv "$TMP/frame$a.jpg" "$OUT/frame$b.jpg"; mv "$TMP/depth$a.png" "$OUT/depth$b.png"; mv "$TMP/normal$a.png" "$OUT/normal$b.png"
  done
  rm -r "$TMP"
else
  (cd "$OUT" && "$RENDERER" "$SRC/mesh.ply" "$SRC/textures" "$TRAJ" "$ROOT/replica/intrinsic_128.txt" "$SRC/glass.sur")
fi
echo "rendered $SCENE -> $OUT"
