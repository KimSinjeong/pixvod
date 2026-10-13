"""Live Rerun viewer stream of run_pixvod.py (PIXVOD_RERUN=1).  Output only: it reads values the runner has
already fetched to the host and never touches the device state, so the run's numbers are those of the run without it.

Adapted from the per-iteration Rerun logger of an earlier PixVOD runner: same entity tree, camera conventions and colours.  The runner stores extrinsics as Ecv (world -> camera, OpenCV); cameras
are logged as Twc = inv(Ecv) tagged RDF; the estimated twist is T_rel = Ecv_cur @ inv(Ecv_ref), so Twc_cur =
Twc_ref @ inv(T_rel).  The estimated depth lives on the reference pixel grid, so it is attached to the reference camera.

The viewer is started here, bound to 127.0.0.1 (rr.spawn() would bind 0.0.0.0, i.e. every interface), unless something
already listens on 127.0.0.1:PORT; every run then streams into that viewer as its own recording.  Timeline 'global_iter':
the initial depth preparation at -num_pre..-1, target iterations from 1.  Logged at the runner's existing log points:
every iteration_log_interval preparation iterations and the end of every target stage.

    world/trajectory_gt, trajectory_est        GT (red) and estimated (green) camera centres
    world/cam_ref                               reference camera: image, depth_gt, normal, depth_est, warped_cur,
                                                photo_residual, depth_residual
    world/cam_target_gt, cam_target_est         current target camera, GT (red) and estimated (green)
    world/pointcloud_gt, pointcloud_est         reference depth unprojected (GT, estimate scaled to GT)
    plots/...                                   geodesic errors (vs main = final target, vs current), twist errors,
                                                depth error and depth scale"""
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

PORT = 9876


def _guarded(fn):
    """A logging failure is reported once and stops the stream; the run itself continues."""
    def wrapper(self, *a, **k):
        if self.dead:
            return
        try:
            fn(self, *a, **k)
        except Exception as e:
            self.dead = True
            print(f'[RERUN] logging failed, stream stopped: {e!r}', flush=True)
    return wrapper


def _se3_exp_np(xi):
    """SE(3) exp map: xi = [wx, wy, wz, vx, vy, vz] (rotation first) -> 4x4."""
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    (omega, v) = (xi[:3], xi[3:6])
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        (R, V) = (np.eye(3), np.eye(3))
    else:
        k = omega / theta
        Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        (s, c) = (np.sin(theta), np.cos(theta))
        R = np.eye(3) + s * Kx + (1.0 - c) * (Kx @ Kx)
        V = np.eye(3) + (1.0 - c) / theta * Kx + (theta - s) / theta * (Kx @ Kx)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def _warp_cur_to_ref(rgb_cur, depth_ref, K, T_rel):
    """rgb_cur sampled at the projections of the reference pixels (centres i + 0.5) through depth_ref and T_rel."""
    import cv2
    (H, W) = depth_ref.shape[:2]
    (yy, xx) = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    p = np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K).T
    d = depth_ref.astype(np.float64)
    valid = np.isfinite(d) & (d > 0.0)
    P = np.concatenate([p * d[..., None], np.ones_like(d)[..., None]], axis=-1) @ T_rel.T
    z = P[..., 2]
    valid &= np.isfinite(z) & (z > 1e-6)
    q = P[..., :3] @ K.T
    zs = np.where(np.abs(z) > 1e-6, z, 1.0)
    # cv2.remap reads array index = coordinate, the projections are corner coordinates
    (map_x, map_y) = ((q[..., 0] / zs - 0.5).astype(np.float32), (q[..., 1] / zs - 0.5).astype(np.float32))
    img = (np.clip(rgb_cur, 0.0, 1.0) * 255.0).astype(np.uint8)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255.0
    warped[~valid] = 0.0
    return warped


def _points_world(depth, K, Twc, rgb, stride=2, max_depth=20.0):
    (H, W) = depth.shape[:2]
    (yy, xx) = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    (yy, xx, d, col) = (yy[::stride, ::stride], xx[::stride, ::stride], depth[::stride, ::stride].astype(np.float64), rgb[::stride, ::stride])
    valid = np.isfinite(d) & (d > 1e-3) & (d < max_depth)
    p = np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K).T
    P = (p * d[..., None])[valid]
    Pw = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1) @ Twc.T
    return (Pw[:, :3].astype(np.float32), (np.clip(col[valid], 0.0, 1.0) * 255.0).astype(np.uint8))


def _listening(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


class RerunLive:

    def __init__(self, app_id, K, H, W, Ecv_gt, Ecv_ref, rgb_ref, depth_ref, normal_ref, port=PORT):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        self.dead = False
        if not _listening(port):
            exe = shutil.which('rerun', path=os.path.dirname(sys.executable)) or shutil.which('rerun')
            if exe is None:
                raise RuntimeError('rerun viewer executable not found next to the interpreter or on PATH')
            subprocess.Popen([exe, '--bind', '127.0.0.1', '--port', str(port)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
            t0 = time.time()
            while not _listening(port):
                if time.time() - t0 > 30.0:
                    raise RuntimeError(f'rerun viewer did not open 127.0.0.1:{port} within 30 s (no display?)')
                time.sleep(0.2)
        rr.init(app_id, spawn=False)
        rr.connect_grpc(f'rerun+http://127.0.0.1:{port}/proxy')
        lines = [rr.SeriesLines(widths=2.0)]
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(name='World', origin='/world'), rrb.Grid(
            rrb.TimeSeriesView(name='Normalized geodesic error', origin='/plots', contents=['/plots/geo_err_main', '/plots/geo_err_cur'], defaults=lines),
            rrb.TimeSeriesView(name='Pose error norm (vs current)', origin='/plots', contents=['/plots/twist_err_norm_cur/rot', '/plots/twist_err_norm_cur/tr'], defaults=lines),
            rrb.TimeSeriesView(name='Pose error per axis (vs current)', origin='/plots/twist_err_cur', defaults=lines),
            rrb.TimeSeriesView(name='Depth error', origin='/plots/depth_err', defaults=lines), grid_columns=2), column_shares=[1, 1]),
            collapse_panels=True), make_active=True, make_default=True)
        print(f'[RERUN] streaming to the viewer at 127.0.0.1:{port} (recording {app_id})', flush=True)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        (self.H, self.W) = (int(H), int(W))
        self.Twc_ref = np.linalg.inv(np.asarray(Ecv_ref, dtype=np.float64))
        self.rgb_ref = np.asarray(rgb_ref, dtype=np.float32)[..., :3]
        self.depth_gt = np.asarray(depth_ref, dtype=np.float32).squeeze().copy()
        self.depth_gt[~np.isfinite(self.depth_gt) | (self.depth_gt <= 0.0)] = 0.0
        self.traj_est = []
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Replica trajectories are Z-up
        centers = np.asarray([np.linalg.inv(np.asarray(E, dtype=np.float64))[:3, 3] for E in Ecv_gt], dtype=np.float32)
        rr.log('world/trajectory_gt', rr.LineStrips3D([centers], colors=[[255, 0, 0]], radii=[0.0025]), static=True)
        rr.log('world/trajectory_gt/points', rr.Points3D(centers, colors=[[255, 0, 0]], radii=0.005), static=True)
        self._camera('world/cam_ref', self.Twc_ref, [200, 200, 200, 255], static=True)
        rr.log('world/cam_ref/image', rr.Image((np.clip(self.rgb_ref, 0.0, 1.0) * 255.0).astype(np.uint8)), static=True)
        rr.log('world/cam_ref/depth_gt', rr.DepthImage(self.depth_gt, meter=1.0, colormap=rr.components.Colormap.Turbo), static=True)
        if normal_ref is not None:
            n = np.asarray(normal_ref, dtype=np.float32)
            rr.log('world/cam_ref/normal', rr.Image(((n * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)), static=True)
        (pts, cols) = _points_world(self.depth_gt, self.K, self.Twc_ref, self.rgb_ref)
        rr.log('world/pointcloud_gt', rr.Points3D(pts, colors=cols, radii=0.005), static=True)

    def _camera(self, path, Twc, color, static=False):
        rr = self.rr
        rr.log(path, rr.Transform3D(translation=Twc[:3, 3], mat3x3=Twc[:3, :3]), static=static)
        rr.log(path, rr.Pinhole(image_from_camera=self.K.astype(np.float32), resolution=[self.W, self.H], camera_xyz=rr.ViewCoordinates.RDF,
                                image_plane_distance=0.1), static=static)
        rr.log(f'{path}/center', rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=0.01), static=static)  # Pinhole has no colour in 0.23

    def _depth(self, depth_est):
        rr = self.rr
        ed = np.asarray(depth_est, dtype=np.float32).copy()
        ed[~np.isfinite(ed)] = 0.0
        ed = np.clip(ed, 0.0, 20.0)
        rr.log('world/cam_ref/depth_est', rr.DepthImage(ed, meter=1.0, colormap=rr.components.Colormap.Viridis))
        valid = self.depth_gt > 0.0
        rr.log('world/cam_ref/depth_residual', rr.Image(np.clip((np.where(valid, ed - self.depth_gt, 0.0) + 1.0) * 0.5, 0.0, 1.0)))
        ok = valid & (ed > 0.0)
        if np.any(ok):
            de = np.abs(ed[ok] - self.depth_gt[ok])
            rr.log('plots/depth_err/median_m', rr.Scalars(float(np.median(de))))
            rr.log('plots/depth_err/median_rel', rr.Scalars(float(np.median(de / np.maximum(self.depth_gt[ok], 1e-6)))))
        (pts, cols) = _points_world(ed, self.K, self.Twc_ref, self.rgb_ref)
        rr.log('world/pointcloud_est', rr.Points3D(pts, colors=cols, radii=0.004))
        return ed

    @_guarded
    def log_prewarm(self, it, num_pre, depth_est_scaled):
        """Initial depth preparation, iteration it of num_pre (timeline it - num_pre - 1: -num_pre .. -1)."""
        self.rr.set_time('global_iter', sequence=int(it) - int(num_pre) - 1)
        self._depth(depth_est_scaled)
        self.rr.log('plots/stage_label', self.rr.TextLog(f'prewarm it={it}/{num_pre}'))

    @_guarded
    def log_stage(self, global_iter, stage_label, est_twist_scaled, gt_twist_eval, gt_twist_cur, Ecv_target_gt, depth_est_scaled,
                  depth_scale, norm_err_main, norm_err_cur, rgb_cur):
        """End of a target stage: estimated and GT target cameras, depth, warp residual and the error series."""
        rr = self.rr
        rr.set_time('global_iter', sequence=int(global_iter))
        est = np.asarray(est_twist_scaled, dtype=np.float64).reshape(-1)
        T_rel = _se3_exp_np(est)
        Twc_est = self.Twc_ref @ np.linalg.inv(T_rel)
        self._camera('world/cam_target_est', Twc_est, [0, 220, 0, 255])
        self._camera('world/cam_target_gt', np.linalg.inv(np.asarray(Ecv_target_gt, dtype=np.float64)), [230, 0, 0, 255])
        self.traj_est.append(Twc_est[:3, 3].astype(np.float32))
        traj = np.asarray(self.traj_est, dtype=np.float32)
        if traj.shape[0] >= 2:
            rr.log('world/trajectory_est', rr.LineStrips3D([traj], colors=[[0, 220, 0]], radii=[0.0025]))
        rr.log('world/trajectory_est/points', rr.Points3D(traj, colors=[[0, 255, 0]], radii=0.005))
        ed = self._depth(depth_est_scaled)
        warped = _warp_cur_to_ref(np.asarray(rgb_cur, dtype=np.float32)[..., :3], ed, self.K, T_rel)
        rr.log('world/cam_ref/warped_cur', rr.Image((np.clip(warped, 0.0, 1.0) * 255.0).astype(np.uint8)))
        res = np.abs(warped - np.clip(self.rgb_ref, 0.0, 1.0)).mean(axis=-1)
        rr.log('world/cam_ref/photo_residual', rr.Image((np.clip(res, 0.0, 1.0) * 255.0).astype(np.uint8)))
        rr.log('plots/geo_err_main', rr.Scalars(float(norm_err_main)))
        rr.log('plots/geo_err_cur', rr.Scalars(float(norm_err_cur)))
        rr.log('plots/depth_scale', rr.Scalars(float(depth_scale)))
        (ge, gc) = (np.asarray(gt_twist_eval, dtype=np.float64).reshape(-1), np.asarray(gt_twist_cur, dtype=np.float64).reshape(-1))
        for (i, n) in enumerate(['rx', 'ry', 'rz', 'tx', 'ty', 'tz']):
            rr.log(f'plots/est_twist/{n}', rr.Scalars(float(est[i])))
            rr.log(f'plots/twist_err_cur/{n}', rr.Scalars(float(est[i] - gc[i])))
            rr.log(f'plots/twist_err_eval/{n}', rr.Scalars(float(est[i] - ge[i])))
        rr.log('plots/twist_err_norm_cur/rot', rr.Scalars(float(np.linalg.norm(est[:3] - gc[:3]))))
        rr.log('plots/twist_err_norm_cur/tr', rr.Scalars(float(np.linalg.norm(est[3:] - gc[3:]))))
        rr.log('plots/stage_label', rr.TextLog(str(stage_label)))
