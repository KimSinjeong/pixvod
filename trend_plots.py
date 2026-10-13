"""Trend figures of run_pixvod.py (twist_geo_norm_error_iter_combined.png, pose_axes_trend_combined.png).

save_combined_plots is the body of TrajectoryPairRunner._save_combined_plots, moved here unchanged so that the same code
can run in a separate process; this module imports neither jax nor the pipeline.

PIXVOD_ASYNC_PLOT=1: run_pixvod.py writes the CSV itself and hands a snapshot of the plotted lists to
AsyncTrendPlots, whose worker (this file run as a script, started with subprocess before the GPU is initialised, i.e.
fork+exec of a fresh interpreter, never a bare fork) draws them.  A sender thread keeps only the newest snapshot not yet
sent, so a snapshot can be skipped when a newer one arrives while the worker is drawing; the last one submitted is always
drawn, and close() returns only after the worker has drawn it and exited.  Every figure on disk at the end is therefore
the figure the synchronous path draws from the same final data.  A failure in the worker is raised in the main process
at the next submit() or at close()."""
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

_FIELDS = ('_comb_iters', '_comb_norm_err', '_comb_norm_err_cur', '_comb_est', '_comb_est_std', '_comb_root_est',
           '_comb_root_err_cur', '_comb_gt', '_comb_gt_cur', '_comb_is_sub')


def save_combined_plots(self) -> None:
    """Render combined error and pose-per-axis plots with dashed (subframe) and solid (main) segments."""
    if len(self._comb_iters) == 0:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (fig_err, ax_err) = plt.subplots(1, 1, figsize=(10, 4))
    xs = np.asarray(self._comb_iters, dtype=np.int32)
    ys_main = np.asarray(self._comb_norm_err, dtype=np.float32)
    ys_cur = np.asarray(self._comb_norm_err_cur, dtype=np.float32)
    ys_root_cur = np.asarray(self._comb_root_err_cur, dtype=np.float32) if len(self._comb_root_err_cur) == xs.size else None
    for i in range(1, xs.size):
        ls = '--' if bool(self._comb_is_sub[i]) else '-'
        ax_err.plot(xs[i - 1:i + 1], ys_cur[i - 1:i + 1], color='b', linestyle=ls, label='error vs current GT' if i == 1 else None)
        ax_err.plot(xs[i - 1:i + 1], ys_main[i - 1:i + 1], color='g', linestyle=ls, label='error vs next main GT' if i == 1 else None)
        if ys_root_cur is not None:
            ax_err.plot(xs[i - 1:i + 1], ys_root_cur[i - 1:i + 1], color='k', linestyle=ls, label='error vs current GT (root)' if i == 1 else None)
    if xs.size == 1:
        ax_err.scatter(xs, ys_cur, color='b', s=8, label='error vs current GT')
        ax_err.scatter(xs, ys_main, color='g', s=8, label='error vs next main GT')
        if ys_root_cur is not None:
            ax_err.scatter(xs, ys_root_cur, color='k', s=8, label='error vs current GT (root)')
    ax_err.set_xlabel('Iteration')
    ax_err.set_ylabel('Normalized geodesic error (SE3)')
    ax_err.set_title('SE(3) normalized error: current GT (blue) vs next main GT (green)')
    ax_err.set_ylim(bottom=0.0, top=1.0)
    ax_err.legend(loc='best')
    fig_err.tight_layout()
    fig_err.savefig(self.logs_dir / 'twist_geo_norm_error_iter_combined.png')
    plt.close(fig_err)
    est_full = np.asarray(self._comb_est, dtype=np.float32)
    est_std_full = np.asarray(self._comb_est_std, dtype=np.float32) if len(self._comb_est_std) == len(self._comb_est) else None
    gt_full = np.asarray(self._comb_gt, dtype=np.float32)
    root_full = np.asarray(self._comb_root_est, dtype=np.float32) if len(self._comb_root_est) == est_full.shape[0] else None
    gt_cur_full = np.asarray(self._comb_gt_cur, dtype=np.float32)
    names = ['wx', 'wy', 'wz', 'tx', 'ty', 'tz']
    dmax = min(est_full.shape[1], 6)
    (fig_ax, axes) = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    axes = axes.flatten()
    for a in range(dmax):
        ax = axes[a]
        for i in range(1, xs.size):
            ls = '--' if bool(self._comb_is_sub[i]) else '-'
            ax.plot(xs[i - 1:i + 1], est_full[i - 1:i + 1, a], color='g', linestyle=ls)
        if xs.size == 1:
            ax.scatter(xs, est_full[:, a], color='g', s=8)
        if root_full is not None and root_full.shape[0] == xs.size:
            for i in range(1, xs.size):
                ls = '--' if bool(self._comb_is_sub[i]) else '-'
                ax.plot(xs[i - 1:i + 1], root_full[i - 1:i + 1, a], color='k', linestyle=ls, label='root est' if i == 1 and a == 0 else None)
            if xs.size == 1:
                ax.scatter(xs, root_full[:, a], color='k', s=8, label='root est' if a == 0 else None)
        if est_std_full is not None and xs.size >= 2 and (est_std_full.shape[0] == xs.size):
            y_lower = est_full[:, a] - est_std_full[:, a]
            y_upper = est_full[:, a] + est_std_full[:, a]
            ax.fill_between(xs, y_lower, y_upper, color='g', alpha=0.2, linewidth=0)
        if gt_full.shape[0] > 0:
            ax.plot([xs[0], xs[-1]], [gt_full[0, a], gt_full[0, a]], label='gt main', color='r', linestyle='--')
        if gt_cur_full.shape[0] == xs.size and gt_cur_full.shape[1] >= a + 1:
            for i in range(1, xs.size):
                ls = '--' if bool(self._comb_is_sub[i]) else '-'
                ax.plot(xs[i - 1:i + 1], gt_cur_full[i - 1:i + 1, a], color='b', linestyle=ls, label='gt current' if i == 1 else None)
        ax.set_title(names[a])
        ax.grid(True, linestyle=':', linewidth=0.5)
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlabel('Iteration')
    fig_ax.tight_layout()
    fig_ax.savefig(self.logs_dir / 'pose_axes_trend_combined.png')
    plt.close(fig_ax)
    self._save_combined_csv()


def snapshot(runner) -> dict:
    """The data save_combined_plots reads, copied (the lists only grow; their elements are never modified)."""
    d = {k: list(getattr(runner, k)) for k in _FIELDS}
    d['logs_dir'] = str(runner.logs_dir)
    return d


class _Snap:
    """Stand-in for the runner inside the worker; the CSV stays with the main process."""

    def __init__(self, d):
        self.__dict__.update(d)
        self.logs_dir = Path(d['logs_dir'])

    def _save_combined_csv(self):
        pass


CLOSE_TIMEOUT_S = 600  # a worker stuck this long (e.g. inside matplotlib) is killed and reported


class AsyncTrendPlots:
    def __init__(self):
        # own session: a terminal Ctrl-C reaches only the main process, which then still waits for the last figure
        self._proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, close_fds=True, start_new_session=True)
        self._cv = threading.Condition()
        self._pending = None
        self._closing = False
        self._error = None
        self.n_submitted = 0
        self.n_drawn = 0
        self._thread = threading.Thread(target=self._loop, name='trend-plot-sender', daemon=True)
        self._thread.start()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f'[ASYNC-PLOT] trend figure worker failed: {self._error}')

    def submit(self, snap: dict) -> None:
        self._raise_if_failed()
        with self._cv:
            if self._closing:
                raise RuntimeError('[ASYNC-PLOT] submit after close')
            self._pending = snap
            self.n_submitted += 1
            self._cv.notify()

    def _loop(self):
        try:
            while True:
                with self._cv:
                    while self._pending is None and not self._closing:
                        self._cv.wait()
                    if self._pending is None:
                        break
                    snap, self._pending = self._pending, None
                data = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
                self._proc.stdin.write(len(data).to_bytes(8, 'big') + data)
                self._proc.stdin.flush()
                ack = self._proc.stdout.readline().decode(errors='replace').rstrip('\n')
                if ack != 'ok':
                    self._error = ack or f'worker exited (code {self._proc.poll()})'
                    break
                self.n_drawn += 1
        except Exception as e:  # broken pipe etc.
            self._error = repr(e)
        finally:
            try:
                self._proc.stdin.write((0).to_bytes(8, 'big'))
                self._proc.stdin.close()
            except Exception:
                pass

    def close(self) -> None:
        """Draw the last snapshot, wait for the worker to exit, raise if anything failed."""
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join(timeout=CLOSE_TIMEOUT_S)
        try:
            rc = self._proc.wait(timeout=5 if self._thread.is_alive() else CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            rc = self._proc.wait()
            if self._error is None:
                self._error = f'worker did not finish within {CLOSE_TIMEOUT_S} s (killed)'
        if self._error is None and rc != 0:
            self._error = f'worker exit code {rc}'
        self._raise_if_failed()


def _worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process decides when drawing stops (it closes stdin)
    try:
        os.nice(5)  # drawing must not take CPU time from the process that drives the GPU
    except OSError:
        pass
    inp = sys.stdin.buffer
    out = sys.stdout
    while True:
        head = inp.read(8)
        if len(head) < 8:
            return
        n = int.from_bytes(head, 'big')
        if n == 0:
            return
        snap = pickle.loads(inp.read(n))
        try:
            save_combined_plots(_Snap(snap))
            out.write('ok\n')
        except Exception as e:
            out.write('error ' + repr(e).replace('\n', ' ') + '\n')
        out.flush()


if __name__ == '__main__':
    _worker()
