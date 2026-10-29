"""
MOT stage (2026-10-23): shared pieces for the online trackers in common/mot/trackers.py.

Box convention everywhere: xyxy in input-image pixels (the detector output after post-processing).
Tracker output per frame: float array [M, 7] = x1, y1, x2, y2, track_id, class, score.

Two Kalman filters, each as in the official code of the trackers that use it:
  KalmanXYSR  - SORT / OC-SORT (filterpy KalmanFilter with the SORT matrices), state [u, v, s, r, du, dv, ds],
                with OC-SORT's freeze / unfreeze (ORU: re-update along a virtual trajectory after a gap).
  KalmanXYAH  - ByteTrack / BoT-SORT / DeepSORT (state [cx, cy, a, h, vx, vy, va, vh], noise relative to h).
assign() reproduces lap.lapjv(cost, extend_cost=True, cost_limit=thr) used by ByteTrack: a pair is matched only if
its cost is below thr, and the assignment is optimal under that limit (dummy rows / cols with cost thr / 2).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6


def iou_matrix(a, b):
    """a [N, 4], b [M, 4] xyxy -> IoU [N, M]."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost, thr=None):
    """-> matches [K, 2] (row, col), unmatched rows, unmatched cols. thr = lapjv cost_limit (None = no limit)."""
    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), int), np.arange(n), np.arange(m)
    if thr is None:
        r, c = linear_sum_assignment(cost)
    else:
        big = np.full((n + m, n + m), BIG)
        big[:n, :m] = cost
        big[:n, m:][np.arange(n), np.arange(n)] = thr / 2.0
        big[n:, :m][np.arange(m), np.arange(m)] = thr / 2.0
        big[n:, m:] = 0.0
        r, c = linear_sum_assignment(big)
        keep = (r < n) & (c < m)
        r, c = r[keep], c[keep]
        ok = cost[r, c] < thr
        r, c = r[ok], c[ok]
    matches = np.stack([r, c], 1).astype(int) if len(r) else np.empty((0, 2), int)
    return matches, np.setdiff1d(np.arange(n), r), np.setdiff1d(np.arange(m), c)


# ------------------------------------------------------------------ SORT / OC-SORT filter
def xyxy_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w * h, w / float(h + 1e-6)]).reshape(4, 1)


def x_to_xyxy(x):
    w = np.sqrt(max(x[2, 0] * x[3, 0], 0.0))
    h = x[2, 0] / w if w > 0 else 0.0
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


class KalmanXYSR:
    """filterpy.KalmanFilter(dim_x=7, dim_z=4) with the SORT set-up; freeze/unfreeze as OC-SORT's kalmanfilter.py."""

    def __init__(self, box):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1))
        self.x[:4] = xyxy_to_z(box)
        self.history_obs, self.observed, self.saved = [], False, None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.shape[0]) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T          # filterpy: Joseph form

    def update(self, z, oru=False):
        """z [4, 1] or None. oru=True: OC-SORT observation-centric re-update after a gap."""
        if not oru:
            if z is not None:
                self._update(z)
            return
        self.history_obs.append(z)
        if z is None:
            if self.observed:
                self.saved = {k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
                              for k, v in self.__dict__.items() if k != "saved"}
            self.observed = False
            return
        if not self.observed and self.saved is not None:
            self._unfreeze()
        self.observed = True
        self._update(z)

    def _unfreeze(self):
        new_history = list(self.history_obs)
        for k, v in self.saved.items():
            setattr(self, k, v)
        self.history_obs = self.history_obs[:-1]
        idx = [i for i, d in enumerate(new_history) if d is not None]
        if len(idx) < 2:
            return
        i1, i2 = idx[-2], idx[-1]
        x1, y1, s1, r1 = new_history[i1].ravel()
        x2, y2, s2, r2 = new_history[i2].ravel()
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        gap = i2 - i1
        dx, dy, dw, dh = (x2 - x1) / gap, (y2 - y1) / gap, (w2 - w1) / gap, (h2 - h1) / gap
        for i in range(gap):
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            box = np.array([x1 + (i + 1) * dx, y1 + (i + 1) * dy, w * h, w / float(h)]).reshape(4, 1)
            self.history_obs.append(box)          # official: update(new_box) appends to history_obs
            self.observed = True
            self._update(box)
            if i != gap - 1:
                self.predict()


# ------------------------------------------------------------------ ByteTrack / BoT-SORT filter
class KalmanXYAH:
    ndim, w_pos, w_vel = 4, 1.0 / 20, 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h,
               self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def project(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]
        return self.H @ mean, self.H @ cov @ self.H.T + np.diag(np.square(std))

    def update(self, mean, cov, z):
        pm, pc = self.project(mean, cov)
        K = np.linalg.solve(pc, (cov @ self.H.T).T).T          # = cov H^T pc^-1 (pc symmetric)
        return mean + (z - pm) @ K.T, cov - K @ pc @ K.T


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w / max(h, 1e-6), h])


def xyah_to_xyxy(m):
    w = m[2] * m[3]
    return np.array([m[0] - w / 2.0, m[1] - m[3] / 2.0, m[0] + w / 2.0, m[1] + m[3] / 2.0])


# ------------------------------------------------------------------ BoT-SORT filter (state x, y, w, h)
class KalmanXYWH(KalmanXYAH):
    """BoT-SORT tracker/kalman_filter.py: same model as KalmanXYAH, noise proportional to w (x, w) and h (y, h)."""

    def initiate(self, z):
        w, h = z[2], z[3]
        p, v = self.w_pos, self.w_vel
        std = [2 * p * w, 2 * p * h, 2 * p * w, 2 * p * h, 10 * v * w, 10 * v * h, 10 * v * w, 10 * v * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, mean, cov):
        w, h = mean[2], mean[3]
        p, v = self.w_pos, self.w_vel
        std = [p * w, p * h, p * w, p * h, v * w, v * h, v * w, v * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def project(self, mean, cov):
        w, h = mean[2], mean[3]
        std = [self.w_pos * w, self.w_pos * h, self.w_pos * w, self.w_pos * h]
        return self.H @ mean, self.H @ cov @ self.H.T + np.diag(np.square(std))


def xyxy_to_xywh(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b[2] - b[0], b[3] - b[1]])


def xywh_to_xyxy(m):
    return np.array([m[0] - m[2] / 2.0, m[1] - m[3] / 2.0, m[0] + m[2] / 2.0, m[1] + m[3] / 2.0])


CHI2INV95_4 = 9.4877                      # DeepSORT kalman_filter.chi2inv95[4]


def gating_distance(kf, mean, cov, z):
    """squared Mahalanobis distance of measurements z [N, 4] (xyah) to the projected state (DeepSORT)."""
    pm, pc = kf.project(mean, cov)
    L = np.linalg.cholesky(pc)
    d = (np.asarray(z) - pm).T
    sol = np.linalg.solve(L, d)
    return (sol * sol).sum(0)


# ------------------------------------------------------------------ camera motion (BoT-SORT gmc.py, sparseOptFlow)
class GMCSparseOptFlow:
    """Global motion between consecutive frames: corners on the half-size gray frame, pyramidal LK flow, RANSAC
    partial affine (rotation, uniform scale, translation). Uses only the previous and the current frame (causal)."""

    def __init__(self, downscale=2):
        import cv2
        self.cv2, self.downscale = cv2, downscale
        self.params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                           useHarrisDetector=False, k=0.04)
        self.prev = self.prev_kp = None

    def apply(self, bgr):
        cv2 = self.cv2
        H = np.eye(2, 3)
        h, w = bgr.shape[:2]
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            frame = cv2.resize(frame, (w // self.downscale, h // self.downscale))
        kp = cv2.goodFeaturesToTrack(frame, mask=None, **self.params)
        if self.prev is None:
            self.prev, self.prev_kp = frame.copy(), kp
            return H
        if self.prev_kp is not None and len(self.prev_kp):
            cur, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, frame, self.prev_kp, None)
            ok = status.ravel() == 1
            p0, p1 = self.prev_kp[ok], cur[ok]
            if len(p0) > 4:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
                if M is not None:
                    H = M
                    if self.downscale > 1:
                        H[0, 2] *= self.downscale
                        H[1, 2] *= self.downscale
        self.prev, self.prev_kp = frame.copy(), kp
        return H


# ------------------------------------------------------------------ Hybrid-SORT filter (state x, y, s, score, r + velocities)
class KalmanHybrid(KalmanXYSR):
    """HybridSORT kalmanfilter_score_new (dim_x 9, dim_z 5): the detection confidence is part of the state (TCM);
    same set-up as SORT for the box, ORU (freeze / unfreeze) with the confidence interpolated as well."""

    def __init__(self, det):
        self.F = np.eye(9)
        for i in range(4):
            self.F[i, 5 + i] = 1.0
        self.H = np.zeros((5, 9))
        self.H[:5, :5] = np.eye(5)
        self.R = np.eye(5)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(9)
        self.P[5:, 5:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(9)
        self.Q[-1, -1] *= 0.01
        self.Q[-2, -2] *= 0.01
        self.Q[5:, 5:] *= 0.01
        self.x = np.zeros((9, 1))
        self.x[:5] = hybrid_z(det)
        self.history_obs, self.observed, self.saved = [], False, None

    def _unfreeze(self):
        new_history = list(self.history_obs)
        for k, v in self.saved.items():
            setattr(self, k, v)
        self.history_obs = self.history_obs[:-1]
        idx = [i for i, d in enumerate(new_history) if d is not None]
        if len(idx) < 2:
            return
        i1, i2 = idx[-2], idx[-1]
        x1, y1, s1, c1, r1 = new_history[i1].ravel()
        x2, y2, s2, c2, r2 = new_history[i2].ravel()
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        gap = i2 - i1
        d = [(x2 - x1) / gap, (y2 - y1) / gap, (w2 - w1) / gap, (h2 - h1) / gap, (c2 - c1) / gap]
        for i in range(gap):
            w, h = w1 + (i + 1) * d[2], h1 + (i + 1) * d[3]
            box = np.array([x1 + (i + 1) * d[0], y1 + (i + 1) * d[1], w * h, c1 + (i + 1) * d[4],
                            w / float(h)]).reshape(5, 1)
            self.history_obs.append(box)
            self.observed = True
            self._update(box)
            if i != gap - 1:
                self.predict()


def hybrid_z(det):
    w, h = det[2] - det[0], det[3] - det[1]
    return np.array([det[0] + w / 2.0, det[1] + h / 2.0, w * h, det[4], w / float(h + 1e-6)]).reshape(5, 1)


def hybrid_x_to_xyxy(x):
    w = np.sqrt(max(x[2, 0] * x[4, 0], 0.0))
    h = x[2, 0] / w if w > 0 else 0.0
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


def hmiou(a, b):
    """HybridSORT association.hmiou: IoU x vertical overlap ratio (height-modulated IoU)."""
    a, b = np.asarray(a, np.float64)[:, None, :4], np.asarray(b, np.float64)[None, :, :4]
    o = (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])) / \
        (np.maximum(a[..., 3], b[..., 3]) - np.minimum(a[..., 1], b[..., 1]))
    w = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    h = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    wh = w * h
    return o * wh / ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - wh)


# ------------------------------------------------------------------ TrackTrack filter (cx, cy, w, h; NSA-style)
class KalmanTT:
    """TrackTrack trackers/kalman_filter.py: noise proportional to w / h, measurement noise x (1 - confidence)."""

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)
        self.motion_cov = np.eye(8)
        self.motion_cov[:4, :4] *= 1.0 / 20
        self.motion_cov[4:, 4:] *= 1.0 / 160
        self.innov = np.eye(4) * (1.0 / 20)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        cov = np.eye(8)
        cov[:4, :4] *= 2.0
        cov[4:, 4:] *= 10.0
        cov[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        cov[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        return mean, np.square(cov)

    def predict(self, mean, cov):
        mean = self.F @ mean
        mc = self.motion_cov.copy()
        mc[[0, 2, 4, 6], [0, 2, 4, 6]] *= mean[2]
        mc[[1, 3, 5, 7], [1, 3, 5, 7]] *= mean[3]
        return mean, self.F @ cov @ self.F.T + np.square(mc)

    def update(self, mean, cov, z, conf):
        pm = self.H @ mean
        ic = self.innov.copy()
        ic[[0, 2], [0, 2]] *= pm[2]
        ic[[1, 3], [1, 3]] *= pm[3]
        pc = self.H @ cov @ self.H.T + np.square(ic) * (1 - conf)
        K = np.linalg.solve(pc, (cov @ self.H.T).T).T
        return mean + (z - pm) @ K.T, cov - K @ pc @ K.T


def iou_plus1(a, b):
    """TrackTrack bbox_overlaps (pixel-inclusive +1 convention)."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    bb = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return np.where(inter > 0, inter / (aa[:, None] + bb[None] - inter), 0.0)
