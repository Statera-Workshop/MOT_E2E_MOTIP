"""
MOT stage (2026-10-23): online (causal) trackers, re-implemented after the official code, official defaults.

    from common.mot.trackers import make_tracker
    trk = make_tracker("bytetrack", frame_rate=10)
    for frame in video:
        out = trk.update(dets)          # dets [N, 6] = x1, y1, x2, y2, score, class   ->  [M, 7] x1..y2, id, class, score

Every tracker sees only the current frame and its own state (no look-ahead, no post-processing): the causality
check in tools/mot/run_tracker.py --causal verifies that the output of frame t does not change when later frames
are removed.

[Unified] decisions (recorded in the progress tab):
  * classes are tracked independently: one tracker instance per class (PerClass), ids made unique across classes.
  * frame-based parameters given for 30 fps are scaled to the sequence frame rate (KITTI 10 fps) the way ByteTrack
    does it (buffer * frame_rate / 30): ByteTrack 30 -> 10 lost frames, OC-SORT max_age 30 -> 10.
  * SORT and the IoU baseline have no score threshold of their own -> 0.5 (ByteTrack's track_thresh).
  * no box-area / aspect-ratio filtering of the output (ByteTrack's MOT17 pedestrian filter does not fit cars).

Sources (official defaults):
  IoU baseline  Bochinski et al. AVSS 2017, online form: greedy IoU >= 0.5 to the last box, no motion model.
  SORT          abewley/sort sort.py: max_age 1, min_hits 3, iou_threshold 0.3.
  ByteTrack     ifzhang/ByteTrack byte_tracker.py: track_thresh 0.5, low 0.1, match_thresh 0.8 (score fused),
                second match 0.5, unconfirmed 0.7, new track >= track_thresh + 0.1, track_buffer 30.
  OC-SORT       noahcao/OC_SORT ocsort.py: det_thresh 0.6, max_age 30, min_hits 3, iou 0.3, delta_t 3,
                inertia 0.2 (OCM), OCR second round on last observations, ORU in the Kalman filter.
"""

import numpy as np

from common.mot.core import (CHI2INV95_4, KalmanHybrid, KalmanTT, KalmanXYAH, KalmanXYSR, KalmanXYWH, assign,
                             gating_distance, hmiou, hybrid_x_to_xyxy, hybrid_z, iou_matrix, iou_plus1, x_to_xyxy,
                             xyah_to_xyxy, xywh_to_xyxy, xyxy_to_xyah, xyxy_to_xywh, xyxy_to_z)

EMPTY = np.zeros((0, 7))


def _out(rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 7) if rows else EMPTY.copy()


# ====================================================================== IoU baseline
class IoUTracker:
    def __init__(self, frame_rate=10, score_thr=0.5, iou_thr=0.5, min_len=2):
        self.score_thr, self.iou_thr, self.min_len = score_thr, iou_thr, min_len
        self.tracks, self.next_id = [], 1                       # track: dict(box, id, len, score)

    def update(self, dets):
        dets = dets[dets[:, 4] >= self.score_thr] if len(dets) else dets
        new_tracks, used = [], np.zeros(len(dets), bool)
        if self.tracks and len(dets):
            iou = iou_matrix([t["box"] for t in self.tracks], dets[:, :4])
            for ti in np.argsort(-iou.max(1)):                  # greedy, best overlap first
                cand = np.where(~used & (iou[ti] >= self.iou_thr))[0]
                if len(cand):
                    d = cand[np.argmax(iou[ti, cand])]
                    used[d] = True
                    t = self.tracks[ti]
                    new_tracks.append({**t, "box": dets[d, :4], "len": t["len"] + 1, "score": dets[d, 4]})
        for d in np.where(~used)[0]:
            new_tracks.append({"box": dets[d, :4], "id": self.next_id, "len": 1, "score": dets[d, 4]})
            self.next_id += 1
        self.tracks = new_tracks                                # unmatched tracks end at once
        return _out([[*t["box"], t["id"], -1, t["score"]] for t in self.tracks if t["len"] >= self.min_len])


# ====================================================================== SORT
class _SortTrack:
    def __init__(self, det, tid, oru=False, delta_t=3):
        self.kf = KalmanXYSR(det[:4])
        self.id = tid                                           # ids per tracker instance (reproducible runs)
        self.time_since_update, self.hits, self.hit_streak, self.age = 0, 0, 0, 0
        self.score = det[4]
        self.oru, self.delta_t = oru, delta_t
        self.last_obs = np.full(5, -1.0)                        # OC-SORT
        self.observations, self.velocity = {}, None             # kf.observed starts False as in OC-SORT

    def predict(self):
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_xyxy(self.kf.x)

    def update(self, det):
        if det is None:                                         # OC-SORT only: unmatched -> ORU bookkeeping
            self.kf.update(None, oru=self.oru)
            return
        if self.oru:
            if self.last_obs.sum() >= 0:
                prev = None
                for i in range(self.delta_t):
                    if self.age - (self.delta_t - i) in self.observations:
                        prev = self.observations[self.age - (self.delta_t - i)]
                        break
                self.velocity = speed_direction(self.last_obs if prev is None else prev, det)
            self.last_obs = np.r_[det[:4], det[4]]
            self.observations[self.age] = self.last_obs
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = det[4]
        self.kf.update(xyxy_to_z(det[:4]), oru=self.oru)

    def state(self):
        return x_to_xyxy(self.kf.x)


def _associate_iou(dets, trks, thr):
    iou = iou_matrix(dets[:, :4], trks)
    if min(iou.shape) == 0:
        return np.empty((0, 2), int), np.arange(len(dets)), np.arange(len(trks))
    a = (iou > thr).astype(int)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        m = np.stack(np.where(a), 1)
    else:
        m, _, _ = assign(-iou)
    good = iou[m[:, 0], m[:, 1]] >= thr if len(m) else np.zeros(0, bool)
    m = m[good]
    return m, np.setdiff1d(np.arange(len(dets)), m[:, 0]), np.setdiff1d(np.arange(len(trks)), m[:, 1])


class SORT:
    def __init__(self, frame_rate=10, max_age=1, min_hits=3, iou_threshold=0.3, score_thr=0.5):
        self.max_age, self.min_hits, self.iou_thr, self.score_thr = max_age, min_hits, iou_threshold, score_thr
        self.trackers, self.frame_count, self.next_id = [], 0, 0

    def update(self, dets):
        self.frame_count += 1
        dets = dets[dets[:, 4] >= self.score_thr] if len(dets) else dets.reshape(0, 6)
        trks = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        keep = ~np.isnan(trks).any(1)
        self.trackers = [t for t, k in zip(self.trackers, keep) if k]
        trks = trks[keep]
        m, ud, _ = _associate_iou(dets, trks, self.iou_thr)
        for d, t in m:
            self.trackers[t].update(dets[d])
        for d in ud:
            self.next_id += 1
            self.trackers.append(_SortTrack(dets[d], self.next_id))
        rows = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                rows.append([*t.state(), t.id, -1, t.score])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return _out(rows)


# ====================================================================== OC-SORT
def speed_direction(b1, b2):
    c1 = np.array([(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])
    c2 = np.array([(b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0])
    s = np.array([c2[1] - c1[1], c2[0] - c1[0]])               # (dy, dx) as the official code
    return s / (np.sqrt((s ** 2).sum()) + 1e-6)


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return np.full(5, -1.0)
    for i in range(k):
        if cur_age - (k - i) in observations:
            return observations[cur_age - (k - i)]
    return observations[max(observations.keys())]


class OCSORT:
    def __init__(self, frame_rate=10, det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3,
                 inertia=0.2):
        self.max_age = max(1, int(round(max_age * frame_rate / 30.0)))
        self.det_thresh, self.min_hits, self.iou_thr = det_thresh, min_hits, iou_threshold
        self.delta_t, self.inertia = delta_t, inertia
        self.trackers, self.frame_count, self.next_id = [], 0, 0

    def _associate(self, dets, trks, velocities, prev_obs):
        if len(trks) == 0 or len(dets) == 0:
            return np.empty((0, 2), int), np.arange(len(dets)), np.arange(len(trks))
        c_d = np.stack([(dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0], 1)
        c_t = np.stack([(prev_obs[:, 0] + prev_obs[:, 2]) / 2.0, (prev_obs[:, 1] + prev_obs[:, 3]) / 2.0], 1)
        dx = c_d[None, :, 0] - c_t[:, None, 0]                  # [T, D]
        dy = c_d[None, :, 1] - c_t[:, None, 1]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        X, Y = dx / norm, dy / norm
        cos = np.clip(velocities[:, 1:2] * X + velocities[:, 0:1] * Y, -1, 1)
        diff = (np.pi / 2.0 - np.abs(np.arccos(cos))) / np.pi
        valid = (prev_obs[:, 4] >= 0).astype(float)[:, None]
        angle_cost = (valid * diff * self.inertia).T * dets[:, 4:5]          # [D, T]
        iou = iou_matrix(dets[:, :4], trks)
        a = (iou > self.iou_thr).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            m = np.stack(np.where(a), 1)
        else:
            m, _, _ = assign(-(iou + angle_cost))
        good = iou[m[:, 0], m[:, 1]] >= self.iou_thr if len(m) else np.zeros(0, bool)
        m = m[good]
        return m, np.setdiff1d(np.arange(len(dets)), m[:, 0]), np.setdiff1d(np.arange(len(trks)), m[:, 1])

    def update(self, dets):
        self.frame_count += 1
        dets = dets[dets[:, 4] > self.det_thresh] if len(dets) else dets.reshape(0, 6)
        trks = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        keep = ~np.isnan(trks).any(1)
        self.trackers = [t for t, k in zip(self.trackers, keep) if k]
        trks = trks[keep]
        vel = np.array([t.velocity if t.velocity is not None else np.zeros(2) for t in self.trackers]).reshape(-1, 2)
        last = np.array([t.last_obs for t in self.trackers]).reshape(-1, 5)
        kobs = np.array([k_previous_obs(t.observations, t.age, self.delta_t) for t in self.trackers]).reshape(-1, 5)
        m, ud, ut = self._associate(dets, trks, vel, kobs)
        for d, t in m:
            self.trackers[t].update(dets[d])
        if len(ud) and len(ut):                                  # OCR: second round on last observations
            iou = iou_matrix(dets[ud, :4], last[ut, :4])
            if iou.max() > self.iou_thr:
                rm, _, _ = assign(-iou)
                rd, rt = [], []
                for i, j in rm:
                    if iou[i, j] < self.iou_thr:
                        continue
                    self.trackers[ut[j]].update(dets[ud[i]])
                    rd.append(ud[i])
                    rt.append(ut[j])
                ud, ut = np.setdiff1d(ud, rd), np.setdiff1d(ut, rt)
        for t in ut:
            self.trackers[t].update(None)
        for d in ud:
            self.next_id += 1
            self.trackers.append(_SortTrack(dets[d], self.next_id, oru=True, delta_t=self.delta_t))
        rows = []
        for t in self.trackers:
            box = t.state() if t.last_obs.sum() < 0 else t.last_obs[:4]
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                rows.append([*box, t.id, -1, t.score])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return _out(rows)


# ====================================================================== ByteTrack
TRACKED, LOST, REMOVED = 0, 1, 2


class _STrack:
    def __init__(self, det):
        self.box, self.score = det[:4].astype(np.float64), float(det[4])
        self.mean = self.cov = None
        self.state, self.is_activated = TRACKED, False
        self.id, self.frame_id, self.start_frame, self.tracklet_len = 0, 0, 0, 0

    def xyxy(self):
        return self.box.copy() if self.mean is None else xyah_to_xyxy(self.mean[:4])

    def activate(self, kf, frame_id, tid):
        self.id, self.kf = tid, kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self.box))
        self.tracklet_len, self.state = 0, TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def update(self, det, frame_id, reactivate=False):
        self.frame_id = frame_id
        self.tracklet_len = 0 if reactivate else self.tracklet_len + 1
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.box))
        self.state, self.is_activated, self.score = TRACKED, True, det.score

    def predict(self):
        m = self.mean.copy()
        if self.state != TRACKED:
            m[7] = 0.0
        self.mean, self.cov = self.kf.predict(m, self.cov)


def _iou_dist(a, b):
    return 1.0 - iou_matrix([t.xyxy() for t in a], [t.xyxy() for t in b]) if len(a) and len(b) \
        else np.zeros((len(a), len(b)))


def _fuse_score(cost, dets):
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * np.array([d.score for d in dets])[None, :]


def _joint(a, b):
    ids = {t.id for t in a}
    return a + [t for t in b if t.id not in ids]


def _sub(a, b):
    ids = {t.id for t in b}
    return [t for t in a if t.id not in ids]


class ByteTrack:
    def __init__(self, frame_rate=10, track_thresh=0.5, low_thresh=0.1, match_thresh=0.8, track_buffer=30):
        self.track_thresh, self.low, self.match_thresh = track_thresh, low_thresh, match_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kf = KalmanXYAH()
        self.tracked, self.lost, self.removed, self.frame_id, self.next_id = [], [], [], 0, 0

    def update(self, dets):
        self.frame_id += 1
        dets = dets.reshape(-1, 6)
        s = dets[:, 4]
        high = [_STrack(d) for d in dets[s > self.track_thresh]]
        second = [_STrack(d) for d in dets[(s > self.low) & (s < self.track_thresh)]]
        activated, refind, lost, removed = [], [], [], []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict()
        cost = _fuse_score(_iou_dist(pool, high), high)
        m, u_track, u_det = assign(cost, self.match_thresh)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                t.update(high[j], self.frame_id)
                activated.append(t)
            else:
                t.update(high[j], self.frame_id, reactivate=True)
                refind.append(t)

        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        m, u_track2, _ = assign(_iou_dist(r_tracked, second), 0.5)
        for i, j in m:
            t = r_tracked[i]
            if t.state == TRACKED:
                t.update(second[j], self.frame_id)
                activated.append(t)
            else:
                t.update(second[j], self.frame_id, reactivate=True)
                refind.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)

        left = [high[j] for j in u_det]
        cost = _fuse_score(_iou_dist(unconfirmed, left), left)
        m, u_unconf, u_det = assign(cost, 0.7)
        for i, j in m:
            unconfirmed[i].update(left[j], self.frame_id)
            activated.append(unconfirmed[i])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])

        for j in u_det:
            t = left[j]
            if t.score < self.det_thresh:
                continue
            self.next_id += 1
            t.activate(self.kf, self.frame_id, self.next_id)
            activated.append(t)

        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(_joint(self.tracked, activated), refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, self.removed + removed)
        self.removed.extend(removed)
        self.removed = self.removed[-1000:]                     # bounded memory (ids only matter for _sub)
        self.tracked, self.lost = self._remove_duplicates(self.tracked, self.lost)
        return _out([[*t.xyxy(), t.id, -1, t.score] for t in self.tracked if t.is_activated])

    @staticmethod
    def _remove_duplicates(a, b):
        d = _iou_dist(a, b)
        dup_a, dup_b = set(), set()
        for p, q in zip(*np.where(d < 0.15)):
            ta, tb = a[p].frame_id - a[p].start_frame, b[q].frame_id - b[q].start_frame
            (dup_b.add(q) if ta > tb else dup_a.add(p))
        return [t for i, t in enumerate(a) if i not in dup_a], [t for i, t in enumerate(b) if i not in dup_b]


# ====================================================================== BoT-SORT
class _BTrack(_STrack):
    """BoT-SORT STrack: xywh Kalman state, appearance EMA (alpha 0.9)."""

    def __init__(self, det, feat=None):
        super().__init__(det)
        self.smooth_feat = self.curr_feat = None
        if feat is not None:
            self.update_features(feat)

    def update_features(self, f):
        f = f / (np.linalg.norm(f) + 1e-12)
        self.curr_feat = f
        self.smooth_feat = f if self.smooth_feat is None else 0.9 * self.smooth_feat + 0.1 * f
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-12)

    def xyxy(self):
        return self.box.copy() if self.mean is None else xywh_to_xyxy(self.mean[:4])

    def activate(self, kf, frame_id, tid):
        self.id, self.kf = tid, kf
        self.mean, self.cov = kf.initiate(xyxy_to_xywh(self.box))
        self.tracklet_len, self.state = 0, TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def update(self, det, frame_id, reactivate=False):
        self.frame_id = frame_id
        self.tracklet_len = 0 if reactivate else self.tracklet_len + 1
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xywh(det.box))
        if det.curr_feat is not None:
            self.update_features(det.curr_feat)
        self.state, self.is_activated, self.score = TRACKED, True, det.score

    def predict(self):
        m = self.mean.copy()
        if self.state != TRACKED:
            m[6] = m[7] = 0.0
        self.mean, self.cov = self.kf.predict(m, self.cov)

    def apply_warp(self, H):
        R8 = np.kron(np.eye(4), H[:2, :2])
        self.mean = R8 @ self.mean
        self.mean[:2] += H[:2, 2]
        self.cov = R8 @ self.cov @ R8.T


def _emb_dist(tracks, dets):
    if not len(tracks) or not len(dets):
        return np.zeros((len(tracks), len(dets)))
    a = np.array([t.smooth_feat for t in tracks])
    b = np.array([d.curr_feat for d in dets])
    return np.maximum(0.0, 1.0 - a @ b.T)                      # scipy cdist "cosine" on unit vectors


class BoTSORT(ByteTrack):
    """NirAharon/BoT-SORT tracker/bot_sort.py. Defaults: high 0.6, low 0.1, new track 0.7, buffer 30, match 0.8,
    proximity 0.5, appearance 0.25. CMC: sparseOptFlow (the official default "file" holds precomputed MOT17
    motion; sparseOptFlow is its online equivalent). Output = all tracked tracks, as the official update()."""
    needs_gmc, with_reid = True, False

    def __init__(self, frame_rate=10, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7,
                 track_buffer=30, match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25):
        super().__init__(frame_rate=frame_rate, track_thresh=track_high_thresh, low_thresh=track_low_thresh,
                         match_thresh=match_thresh, track_buffer=track_buffer)
        self.new_track_thresh, self.prox, self.app = new_track_thresh, proximity_thresh, appearance_thresh
        self.kf = KalmanXYWH()
        self.needs_feats = self.with_reid

    def _dist(self, tracks, dets):
        iou_d = _iou_dist(tracks, dets)
        mask = iou_d > self.prox
        iou_d = _fuse_score(iou_d, dets)
        if not self.with_reid:
            return iou_d
        emb = _emb_dist(tracks, dets) / 2.0
        emb[emb > self.app] = 1.0
        emb[mask] = 1.0
        return np.minimum(iou_d, emb)

    def update(self, dets, feats=None, warp=None):
        self.frame_id += 1
        dets = dets.reshape(-1, 6)
        keep = dets[:, 4] > self.low
        dets = dets[keep]
        feats = feats[keep] if (self.with_reid and feats is not None) else None
        hi = dets[:, 4] > self.track_thresh
        high = [_BTrack(d, feats[i] if feats is not None else None) for i, d in zip(np.where(hi)[0], dets[hi])]
        second = [_BTrack(d) for d in dets[(dets[:, 4] < self.track_thresh)]]
        activated, refind, lost, removed = [], [], [], []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict()
        if warp is not None:
            for t in pool + unconfirmed:
                t.apply_warp(warp)
        m, u_track, u_det = assign(self._dist(pool, high), self.match_thresh)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                t.update(high[j], self.frame_id)
                activated.append(t)
            else:
                t.update(high[j], self.frame_id, reactivate=True)
                refind.append(t)

        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        m, u_track2, _ = assign(_iou_dist(r_tracked, second), 0.5)
        for i, j in m:
            t = r_tracked[i]
            t.update(second[j], self.frame_id)
            activated.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)

        left = [high[j] for j in u_det]
        m, u_unconf, u_det = assign(self._dist(unconfirmed, left), 0.7)
        for i, j in m:
            unconfirmed[i].update(left[j], self.frame_id)
            activated.append(unconfirmed[i])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])
        for j in u_det:
            t = left[j]
            if t.score < self.new_track_thresh:
                continue
            self.next_id += 1
            t.activate(self.kf, self.frame_id, self.next_id)
            activated.append(t)

        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(_joint(self.tracked, activated), refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, self.removed + removed)
        self.removed.extend(removed)
        self.removed = self.removed[-1000:]
        self.tracked, self.lost = self._remove_duplicates(self.tracked, self.lost)
        return _out([[*t.xyxy(), t.id, -1, t.score] for t in self.tracked])


class BoTSORTReID(BoTSORT):
    with_reid = True


# ====================================================================== FairMOT association (JDETracker)
class _FTrack(_STrack):
    """FairMOT STrack: xyah Kalman state (as ByteTrack), appearance EMA alpha 0.9 on the network embedding."""
    update_features = _BTrack.update_features

    def __init__(self, det, feat):
        super().__init__(det)
        self.smooth_feat = self.curr_feat = None
        self.update_features(feat)

    def update(self, det, frame_id, reactivate=False):
        super().update(det, frame_id, reactivate)
        self.update_features(det.curr_feat)


class FairMOTTracker(ByteTrack):
    """ifzhang/FairMOT src/lib/tracker/multitracker.py JDETracker.update + matching.py. Defaults: conf_thres 0.4
    (detections below are dropped; new tracks need >= 0.4), track_buffer 30 (-> frame-rate scaled); association
    1: cosine distance of the embeddings (track EMA vs detection) fused with the Kalman gating distance
    (lambda 0.98, chi2 95 % gate), limit 0.4; 2: IoU 0.5 for tracked tracks; unconfirmed: IoU 0.7; output =
    activated tracks. The embeddings are the network's own id head (FairMOT) - any 'feats' work."""
    needs_feats = True

    def __init__(self, frame_rate=10, conf_thres=0.4, track_buffer=30):
        super().__init__(frame_rate=frame_rate, track_thresh=conf_thres, track_buffer=track_buffer)
        self.det_thresh = conf_thres

    def _emb_motion(self, tracks, dets):
        if not len(tracks) or not len(dets):
            return np.zeros((len(tracks), len(dets)))
        cost = _emb_dist(tracks, dets)
        z = np.array([xyxy_to_xyah(d.box) for d in dets])
        for r, t in enumerate(tracks):
            gd = gating_distance(self.kf, t.mean, t.cov, z)
            cost[r, gd > CHI2INV95_4] = np.inf
            cost[r] = 0.98 * cost[r] + 0.02 * gd
        return np.where(np.isfinite(cost), cost, 1e5)

    def update(self, dets, feats=None):
        self.frame_id += 1
        keep = dets[:, 4] > self.track_thresh
        dets, feats = dets[keep], feats[keep]
        detections = [_FTrack(d, f) for d, f in zip(dets, feats)]
        activated, refind, lost, removed = [], [], [], []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict()
        m, u_track, u_det = assign(self._emb_motion(pool, detections), 0.4)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                t.update(detections[j], self.frame_id)
                activated.append(t)
            else:
                t.update(detections[j], self.frame_id, reactivate=True)
                refind.append(t)
        detections = [detections[j] for j in u_det]
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        m, u_track2, u_det = assign(_iou_dist(r_tracked, detections), 0.5)
        for i, j in m:
            r_tracked[i].update(detections[j], self.frame_id)
            activated.append(r_tracked[i])
        for i in u_track2:
            t = r_tracked[i]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)
        detections = [detections[j] for j in u_det]
        m, u_unconf, u_det = assign(_iou_dist(unconfirmed, detections), 0.7)
        for i, j in m:
            unconfirmed[i].update(detections[j], self.frame_id)
            activated.append(unconfirmed[i])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])
        for j in u_det:
            t = detections[j]
            if t.score < self.det_thresh:
                continue
            self.next_id += 1
            t.activate(self.kf, self.frame_id, self.next_id)
            activated.append(t)
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(_joint(self.tracked, activated), refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, self.removed + removed)
        self.removed.extend(removed)
        self.removed = self.removed[-1000:]
        self.tracked, self.lost = self._remove_duplicates(self.tracked, self.lost)
        return _out([[*t.xyxy(), t.id, -1, t.score] for t in self.tracked if t.is_activated])


# ====================================================================== DeepSORT
class _DTrack:
    TENTATIVE, CONFIRMED, DELETED = 1, 2, 3

    def __init__(self, mean, cov, tid, n_init, max_age, feat, score):
        self.mean, self.cov, self.id = mean, cov, tid
        self.hits, self.age, self.time_since_update = 1, 1, 0
        self.state, self.n_init, self.max_age = self.TENTATIVE, n_init, max_age
        self.features, self.score = [feat], score

    def predict(self, kf):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, det, feat):
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det[:4]))
        self.features.append(feat)
        self.hits += 1
        self.time_since_update = 0
        self.score = det[4]
        if self.state == self.TENTATIVE and self.hits >= self.n_init:
            self.state = self.CONFIRMED

    def mark_missed(self):
        if self.state == self.TENTATIVE or self.time_since_update > self.max_age:
            self.state = self.DELETED


class DeepSORT:
    """nwojke/deep_sort (tracker.py, linear_assignment.py, nn_matching.py, deep_sort_app.py). Defaults: detection
    confidence >= 0.8, no NMS (nms_max_overlap 1.0), cosine 0.2, nn_budget None, max_iou_distance 0.7, max_age 30
    (-> frame-rate scaled), n_init 3. Output: confirmed tracks with time_since_update <= 1 (app rule)."""
    needs_feats, INF = True, 1e5

    def __init__(self, frame_rate=10, min_confidence=0.8, max_cosine_distance=0.2, nn_budget=None,
                 max_iou_distance=0.7, max_age=30, n_init=3):
        self.min_conf, self.max_cos, self.budget = min_confidence, max_cosine_distance, nn_budget
        self.max_iou, self.n_init = max_iou_distance, n_init
        self.max_age = max(1, int(round(max_age * frame_rate / 30.0)))
        self.kf, self.tracks, self.samples, self.next_id = KalmanXYAH(), [], {}, 1

    @staticmethod
    def _min_cost(cost, max_d, rows, cols):
        if len(rows) == 0 or len(cols) == 0:
            return [], list(rows), list(cols)
        cost = cost.copy()
        cost[cost > max_d] = max_d + 1e-5
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        matches, um_r, um_c = [], [rows[i] for i in range(len(rows)) if i not in r], \
            [cols[j] for j in range(len(cols)) if j not in c]
        for i, j in zip(r, c):
            if cost[i, j] > max_d:
                um_r.append(rows[i])
                um_c.append(cols[j])
            else:
                matches.append((rows[i], cols[j]))
        return matches, um_r, um_c

    def _app_cost(self, tr, dt, dets, feats):
        cost = np.zeros((len(tr), len(dt)))
        f = feats[dt] / (np.linalg.norm(feats[dt], axis=1, keepdims=True) + 1e-12)
        z = np.array([xyxy_to_xyah(dets[j, :4]) for j in dt])
        for r, ti in enumerate(tr):
            t = self.tracks[ti]
            s = np.asarray(self.samples[t.id])
            s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
            cost[r] = (1.0 - s @ f.T).min(0)
            cost[r, gating_distance(self.kf, t.mean, t.cov, z) > CHI2INV95_4] = self.INF
        return cost

    def _iou_cost(self, tr, dt, dets):
        cost = np.zeros((len(tr), len(dt)))
        for r, ti in enumerate(tr):
            t = self.tracks[ti]
            if t.time_since_update > 1:
                cost[r] = self.INF
                continue
            cost[r] = 1.0 - iou_matrix(xyah_to_xyxy(t.mean[:4])[None], dets[dt, :4])[0]
        return cost

    def update(self, dets, feats=None):
        keep = dets[:, 4] >= self.min_conf
        dets, feats = dets[keep], feats[keep]
        for t in self.tracks:
            t.predict(self.kf)
        confirmed = [i for i, t in enumerate(self.tracks) if t.state == _DTrack.CONFIRMED]
        unconfirmed = [i for i, t in enumerate(self.tracks) if t.state != _DTrack.CONFIRMED]
        um_d, matches_a = list(range(len(dets))), []
        for level in range(self.max_age):                       # matching cascade
            if not um_d:
                break
            tl = [k for k in confirmed if self.tracks[k].time_since_update == 1 + level]
            if not tl:
                continue
            ml, _, um_d = self._min_cost(self._app_cost(tl, um_d, dets, feats), self.max_cos, tl, um_d)
            matches_a += ml
        um_a = list(set(confirmed) - {k for k, _ in matches_a})
        cand = unconfirmed + [k for k in um_a if self.tracks[k].time_since_update == 1]
        um_a = [k for k in um_a if self.tracks[k].time_since_update != 1]
        matches_b, um_b, um_d = self._min_cost(self._iou_cost(cand, um_d, dets), self.max_iou, cand, um_d)
        for ti, di in matches_a + matches_b:
            self.tracks[ti].update(self.kf, dets[di], feats[di])
        for ti in set(um_a + um_b):
            self.tracks[ti].mark_missed()
        for di in um_d:
            mean, cov = self.kf.initiate(xyxy_to_xyah(dets[di, :4]))
            self.tracks.append(_DTrack(mean, cov, self.next_id, self.n_init, self.max_age, feats[di], dets[di, 4]))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.state != _DTrack.DELETED]
        active = [t.id for t in self.tracks if t.state == _DTrack.CONFIRMED]
        for t in self.tracks:                                   # metric.partial_fit
            if t.state != _DTrack.CONFIRMED:
                continue
            self.samples.setdefault(t.id, []).extend(t.features)
            if self.budget is not None:
                self.samples[t.id] = self.samples[t.id][-self.budget:]
            t.features = []
        self.samples = {k: self.samples[k] for k in active}
        return _out([[*xyah_to_xyxy(t.mean[:4]), t.id, -1, t.score] for t in self.tracks
                     if t.state == _DTrack.CONFIRMED and t.time_since_update <= 1])


# ====================================================================== Hybrid-SORT
def _corner(b, k):
    """corners used by HybridSORT speed_direction_{lt, rt, lb, rb}: (x, y) = (b0,b1), (b0,b3), (b2,b1), (b2,b3)."""
    return [(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])][k]


def _dir(b1, b2, k):
    (x1, y1), (x2, y2) = _corner(b1, k), _corner(b2, k)
    v = np.array([y2 - y1, x2 - x1])
    return v / (np.sqrt((v ** 2).sum()) + 1e-6)


class _HTrack:
    def __init__(self, det, tid, delta_t, track_thresh):
        self.kf = KalmanHybrid(det)
        self.id, self.delta_t, self.thr = tid, delta_t, track_thresh
        self.time_since_update, self.hits, self.hit_streak, self.age = 0, 0, 0, 0
        self.last_obs = np.full(5, -1.0)
        self.observations, self.vel = {}, [None] * 4
        self.conf_pre, self.conf = None, float(det[4])

    def predict(self):
        if self.kf.x[7, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[7, 0] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        kscore = float(np.clip(self.kf.x[3, 0], self.thr, 1.0))
        simple = self.conf if not self.conf_pre else self.conf - (self.conf_pre - self.conf)
        return hybrid_x_to_xyxy(self.kf.x), kscore, float(np.clip(simple, 0.1, self.thr))

    def update(self, det):
        if det is None:
            self.kf.update(None, oru=True)
            self.conf_pre = None
            return
        if self.last_obs.sum() >= 0:
            prev, acc = None, [None] * 4
            for i in range(self.delta_t):
                if self.age - i - 1 in self.observations:
                    prev = self.observations[self.age - i - 1]
                    for k in range(4):
                        d = _dir(prev, det, k)
                        acc[k] = d if acc[k] is None else acc[k] + d
            self.vel = [_dir(self.last_obs, det, k) for k in range(4)] if prev is None else acc
        self.last_obs = np.asarray(det[:5], np.float64).copy()
        self.observations[self.age] = self.last_obs
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(hybrid_z(det), oru=True)
        self.conf_pre, self.conf = self.conf, float(det[4])


class HybridSORT:
    """ymzis69/HybridSORT trackers/hybrid_sort_tracker/hybrid_sort.py + association.py, non-ReID variant with the
    MOT17 ablation setting (exps/.../yolox_x_ablation_hybrid_sort.py): track_thresh 0.6, use_byte, inertia 0.05,
    iou_thresh 0.25, Height_Modulated_IoU, TCM in the first and the byte step (weight 1), delta_t 3, min_hits 3,
    max_age 30 (-> frame-rate scaled). First step: HMIoU + sum of the 4 corner direction costs - |track Kalman
    confidence - det score|; byte step: HMIoU - |predicted simple confidence - det score| on 0.1 < score < 0.6;
    OCR on last observations; ORU in the Kalman filter (confidence interpolated too)."""

    def __init__(self, frame_rate=10, det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.25, delta_t=3,
                 inertia=0.05, tcm_first=1.0, tcm_byte=1.0):
        self.max_age = max(1, int(round(max_age * frame_rate / 30.0)))
        self.det_thresh, self.min_hits, self.iou_thr, self.delta_t = det_thresh, min_hits, iou_threshold, delta_t
        self.inertia, self.w1, self.w2 = inertia, tcm_first, tcm_byte
        self.trackers, self.frame_count, self.next_id = [], 0, 0

    def _vel_cost(self, dets, prev_obs, k):
        if len(self.trackers) == 0:
            return np.zeros((len(dets), 0))
        cd = np.array([_corner(d, k) for d in dets]).reshape(-1, 2)
        ct = np.array([_corner(o, k) for o in prev_obs]).reshape(-1, 2)
        dx, dy = cd[None, :, 0] - ct[:, None, 0], cd[None, :, 1] - ct[:, None, 1]
        n = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        X, Y = dx / n, dy / n                                                   # (T, D)
        v = np.array([t.vel[k] if t.vel[k] is not None else np.zeros(2) for t in self.trackers])
        cos = np.clip(v[:, 1:2] * X + v[:, 0:1] * Y, -1, 1)
        diff = (np.pi / 2.0 - np.abs(np.arccos(cos))) / np.pi
        valid = (prev_obs[:, 4] >= 0).astype(float)[:, None]
        return (valid * diff * self.inertia).T * dets[:, 4:5]

    def update(self, dets):
        self.frame_count += 1
        dets = dets.reshape(-1, 6)
        s = dets[:, 4]
        second = dets[(s > 0.1) & (s < self.det_thresh)][:, :5]
        dets = dets[s > self.det_thresh][:, :5]
        trks = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(self.trackers):
            pos, ks, ss = trk.predict()
            trks[t] = [*pos, ks, ss]
        keep = ~np.isnan(trks).any(1)
        self.trackers = [t for t, k in zip(self.trackers, keep) if k]
        trks = trks[keep]
        last = np.array([t.last_obs for t in self.trackers]).reshape(-1, 5)
        kobs = np.array([k_previous_obs(t.observations, t.age, self.delta_t) for t in self.trackers]).reshape(-1, 5)
        # first association (associate_4_points_with_score)
        if len(trks) == 0 or len(dets) == 0:
            m, ud, ut = np.empty((0, 2), int), np.arange(len(dets)), np.arange(len(trks))
        else:
            ang = sum(self._vel_cost(dets, kobs, k) for k in range(4))
            ang = ang - np.abs(trks[None, :, 4] - dets[:, None, 4]) * self.w1
            iou = hmiou(dets, trks)
            a = (iou > self.iou_thr).astype(int)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                m = np.stack(np.where(a), 1)
            else:
                m, _, _ = assign(-(iou + ang))
            good = iou[m[:, 0], m[:, 1]] >= self.iou_thr if len(m) else np.zeros(0, bool)
            m = m[good]
            ud = np.setdiff1d(np.arange(len(dets)), m[:, 0])
            ut = np.setdiff1d(np.arange(len(trks)), m[:, 1])
        for d, t in m:
            self.trackers[t].update(dets[d])
        # byte step with TCM
        if len(second) and len(ut):
            iou_left = hmiou(second, trks[ut])
            if iou_left.max() > self.iou_thr:
                iou_left = iou_left - np.abs(trks[ut][None, :, 5] - second[:, None, 4]) * self.w2
                mm, _, _ = assign(-iou_left)
                rm = []
                for i, j in mm:
                    if iou_left[i, j] < self.iou_thr:
                        continue
                    self.trackers[ut[j]].update(second[i])
                    rm.append(ut[j])
                ut = np.setdiff1d(ut, rm)
        # OCR on last observations
        if len(ud) and len(ut):
            iou_left = hmiou(dets[ud], last[ut])
            if iou_left.max() > self.iou_thr:
                mm, _, _ = assign(-iou_left)
                rd, rt = [], []
                for i, j in mm:
                    if iou_left[i, j] < self.iou_thr:
                        continue
                    self.trackers[ut[j]].update(dets[ud[i]])
                    rd.append(ud[i])
                    rt.append(ut[j])
                ud, ut = np.setdiff1d(ud, rd), np.setdiff1d(ut, rt)
        for t in ut:
            self.trackers[t].update(None)
        for d in ud:
            self.next_id += 1
            self.trackers.append(_HTrack(dets[d], self.next_id, self.delta_t, self.det_thresh))
        rows = []
        for t in self.trackers:
            box = hybrid_x_to_xyxy(t.kf.x) if t.last_obs.sum() < 0 else t.last_obs[:4]
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                rows.append([*box, t.id, -1, t.conf])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return _out(rows)


# ====================================================================== TrackTrack
NEW, TT_TRACKED, TT_LOST, TT_REMOVED = 0, 1, 2, 3


class _TTrack:
    def __init__(self, det, feat):
        self.box, self.score = det[:4].astype(np.float64), float(det[4])
        self.feat = (feat / (np.linalg.norm(feat) + 1e-12))[None].astype(np.float64)
        self.history, self.mean, self.cov = {}, None, None
        self.velocity = np.zeros((4, 2))
        self.state, self.id, self.end_frame = NEW, 0, 0

    @property
    def cxcywh(self):
        if self.mean is None:
            b = self.box
            return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])
        return self.mean[:4].copy()

    @property
    def xyxy(self):
        c = self.cxcywh
        return np.array([c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2])

    def initiate(self, frame_id, tid, kf):
        self.id, self.kf = tid, kf
        self.mean, self.cov = kf.initiate(self.cxcywh)
        self.history[frame_id] = [self.box.copy(), self.score]
        self.end_frame, self.state = frame_id, NEW

    def update(self, frame_id, det, min_len):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, det.cxcywh, det.score)
        beta = 0.95 + 0.05 * (1 - det.score)
        self.feat = beta * self.feat + (1 - beta) * det.feat
        self.feat /= np.linalg.norm(self.feat)
        self.history[frame_id] = [det.box.copy(), det.score]
        self.velocity = np.zeros((4, 2))
        for dt in range(1, 4):
            self.velocity += _tt_vel(_prev_box(self.history, frame_id, dt), det.xyxy) / dt
        self.velocity /= 3
        self.box, self.score, self.end_frame = det.box.copy(), det.score, frame_id
        self.state = TT_TRACKED if len(self.history) >= min_len else NEW


def _prev_box(history, frame_id, dt):
    k = frame_id - dt
    return history[k][0] if k in history else history[max(history)][0]


def _tt_vel(b1, b2):
    d = b2 - b1
    nlt = np.sqrt(d[0] ** 2 + d[1] ** 2) + 1e-5
    nlb = np.sqrt(d[0] ** 2 + d[3] ** 2) + 1e-5
    nrt = np.sqrt(d[2] ** 2 + d[1] ** 2) + 1e-5
    nrb = np.sqrt(d[2] ** 2 + d[3] ** 2) + 1e-5
    return np.stack([np.array([d[0], d[1]]) / nlt, np.array([d[0], d[3]]) / nlb,
                     np.array([d[2], d[1]]) / nrt, np.array([d[2], d[3]]) / nrb])


def _tt_iou_dist(a, b):
    A = np.array([t.xyxy for t in a]).reshape(-1, 4)
    B = np.array([t.xyxy for t in b]).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        z = np.zeros((len(A), len(B)))
        return z, 1 - z
    h = (np.minimum(A[:, 3:4], B[:, 3:4].T) - np.maximum(A[:, 1:2], B[:, 1:2].T)) / \
        (np.maximum(A[:, 3:4], B[:, 3:4].T) - np.minimum(A[:, 1:2], B[:, 1:2].T))
    sim = iou_plus1(A, B)
    return sim, 1 - h * sim


class TrackTrack:
    """kamkyu94/TrackTrack 3. Tracker/trackers (tracker.py, track.py, utils.py, kalman_filter.py), online part only
    (no AFLink / interpolation). MOT17 default thresholds (utils/etc.py, sequences without their own values):
    det_thr 0.60, init_thr 0.70, match_thr 0.70; penalty_p 0.20 (low-score detections), penalty_q 0.40,
    reduce_step 0.05, tai_thr 0.55 (track-aware initialisation), min_len 3, max_time_lost = 2 s (frame rate x 2).
    TPA: cost = 0.5 HM-IoU distance + 0.5 cosine + 0.1 confidence + 0.05 angle (4 corners), iterated mutual-min
    matching with a decreasing threshold. [Unified]: detections removed only by NMS 0.95 vs 0.8 (dets_95) are not
    available -> empty; camera motion = online sparseOptFlow (official: pre-computed GMC files); detections below
    0.1 are not used (no ReID feature there; the official detections start at that confidence)."""
    needs_feats = needs_gmc = True

    def __init__(self, frame_rate=10, det_thr=0.60, init_thr=0.70, match_thr=0.70, penalty_p=0.20, penalty_q=0.40,
                 reduce_step=0.05, tai_thr=0.55, min_len=3):
        self.det_thr, self.init_thr, self.match_thr = det_thr, init_thr, match_thr
        self.pp, self.pq, self.step_, self.tai, self.min_len = penalty_p, penalty_q, reduce_step, tai_thr, min_len
        self.max_time_lost = int(frame_rate * 2)
        self.tracks, self.frame_id, self.next_id, self.kf = [], 0, 0, KalmanTT()

    def _cost(self, tracks, high, low):
        dets = high + low
        sim, dist = _tt_iou_dist(tracks, dets)
        if len(tracks) == 0 or len(dets) == 0:
            return np.ones((len(tracks), len(dets))), sim
        tf = np.concatenate([t.feat for t in tracks])
        df = np.concatenate([d.feat for d in dets])
        cos = np.clip(1 - tf @ df.T, 0.0, 1.0)
        prev = []
        for t in tracks:
            ks = sorted(t.history, reverse=True)
            prev.append(t.history[ks[min(1, len(ks) - 1)]][1])
        ts = np.array([t.score for t in tracks])
        ts = ts + (ts - np.array(prev))
        conf = np.abs(ts[:, None] - np.array([d.score for d in dets])[None])
        pb = np.stack([_prev_box(t.history, self.frame_id, 3) for t in tracks])
        db = np.stack([d.xyxy for d in dets])
        d = db[None] - pb[:, None]                                              # utils.get_vel_t_d, vectorised
        corners = [(0, 1), (0, 3), (2, 1), (2, 3)]                             # lt, lb, rt, rb
        vt = np.stack([t.velocity for t in tracks])                            # (T, 4, 2)
        ang = np.zeros((len(tracks), len(dets)))
        for k, (ix, iy) in enumerate(corners):
            n = np.sqrt(d[..., ix] ** 2 + d[..., iy] ** 2) + 1e-5
            c = np.clip(vt[:, None, k, 0] * d[..., ix] / n + vt[:, None, k, 1] * d[..., iy] / n, -1, 1)
            ang += np.abs(np.arccos(c)) / np.pi / 4
        ang *= np.array([d_.score for d_ in dets])[None]
        cost = 0.5 * dist + 0.5 * cos + 0.1 * conf + 0.05 * ang
        cost[:, len(high):] += self.pp
        cost[sim <= 0.10] = 1.0
        return np.clip(cost, 0, 1), sim

    def _iterative(self, tracks, high, low):
        cost, _ = self._cost(tracks, high, low)
        thr, matches = self.match_thr, []
        while True:
            mm = []
            if cost.shape[0] and cost.shape[1]:
                md, mt = cost.argmin(1), cost.argmin(0)
                mm = [[t, d] for t, d in enumerate(md) if mt[d] == t and cost[t, d] < thr]
            thr -= self.step_
            if not mm:
                break
            matches += mm
            for t, d in matches:
                cost[t, :] = 1.0
                cost[:, d] = 1.0
        mt_, md_ = {t for t, _ in matches}, {d for _, d in matches}
        return matches, [t for t in range(len(tracks)) if t not in mt_], [d for d in range(len(high) + len(low)) if d not in md_]

    def _warp(self, tracks, H):
        R8 = np.kron(np.eye(4), H[:, :2])
        for t in tracks:
            t.mean = R8 @ t.mean
            t.mean[:2] += H[:, 2]
            t.cov = R8 @ t.cov @ R8.T

    def update(self, dets, feats=None, warp=None):
        self.frame_id += 1
        keep = dets[:, 4] >= 0.1
        objs = [_TTrack(d, f) for d, f in zip(dets[keep], feats[keep])]
        high = [d for d in objs if d.score > self.det_thr]
        low = [d for d in objs if d.score <= self.det_thr]
        tl = [t for t in self.tracks if t.state in (TT_TRACKED, TT_LOST)]
        new = [t for t in self.tracks if t.state == NEW]
        H = warp if warp is not None else np.eye(2, 3)
        self._warp(tl, H)
        self._warp(new, H)
        for t in tl + new:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        cand = high + low
        m, ut, ud = self._iterative(tl, high, low)
        for t, d in m:
            tl[t].update(self.frame_id, cand[d], self.min_len)
        for t in ut:
            tl[t].state = TT_LOST
        left = [cand[i] for i in ud if i < len(high)]
        m, ut, ud = self._iterative(new, left, [])
        for t, d in m:
            new[t].update(self.frame_id, left[d], self.min_len)
        for t in ut:
            new[t].state = TT_REMOVED
        for t in self.tracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.state = TT_REMOVED
        self.tracks = [t for t in self.tracks if t.state != TT_REMOVED]
        self._init_tracks([left[i] for i in ud])
        return _out([[*t.xyxy, t.id, -1, t.score] for t in self.tracks if t.state == TT_TRACKED])

    def _init_tracks(self, dets):
        tr = [t for t in self.tracks if t.state in (TT_TRACKED, NEW)]
        allb = tr + dets
        sim = _tt_iou_dist(allb, allb)[0]
        nt, scores = len(tr), np.array([d.score for d in dets])
        allow = (scores > self.init_thr).astype(int) if len(dets) else np.zeros(0, int)
        for i in range(len(dets)):
            if not allow[i]:
                continue
            if nt and sim[nt + i, :nt].max() > self.tai:
                allow[i] = 0
                continue
            for j in range(len(dets)):
                if i != j and allow[j] and scores[i] > scores[j] and sim[nt + i, nt + j] > self.tai:
                    allow[j] = 0
        for i, d in enumerate(dets):
            if allow[i]:
                self.next_id += 1
                d.initiate(self.frame_id, self.next_id, self.kf)
                self.tracks.append(d)


# ====================================================================== per-class wrapper and registry
class PerClass:
    """[Unified] one tracker instance per class; output ids = class * 100000 + id (unique across classes).
    feats (appearance, rows aligned with dets) and warp (camera motion of this frame) are passed through."""

    def __init__(self, factory):
        self.factory, self.inst = factory, {}
        probe = factory()
        self.needs_feats, self.needs_gmc = getattr(probe, "needs_feats", False), getattr(probe, "needs_gmc", False)

    def update(self, dets, feats=None, warp=None):
        dets = np.asarray(dets, np.float64).reshape(-1, 6)
        classes = set(np.unique(dets[:, 5]).astype(int)) | set(self.inst)
        outs = []
        for c in sorted(classes):
            if c not in self.inst:
                self.inst[c] = self.factory()
            m = dets[:, 5] == c
            kw = {}
            if self.needs_feats:
                kw["feats"] = feats[m]
            if self.needs_gmc:
                kw["warp"] = warp
            o = self.inst[c].update(dets[m], **kw)
            if len(o):
                o = o.copy()
                o[:, 4] += c * 100000
                o[:, 5] = c
                outs.append(o)
        return np.concatenate(outs) if outs else EMPTY.copy()


TRACKERS = {"iou": IoUTracker, "sort": SORT, "bytetrack": ByteTrack, "ocsort": OCSORT, "deepsort": DeepSORT,
            "botsort": BoTSORT, "botsort_reid": BoTSORTReID, "fairmot": FairMOTTracker,
            "hybridsort": HybridSORT, "tracktrack": TrackTrack}


# main detection-score threshold of each tracker (sensitivity analysis: run_tracker.py --score-thr)
SCORE_KW = {"iou": "score_thr", "sort": "score_thr", "bytetrack": "track_thresh", "ocsort": "det_thresh",
            "deepsort": "min_confidence", "botsort": "track_high_thresh", "botsort_reid": "track_high_thresh",
            "fairmot": "conf_thres", "hybridsort": "det_thresh", "tracktrack": "det_thr"}


def make_tracker(name, frame_rate=10, score_thr=None, **kw):
    """score_thr: overrides the tracker's main score threshold (official default otherwise). Thresholds tied to it
    keep their official offset: ByteTrack new track = thr + 0.1 (built in), BoT-SORT new track = thr + 0.1."""
    if name not in TRACKERS:
        raise ValueError(f"unknown tracker {name}; available: {sorted(TRACKERS)}")
    if score_thr is not None:
        kw[SCORE_KW[name]] = score_thr
        if name.startswith("botsort"):
            kw["new_track_thresh"] = score_thr + 0.1
    return PerClass(lambda: TRACKERS[name](frame_rate=frame_rate, **kw))
