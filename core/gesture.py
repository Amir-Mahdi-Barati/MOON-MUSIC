"""
MOON MUSIC — core/gesture.py  v5
Advanced CV Gesture Engine
  · 11 gestures (angle-based finger analysis + temporal voting)
  · Face detection (ADMIN box + keypoints + analytics)
  · Reliable swipe (velocity + time window + direction consistency)
  · Smoothed pinch-volume control
Author: Amir Mahdi Barati | github.com/Amir-Mahdi-Barati
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Deque

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    CV_OK = True
except ImportError:
    CV_OK = False

# ── Config ───────────────────────────────────────────────────────────────────
COOLDOWNS: dict = {
    "palm": 1.40, "swipe_right": 0.90, "swipe_left": 0.90,
    "fist": 1.50, "peace": 1.10, "thumb_up": 1.10,
    "thumb_down": 1.10, "pinch": 0.04, "point_up": 1.10,
    "call_me": 1.20, "rock": 1.20,
}
PALM_HOLD_FRAMES = 18
VOTE_WINDOW = 9
MIN_CONFIDENCE = 0.68
SWIPE_FRAMES = 22
SWIPE_MIN_DIST = 0.15
SWIPE_MAX_TIME = 0.55
SWIPE_CONSIST = 0.60
PINCH_ALPHA = 0.18


class COL:
    ACCENT = (110, 255, 165)
    MOON = (210, 155, 255)
    FACE = (45, 200, 255)
    KP = (30, 165, 235)
    WHITE = (240, 240, 252)
    DIM = (90, 90, 115)
    BLACK = (6, 6, 14)
    PANEL = (14, 14, 28)
    RED = (55, 55, 215)
    ORANGE = (40, 165, 255)
    GREEN = (75, 220, 100)
    ROCK = (180, 75, 255)


@dataclass
class FingerData:
    name: str
    extended: bool
    curl: float
    angle: float


@dataclass
class HandSnapshot:
    fingers: List[FingerData]
    gesture: str
    confidence: float
    pinch_raw: float
    pinch_smooth: float
    pos_x: float
    pos_y: float
    depth_z: float
    tilt_deg: float
    palm_facing: str
    bbox: Tuple[int, int, int, int]


@dataclass
class FaceSnapshot:
    bbox: Tuple[int, int, int, int]
    conf: float
    keypoints: List[Tuple[int, int]]
    cx: int
    cy: int
    fw: int
    fh: int


@dataclass
class GestureEvent:
    gesture: str
    label: str
    ts: float = field(default_factory=time.time)


class FingerAnalyzer:
    TIPS = [4, 8, 12, 16, 20]
    DIPS = [3, 7, 11, 15, 19]
    PIPS = [2, 6, 10, 14, 18]
    MCPS = [1, 5, 9, 13, 17]
    NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def analyze_all(self, lm, img_w: int, img_h: int) -> List[FingerData]:
        return [self._analyze_finger(lm, i, img_w, img_h) for i in range(5)]

    def _analyze_finger(self, lm, fi: int, w: int, h: int) -> FingerData:
        tip, dip, pip_, mcp = lm[self.TIPS[fi]], lm[self.DIPS[fi]], lm[self.PIPS[fi]], lm[self.MCPS[fi]]

        def p(l):
            return np.array([l.x * w, l.y * h], dtype=np.float32)

        v1 = p(pip_) - p(mcp)
        v2 = p(tip) - p(dip)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0.5 and n2 > 0.5:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        else:
            angle = 180.0
        curl = max(0.0, min(1.0, 1.0 - angle / 178.0))

        if fi == 0:
            extended = tip.x < lm[3].x if lm[17].x > lm[5].x else tip.x > lm[3].x
        else:
            extended = (tip.y < pip_.y) and (curl < 0.38)

        return FingerData(self.NAMES[fi], bool(extended), round(curl, 3), round(angle, 1))

    @staticmethod
    def pinch_distance(lm) -> float:
        return float(math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))

    @staticmethod
    def wrist_tilt(lm) -> float:
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8)))

    @staticmethod
    def palm_facing(lm) -> str:
        w = np.array([lm[0].x, lm[0].y, lm[0].z])
        im = np.array([lm[5].x, lm[5].y, lm[5].z])
        pm = np.array([lm[17].x, lm[17].y, lm[17].z])
        normal = np.cross(im - w, pm - w)
        return "camera" if float(normal[2]) < 0 else "away"

    @staticmethod
    def landmark_bbox(lm, w: int, h: int, pad: int = 24) -> Tuple[int, int, int, int]:
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                min(w - 1, int(max(xs)) + pad), min(h - 1, int(max(ys)) + pad))


class GestureClassifier:
    def classify(self, fingers: List[FingerData], pinch_d: float, palm: str, lm) -> Tuple[str, float]:
        ext = [f.extended for f in fingers]
        curls = [f.curl for f in fingers]
        cnt = sum(ext)
        mask = sum((1 << i) for i, e in enumerate(ext) if e)

        if pinch_d < 0.065 and ext[0] and ext[1] and not ext[2] and not ext[3]:
            return "pinch", max(0.70, min(0.99, 1.0 - pinch_d * 10.0))

        if cnt == 0 and all(c > 0.48 for c in curls):
            return "fist", 0.97

        if ext[1] and ext[4] and not ext[2] and not ext[3] and curls[2] > 0.42 and curls[3] > 0.42:
            return "rock", 0.95

        if ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "call_me", 0.93

        if mask == 1:
            if lm[4].y < lm[3].y < lm[2].y:
                return "thumb_up", 0.94
            if lm[4].y > lm[3].y > lm[2].y:
                return "thumb_down", 0.90

        if mask == 2:
            if lm[8].y < lm[5].y:
                return "point_up", 0.93
            return "point_forward", 0.78

        if mask in (6, 7):
            spread = math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
            return "peace", min(0.97, 0.78 + spread * 3.5)

        if cnt == 5 or mask in (30, 31):
            spread = self._fingertip_spread(lm)
            return "open_palm", min(0.98, 0.75 + spread * 2.0)

        if cnt > 0:
            return f"fingers_{cnt}", 0.55
        return "unknown", 0.25

    @staticmethod
    def _fingertip_spread(lm) -> float:
        tips = [8, 12, 16, 20]
        dists = [math.hypot(lm[tips[i]].x - lm[tips[i+1]].x, lm[tips[i]].y - lm[tips[i+1]].y)
                 for i in range(len(tips) - 1)]
        return sum(dists) / len(dists) if dists else 0.0


class TemporalVoter:
    def __init__(self, window: int = VOTE_WINDOW):
        self._g: Deque[str] = deque(maxlen=window)
        self._c: Deque[float] = deque(maxlen=window)

    def push(self, gesture: str, conf: float) -> Tuple[str, float]:
        self._g.append(gesture)
        self._c.append(conf)
        if len(self._g) < 3:
            return gesture, conf
        counts = Counter(self._g)
        winner, win_count = counts.most_common(1)[0]
        win_confs = [c for g, c in zip(self._g, self._c) if g == winner]
        avg_conf = sum(win_confs) / len(win_confs) if win_confs else conf
        stability = win_count / len(self._g)
        return winner, min(0.99, avg_conf * (0.85 + 0.15 * stability))

    def reset(self) -> None:
        self._g.clear()
        self._c.clear()


class SwipeDetector:
    def __init__(self):
        self._xs: Deque[float] = deque(maxlen=SWIPE_FRAMES)
        self._ts: Deque[float] = deque(maxlen=SWIPE_FRAMES)

    def push(self, x: float) -> Optional[str]:
        now = time.time()
        self._xs.append(x)
        self._ts.append(now)
        if len(self._xs) < SWIPE_FRAMES // 2:
            return None
        dt = max(0.001, self._ts[-1] - self._ts[0])
        if dt > SWIPE_MAX_TIME:
            self._xs.popleft()
            self._ts.popleft()
            return None
        delta = self._xs[-1] - self._xs[0]
        if abs(delta) < SWIPE_MIN_DIST:
            return None
        diffs = [self._xs[i] - self._xs[i - 1] for i in range(1, len(self._xs))]
        ratio = (sum(d > 0 for d in diffs) if delta > 0 else sum(d < 0 for d in diffs)) / len(diffs)
        if ratio >= SWIPE_CONSIST:
            self._xs.clear()
            self._ts.clear()
            return "swipe_right" if delta > 0 else "swipe_left"
        return None

    def reset(self) -> None:
        self._xs.clear()
        self._ts.clear()


class PinchSmoother:
    def __init__(self, alpha: float = PINCH_ALPHA):
        self._alpha = alpha
        self._val = 0.15

    def update(self, raw: float) -> float:
        self._val = self._alpha * raw + (1.0 - self._alpha) * self._val
        return self._val

    def reset(self) -> None:
        self._val = 0.15


class CVRenderer:
    @staticmethod
    def overlay_rect(frame, x1, y1, x2, y2, color=None, alpha: float = 0.78) -> None:
        if color is None:
            color = COL.PANEL
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)

    @staticmethod
    def corner_frame(frame, x1, y1, x2, y2, color=COL.ACCENT, thickness: int = 2, corner_len: int = 22) -> None:
        for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (px, py), (px + dx * corner_len, py), color, thickness)
            cv2.line(frame, (px, py), (px, py + dy * corner_len), color, thickness)

    @staticmethod
    def progress_bar(frame, x, y, w, h, value: float, bg=COL.PANEL, fg=COL.ACCENT) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        fill = int(w * max(0.0, min(1.0, value)))
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + h), fg, -1)

    @staticmethod
    def label(frame, text, x, y, scale: float = 0.36, color=COL.WHITE, thickness: int = 1) -> None:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_face(self, frame, fd: FaceSnapshot, index: int, frame_num: int) -> None:
        x1, y1, x2, y2 = fd.bbox
        h_, w_ = frame.shape[:2]
        self.corner_frame(frame, x1, y1, x2, y2, COL.FACE, thickness=3, corner_len=28)

        label_text = f" ADMIN #{index+1}   {fd.conf*100:.0f}% "
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1-lh-16), (x1+lw+12, y1), COL.FACE, -1)
        cv2.putText(frame, label_text, (x1+6, y1-6), cv2.FONT_HERSHEY_DUPLEX, 0.52, COL.BLACK, 1, cv2.LINE_AA)

        KP_NAMES = ["R-EYE", "L-EYE", "NOSE", "MOUTH", "R-EAR", "L-EAR"]
        pts = fd.keypoints[:6]
        if len(pts) >= 2:
            cv2.line(frame, pts[0], pts[1], COL.DIM, 1, cv2.LINE_AA)
        if len(pts) >= 4:
            cv2.line(frame, pts[2], pts[3], COL.DIM, 1, cv2.LINE_AA)
        for i, (kx, ky) in enumerate(pts):
            cv2.circle(frame, (kx, ky), 5, COL.FACE, -1)
            cv2.circle(frame, (kx, ky), 6, COL.WHITE, 1)
            if i < len(KP_NAMES):
                self.label(frame, KP_NAMES[i], kx+7, ky-5, 0.26, COL.KP)

        px = min(x2 + 10, w_ - 135)
        if px + 132 < w_:
            eye_dist = int(math.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1])) if len(pts) >= 2 else 0
            rows = [
                ("FACE DATA", "", COL.FACE), ("CONF", f"{fd.conf*100:.1f}%", COL.WHITE),
                ("SIZE", f"{fd.fw}x{fd.fh}px", COL.DIM), ("CENTER", f"{fd.cx},{fd.cy}", COL.DIM),
                ("EYE DIST", f"{eye_dist}px", COL.KP),
            ]
            ph = len(rows) * 17 + 14
            self.overlay_rect(frame, px-4, y1, px+132, y1+ph)
            cv2.rectangle(frame, (px-4, y1), (px+132, y1+ph), COL.FACE, 1)
            for j, (k, v, c) in enumerate(rows):
                yy = y1 + 14 + j * 17
                if v == "":
                    self.label(frame, k, px+4, yy, 0.36, c)
                else:
                    self.label(frame, f"{k}:", px+4, yy, 0.28, COL.DIM)
                    self.label(frame, v, px+60, yy, 0.34, c)

    def draw_hand(self, frame, snap: HandSnapshot, palm_hold: int) -> None:
        x1, y1, x2, y2 = snap.bbox
        h_, w_ = frame.shape[:2]
        gesture_colors = {
            "swipe_right": COL.GREEN, "swipe_left": COL.GREEN, "fist": COL.RED,
            "open_palm": COL.ACCENT, "thumb_up": COL.GREEN, "thumb_down": COL.ORANGE,
            "peace": COL.KP, "pinch": COL.MOON, "rock": COL.ROCK, "point_up": COL.ACCENT,
        }
        box_color = gesture_colors.get(snap.gesture, COL.DIM)
        self.corner_frame(frame, x1, y1, x2, y2, box_color, thickness=2, corner_len=20)
        self.progress_bar(frame, x1, y2+6, x2-x1, 5, snap.confidence, COL.PANEL, box_color)

        pill_x = min(x2 + 10, w_ - 62)
        abbrs = ["THB", "IDX", "MID", "RNG", "PNK"]
        for fi, fs in enumerate(snap.fingers):
            fy = y1 + fi * 31 + 6
            col = box_color if fs.extended else COL.DIM
            cv2.rectangle(frame, (pill_x, fy-12), (pill_x+58, fy+10), COL.PANEL, -1)
            cv2.rectangle(frame, (pill_x, fy-12), (pill_x+58, fy+10), col, 1)
            fill_w = int(56 * (1.0 - fs.curl))
            if fill_w > 0 and fs.extended:
                cv2.rectangle(frame, (pill_x+1, fy-11), (pill_x+1+fill_w, fy+9), col, -1)
            cv2.putText(frame, abbrs[fi], (pill_x+3, fy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        COL.BLACK if fs.extended else COL.DIM, 1, cv2.LINE_AA)
            self.label(frame, f"{int(fs.curl*100)}%", pill_x+34, fy+4, 0.27, COL.DIM)

        dp_x = max(2, x1 - 136)
        conf_c = COL.GREEN if snap.confidence >= 0.88 else (COL.ACCENT if snap.confidence >= 0.70 else COL.ORANGE)
        g_disp = snap.gesture.replace("_", " ").upper()
        ext_icons = "".join("1" if f.extended else "0" for f in snap.fingers)
        rows = [
            ("HAND", "", COL.ACCENT), ("GESTURE", g_disp, COL.WHITE),
            ("CONF", f"{snap.confidence*100:.0f}%", conf_c), ("PINCH", f"{snap.pinch_raw*100:.1f}", COL.DIM),
            ("POS X", f"{snap.pos_x:.3f}", COL.DIM), ("POS Y", f"{snap.pos_y:.3f}", COL.DIM),
            ("TILT", f"{snap.tilt_deg:.0f}deg", COL.DIM), ("PALM", snap.palm_facing.upper(), COL.KP),
            ("EXT", ext_icons, box_color),
        ]
        ph = len(rows) * 17 + 14
        self.overlay_rect(frame, dp_x, y1, dp_x+134, y1+ph)
        cv2.rectangle(frame, (dp_x, y1), (dp_x+134, y1+ph), COL.ACCENT, 1)
        for j, (k, v, c) in enumerate(rows):
            yy = y1 + 14 + j * 17
            if v == "":
                self.label(frame, k, dp_x+4, yy, 0.36, c)
            else:
                self.label(frame, f"{k}:", dp_x+4, yy, 0.28, COL.DIM)
                self.label(frame, v, dp_x+55, yy, 0.34, c)

        if snap.gesture not in ("unknown", "open_palm") and snap.confidence >= MIN_CONFIDENCE:
            ICONS = {
                "swipe_right": "NEXT", "swipe_left": "PREV", "fist": "MUTE",
                "thumb_up": "LIKE", "thumb_down": "SHUFFLE", "peace": "SHUFFLE",
                "point_up": "REPEAT", "pinch": "VOLUME", "rock": "ROCK MODE", "call_me": "MODE",
            }
            badge = ICONS.get(snap.gesture, snap.gesture.replace("_", " ").upper())
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.88, 2)
            bx, by = (w_ - bw) // 2, h_ // 2
            self.overlay_rect(frame, bx-20, by-36, bx+bw+20, by+16, COL.BLACK, 0.74)
            cv2.rectangle(frame, (bx-20, by-36), (bx+bw+20, by+16), box_color, 1)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 0.88, box_color, 2, cv2.LINE_AA)

        if snap.gesture == "open_palm":
            prog = min(1.0, palm_hold / PALM_HOLD_FRAMES)
            cx_arc, cy_arc = w_ - 68, 68
            cv2.circle(frame, (cx_arc, cy_arc), 50, COL.PANEL, -1)
            cv2.circle(frame, (cx_arc, cy_arc), 50, COL.DIM, 1)
            arc_c = COL.GREEN if prog > 0.80 else COL.ACCENT
            cv2.ellipse(frame, (cx_arc, cy_arc), (50, 50), -90, 0, int(360*prog), arc_c, 6, cv2.LINE_AA)
            pct = f"{int(prog*100)}%"
            (pw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
            cv2.putText(frame, pct, (cx_arc-pw//2, cy_arc+6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, arc_c, 1, cv2.LINE_AA)
            self.label(frame, "HOLD", cx_arc-16, cy_arc+24, 0.30, COL.DIM)

        if snap.gesture == "pinch":
            vol_pct = max(0.0, min(1.0, 1.0 - snap.pinch_smooth * 7.0))
            self.progress_bar(frame, w_-192, 12, 178, 12, vol_pct, COL.PANEL, COL.MOON)
            cv2.putText(frame, f"VOLUME {int(vol_pct*100)}%", (w_-192, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL.MOON, 1, cv2.LINE_AA)

        if snap.gesture == "rock":
            cx_r, cy_r = (x1+x2)//2, (y1+y2)//2
            t_ = time.time()
            for ri in range(4):
                r = int(22 + ri*16 + 6*math.sin(t_*4 + ri))
                cv2.circle(frame, (cx_r, cy_r), r, COL.ROCK, 1, cv2.LINE_AA)

    def draw_hud(self, frame, title: str, artist: str, playing: bool, volume: float,
                 hand_vis: bool, face_vis: bool, face_count: int, fps: int, g_history: List[str]) -> None:
        h, w = frame.shape[:2]
        self.overlay_rect(frame, 0, h-92, w, h, COL.BLACK, 0.84)
        cv2.line(frame, (0, h-92), (w, h-92), COL.ACCENT, 1)
        icon = ">" if playing else "||"
        cv2.putText(frame, f"{icon}  {title}  -  {artist}", (18, h-58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, COL.WHITE, 2, cv2.LINE_AA)
        self.label(frame, f"VOL {int(volume*100)}%", w-224, h-36, 0.36, COL.DIM)
        self.progress_bar(frame, w-224, h-26, 188, 7, volume, COL.PANEL, COL.ACCENT)
        fps_c = COL.GREEN if fps >= 24 else (COL.ORANGE if fps >= 14 else COL.RED)
        self.label(frame, f"FPS {fps}", w-78, h-6, 0.38, fps_c)
        if g_history:
            self.label(frame, " > ".join(g_history[-5:]), 18, h-10, 0.28, COL.DIM)

        legs = [
            ("HOLD PALM", "Play/Pause"), ("=> SWIPE R", "Next"), ("<= SWIPE L", "Prev"),
            ("FIST", "Mute"), ("THUMB UP", "Like"), ("THUMB DOWN", "Shuffle"),
            ("PEACE", "Shuffle"), ("POINT UP", "Repeat"), ("PINCH", "Volume"),
            ("ROCK", "Rock Mode"), ("CALL ME", "Mode"),
        ]
        pw, ph = 254, len(legs)*21+32
        self.overlay_rect(frame, 0, 0, pw, ph, COL.BLACK, 0.80)
        cv2.rectangle(frame, (0, 0), (pw, ph), COL.ACCENT, 1)
        cv2.putText(frame, "GESTURE MAP", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL.ACCENT, 1, cv2.LINE_AA)
        for i, (g, a) in enumerate(legs):
            yy = 32 + i*21
            self.label(frame, g, 10, yy, 0.33, COL.ACCENT)
            self.label(frame, f"-> {a}", 148, yy, 0.31, COL.DIM)

        hc = COL.ACCENT if hand_vis else COL.DIM
        cv2.circle(frame, (w-22, 22), 11, hc, -1)
        cv2.circle(frame, (w-22, 22), 11, COL.WHITE, 1)
        cv2.putText(frame, "H", (w-27, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL.BLACK, 1)
        fc_c = COL.FACE if face_vis else COL.DIM
        cv2.circle(frame, (w-54, 22), 11, fc_c, -1)
        cv2.circle(frame, (w-54, 22), 11, COL.WHITE, 1)
        cv2.putText(frame, "F", (w-59, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL.BLACK, 1)
        if face_count > 0:
            self.label(frame, f"{face_count} FACE{'S' if face_count > 1 else ''}", w-104, 48, 0.34, COL.FACE)


class GestureEngine:
    def __init__(self):
        self._callbacks: dict = {}
        self._last_fired: dict = {}
        self._finger_analyzer = FingerAnalyzer()
        self._classifier = GestureClassifier()
        self._voter = TemporalVoter()
        self._swipe_det = SwipeDetector()
        self._pinch_sm = PinchSmoother()
        self._renderer = CVRenderer()
        self._palm_hold = 0
        self._g_history: List[str] = []
        self._frame_num = 0

        self.running = False
        self.camera_ok = False
        self.hand_visible = False
        self.face_visible = False
        self.current_gesture = ""
        self.last_event: Optional[GestureEvent] = None
        self.error = ""
        self._thread: Optional[threading.Thread] = None

        self._p_title = "No track selected"
        self._p_artist = "—"
        self._p_playing = False
        self._p_volume = 0.75

    def on(self, gesture: str, cb: Callable) -> "GestureEngine":
        self._callbacks[gesture] = cb
        return self

    def start(self) -> bool:
        if not CV_OK:
            self.error = "pip install opencv-python mediapipe==0.10.21 numpy"
            return False
        if self.running:
            return True
        self._thread = threading.Thread(target=self._cv_loop, daemon=True, name="GestureCV")
        self._thread.start()
        return True

    def stop(self) -> None:
        self.running = False

    def sync_player_state(self, title: str, artist: str, playing: bool, volume: float) -> None:
        self._p_title = title
        self._p_artist = artist
        self._p_playing = playing
        self._p_volume = volume

    def _cv_loop(self) -> None:
        self.running = True
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_detection
        mp_draw = mp.solutions.drawing_utils
        mp_style = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                min_detection_confidence=0.82, min_tracking_confidence=0.76,
                                model_complexity=1)
        face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.52)

        cap = None
        for cam_idx in range(5):
            c = cv2.VideoCapture(cam_idx)
            if c.isOpened():
                cap = c
                break
            c.release()
        if cap is None:
            self.error = "Camera not found (checked indices 0-4)"
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_ok = True

        WIN_NAME = "MOON MUSIC  |  Vision Engine v5"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1024, 576)

        fps_time, fps_count, fps_value = time.time(), 0, 0

        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            self._frame_num += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            h_result = hands.process(rgb)
            f_result = face_detector.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps_value, fps_count, fps_time = fps_count, 0, time.time()

            face_count = 0
            self.face_visible = False
            if f_result.detections:
                face_count = len(f_result.detections)
                self.face_visible = True
                for fi, det in enumerate(f_result.detections):
                    face_snap = self._parse_face(det, frame_w, frame_h)
                    self._renderer.draw_face(frame, face_snap, fi, self._frame_num)

            gesture_name = ""
            if h_result.multi_hand_landmarks:
                hand_lm = h_result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                        mp_style.get_default_hand_landmarks_style(),
                                        mp_style.get_default_hand_connections_style())
                lm = hand_lm.landmark

                fingers = self._finger_analyzer.analyze_all(lm, frame_w, frame_h)
                pinch_raw = self._finger_analyzer.pinch_distance(lm)
                pinch_sm = self._pinch_sm.update(pinch_raw)
                palm_dir = self._finger_analyzer.palm_facing(lm)
                tilt = self._finger_analyzer.wrist_tilt(lm)
                bbox_ = self._finger_analyzer.landmark_bbox(lm, frame_w, frame_h)

                raw_g, raw_c = self._classifier.classify(fingers, pinch_raw, palm_dir, lm)
                voted_g, voted_c = self._voter.push(raw_g, raw_c)

                swipe = self._swipe_det.push(lm[0].x)
                if swipe:
                    gesture_name, voted_c = swipe, 0.95
                else:
                    gesture_name = voted_g

                snap = HandSnapshot(
                    fingers=fingers, gesture=gesture_name, confidence=voted_c,
                    pinch_raw=pinch_raw, pinch_smooth=pinch_sm,
                    pos_x=float(lm[0].x), pos_y=float(lm[0].y), depth_z=float(lm[0].z),
                    tilt_deg=tilt, palm_facing=palm_dir, bbox=bbox_,
                )

                self._palm_hold = (self._palm_hold + 1) if gesture_name == "open_palm" else 0
                self.hand_visible = True
                self._renderer.draw_hand(frame, snap, self._palm_hold)
                self._dispatch(gesture_name, snap)
            else:
                self.hand_visible = False
                self._palm_hold = 0
                self._swipe_det.reset()
                self._voter.reset()
                self._pinch_sm.reset()

            self.current_gesture = gesture_name
            self._renderer.draw_hud(frame, self._p_title, self._p_artist, self._p_playing,
                                     self._p_volume, self.hand_visible, self.face_visible,
                                     face_count, fps_value, self._g_history)

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.camera_ok = False
        self.running = False

    @staticmethod
    def _parse_face(det, w: int, h: int) -> FaceSnapshot:
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w-1, x1 + int(bb.width * w))
        y2 = min(h-1, y1 + int(bb.height * h))
        conf = float(det.score[0]) if det.score else 0.0
        pts = [(int(kp.x * w), int(kp.y * h)) for kp in det.location_data.relative_keypoints]
        return FaceSnapshot(bbox=(x1, y1, x2, y2), conf=conf, keypoints=pts,
                             cx=(x1+x2)//2, cy=(y1+y2)//2, fw=x2-x1, fh=y2-y1)

    def _can_fire(self, key: str) -> bool:
        now = time.time()
        cooldown = COOLDOWNS.get(key, 1.0)
        if now - self._last_fired.get(key, 0.0) >= cooldown:
            self._last_fired[key] = now
            return True
        return False

    def _fire(self, key: str, label: str, **kwargs) -> None:
        cb = self._callbacks.get(key)
        if cb:
            try:
                cb(**kwargs)
            except Exception as exc:
                print(f"[GestureEngine] {key}: {exc}")
        self.last_event = GestureEvent(gesture=key, label=label)
        if not self._g_history or self._g_history[-1] != key:
            self._g_history.append(key)
            if len(self._g_history) > 10:
                self._g_history.pop(0)

    def _dispatch(self, gesture: str, snap: HandSnapshot) -> None:
        if gesture == "open_palm" and snap.confidence > 0.62:
            if self._palm_hold >= PALM_HOLD_FRAMES and self._can_fire("palm"):
                self._fire("palm", "Play / Pause")
            return

        if gesture == "swipe_right" and self._can_fire("swipe_right"):
            self._fire("swipe_right", "Next Track")
            return
        if gesture == "swipe_left" and self._can_fire("swipe_left"):
            self._fire("swipe_left", "Prev Track")
            return

        if snap.confidence < MIN_CONFIDENCE:
            return

        if gesture == "pinch" and self._can_fire("pinch"):
            self._fire("pinch", "Volume Control", pinch=snap.pinch_raw)
            return

        ACTION_MAP = {
            "fist": ("fist", "Mute Toggle"), "thumb_up": ("thumb_up", "Like Track"),
            "thumb_down": ("thumb_down", "Shuffle"), "peace": ("peace", "Shuffle"),
            "point_up": ("point_up", "Cycle Repeat"), "rock": ("rock", "Rock Mode"),
            "call_me": ("call_me", "Mode Change"),
        }
        if gesture in ACTION_MAP:
            key, label = ACTION_MAP[gesture]
            if self._can_fire(key):
                self._fire(key, label)

    @property
    def status(self) -> dict:
        evt = self.last_event
        return {
            "running": self.running, "camera_ok": self.camera_ok,
            "hand_visible": self.hand_visible, "face_visible": self.face_visible,
            "current_gesture": self.current_gesture,
            "last_event": {"gesture": evt.gesture, "label": evt.label, "ts": evt.ts} if evt else None,
            "error": self.error,
        }
