"""
DeepSecure — AI Deepfake Detector for Online Interviews
Refurbished version — fixes false positives, blank-screen bug, sound test
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import threading
import time
import os
import math
import wave
import struct
import tempfile
from collections import deque

# ── Optional libs ──────────────────────────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
    _detector = dlib.get_frontal_face_detector()
    if os.path.exists("shape_predictor_68_face_landmarks.dat"):
        _predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        LANDMARKS_AVAILABLE = True
    else:
        LANDMARKS_AVAILABLE = False
except ImportError:
    DLIB_AVAILABLE = False
    LANDMARKS_AVAILABLE = False

# ── Theme ──────────────────────────────────────────────────────────────────────
BG     = "#080810"
PANEL  = "#10101c"
CARD   = "#16162a"
ACCENT = "#00f5c4"
DANGER = "#ff2d55"
WARN   = "#ffd60a"
TEXT   = "#e8e8f0"
MUTED  = "#4a4a6a"
GRID   = "#1a1a30"

# Minimum face size — rejects noise/false positives from blank screens
MIN_FACE_W = 80
MIN_FACE_H = 80

# If face present in fewer than this % of frames → "NO VALID SUBJECT"
MIN_FACE_PRESENCE = 0.28


# ══════════════════════════════════════════════════════════════════════════════
#  SOUND — pure Python, no external lib needed
# ══════════════════════════════════════════════════════════════════════════════

def _write_wav(filename, freq, duration, volume=0.6, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = []
        for i in range(n_samples):
            t = i / sample_rate
            env = 1.0 if t < duration * 0.9 else (duration - t) / (duration * 0.1)
            sample = int(32767 * volume * env * math.sin(2 * math.pi * freq * t))
            data.append(struct.pack('<h', sample))
        wf.writeframes(b''.join(data))


def _play_wav(filename):
    """Play wav using platform tools — no pygame/playsound needed."""
    import subprocess, sys
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            subprocess.Popen(["afplay", filename])
        else:  # Linux
            for player in ["aplay", "paplay", "ffplay"]:
                try:
                    result = subprocess.run(["which", player], capture_output=True)
                    if result.returncode == 0:
                        subprocess.Popen([player, "-nodisp" if player == "ffplay" else "", filename]
                                         if player == "ffplay" else [player, filename])
                        break
                except Exception:
                    continue
    except Exception:
        pass


_sound_lock = threading.Lock()


def play_genuine_sound():
    """Pleasant two-tone ding."""
    def _run():
        with _sound_lock:
            f = os.path.join(tempfile.gettempdir(), "ds_genuine.wav")
            _write_wav(f, 880,  0.18)
            _play_wav(f)
            time.sleep(0.22)
            _write_wav(f, 1174, 0.40)
            _play_wav(f)
            time.sleep(0.42)
    threading.Thread(target=_run, daemon=True).start()


def play_alarm_sound():
    """Urgent alternating alarm for ~7 seconds."""
    def _run():
        with _sound_lock:
            f = os.path.join(tempfile.gettempdir(), "ds_alarm.wav")
            for _ in range(10):
                _write_wav(f, 1480, 0.35, volume=0.92)
                _play_wav(f)
                time.sleep(0.38)
                _write_wav(f, 880,  0.30, volume=0.92)
                _play_wav(f)
                time.sleep(0.33)
    threading.Thread(target=_run, daemon=True).start()


def test_ding():
    threading.Thread(target=play_genuine_sound, daemon=True).start()


def test_alarm():
    threading.Thread(target=play_alarm_sound, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _ear(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _dct_score(gray_roi):
    """High-frequency DCT energy → GAN artifact fingerprint. 0=natural, 1=suspicious."""
    if gray_roi.size == 0:
        return 0.0
    patch = cv2.resize(gray_roi, (64, 64)).astype(np.float32)
    dct   = cv2.dct(patch)
    total = np.sum(dct ** 2) + 1e-6
    hf    = np.sum(dct[32:, 32:] ** 2)
    return float(np.clip((hf / total - 0.03) / 0.07, 0.0, 1.0))


def _texture_score(face_bgr):
    """GAN skin smoothing check — too uniform = suspicious. 0=natural, 1=suspicious."""
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    std  = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).std()
    return float(np.clip(1.0 - std / 20.0, 0.0, 1.0))


def _lighting_score(prev_gray, curr_gray, rect):
    """Sudden face illumination change between frames. 0=consistent, 1=suspicious."""
    if prev_gray is None:
        return 0.0
    x, y, w, h = rect
    try:
        p = cv2.resize(prev_gray[y:y+h, x:x+w], (32, 32)).astype(np.float32)
        c = cv2.resize(curr_gray[y:y+h, x:x+w], (32, 32)).astype(np.float32)
        return float(np.clip(np.abs(p - c).mean() / 40.0, 0.0, 1.0))
    except Exception:
        return 0.0


def _jitter_score(prev_rect, curr_rect):
    """Unnatural face-box flicker between frames. 0=stable, 1=jittery."""
    if prev_rect is None:
        return 0.0
    dx = abs(prev_rect[0] - curr_rect[0])
    dy = abs(prev_rect[1] - curr_rect[1])
    return float(np.clip((dx + dy) / 2.0 / 30.0, 0.0, 1.0))


def _is_valid_face(x, y, w, h, frame_w, frame_h):
    """
    Reject tiny/edge detections that are almost certainly false positives
    from blank screens, walls, or low-light noise.
    """
    if w < MIN_FACE_W or h < MIN_FACE_H:
        return False
    # Face must be reasonably centred — not in the extreme corners
    cx, cy = x + w // 2, y + h // 2
    if cx < frame_w * 0.05 or cx > frame_w * 0.95:
        return False
    if cy < frame_h * 0.05 or cy > frame_h * 0.95:
        return False
    # Aspect ratio sanity (faces are roughly square)
    ratio = w / max(h, 1)
    if ratio < 0.5 or ratio > 2.0:
        return False
    return True


class Analyzer:
    """Collects per-frame signals and produces a final verdict."""

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.reset()

    def reset(self):
        self.scores       = []
        self.dct_sc       = []
        self.tex_sc       = []
        self.light_sc     = []
        self.jitter_sc    = []
        self.blink_events = 0
        self.ear_hist     = deque(maxlen=60)
        self.emotions     = set()
        self.prev_gray    = None
        self.prev_rect    = None
        self.total        = 0
        self.with_face    = 0

    # ── face detection ──────────────────────────────────────────────────────

    def _detect(self, frame, gray):
        h, w = gray.shape[:2]
        faces = []

        if DLIB_AVAILABLE:
            rects = _detector(gray, 0)
            for r in rects:
                fw, fh = r.width(), r.height()
                if _is_valid_face(r.left(), r.top(), fw, fh, w, h):
                    faces.append((r.left(), r.top(), fw, fh))
        
        if not faces:
            det = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6,
                minSize=(MIN_FACE_W, MIN_FACE_H))
            if len(det) > 0:
                for (fx, fy, fw, fh) in det:
                    if _is_valid_face(fx, fy, fw, fh, w, h):
                        faces.append((fx, fy, fw, fh))

        # Stricter fallback only if nothing found yet
        if not faces:
            eq = cv2.equalizeHist(gray)
            det = self._cascade.detectMultiScale(
                eq, scaleFactor=1.05, minNeighbors=8,
                minSize=(MIN_FACE_W + 20, MIN_FACE_H + 20))
            if len(det) > 0:
                for (fx, fy, fw, fh) in det:
                    if _is_valid_face(fx, fy, fw, fh, w, h):
                        faces.append((fx, fy, fw, fh))

        return faces

    # ── main per-frame call ─────────────────────────────────────────────────

    def process(self, frame):
        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out    = frame.copy()
        self.total += 1

        faces = self._detect(frame, gray)
        if not faces:
            self.prev_gray = gray
            self.prev_rect = None
            # Overlay "NO FACE" indicator
            cv2.rectangle(out, (0, 0), (fw, 36), (10, 10, 25), -1)
            cv2.putText(out, "NO FACE DETECTED", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 200), 2)
            return out, {}

        self.with_face += 1
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = frame[max(0, y):y+h, max(0, x):x+w]
        if roi.size == 0:
            return out, {}

        g_roi   = gray[max(0, y):y+h, max(0, x):x+w]
        dct_s   = _dct_score(g_roi)
        tex_s   = _texture_score(roi)
        light_s = _lighting_score(self.prev_gray, gray, (x, y, w, h))
        jit_s   = _jitter_score(self.prev_rect, (x, y, w, h))

        self.dct_sc.append(dct_s)
        self.tex_sc.append(tex_s)
        self.light_sc.append(light_s)
        self.jitter_sc.append(jit_s)

        # ── blink (EAR) ──
        ear = None
        if LANDMARKS_AVAILABLE:
            try:
                drects = _detector(gray, 0)
                if drects:
                    shape = _predictor(gray, drects[0])
                    pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                    le, re = pts[36:42], pts[42:48]
                    ear = (_ear(le) + _ear(re)) / 2.0
                    self.ear_hist.append(ear)
                    if len(self.ear_hist) >= 3:
                        if self.ear_hist[-2] < 0.21 < self.ear_hist[-3]:
                            self.blink_events += 1
                    for pt in np.vstack([le, re]):
                        cv2.circle(out, tuple(pt), 2, (0, 255, 255), -1)
            except Exception:
                pass

        # ── emotion (every 20 frames) ──
        if DEEPFACE_AVAILABLE and self.total % 20 == 0:
            try:
                res = DeepFace.analyze(frame, actions=['emotion'],
                                       enforce_detection=False, silent=True)
                em  = res[0]['dominant_emotion'] if isinstance(res, list) else res['dominant_emotion']
                self.emotions.add(em)
            except Exception:
                pass

        # ── composite score ──
        fs = 0.35 * dct_s + 0.30 * tex_s + 0.20 * light_s + 0.15 * jit_s
        self.scores.append(fs)

        # ── draw face box ──
        color = (0, 200, 80) if fs < 0.4 else (0, 140, 255) if fs < 0.65 else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

        # risk bar under face box
        bw = int(w * fs)
        cv2.rectangle(out, (x, y+h+4), (x+w, y+h+14), (30, 30, 45), -1)
        if bw > 0:
            cv2.rectangle(out, (x, y+h+4), (x+bw, y+h+14), color, -1)

        # HUD signals top-left
        cv2.rectangle(out, (0, 0), (fw, 38), (10, 10, 25), -1)
        cv2.putText(out, f"Risk {int(fs*100)}%", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Signal mini-labels above face box
        for i, (lbl, val) in enumerate([("DCT", dct_s), ("TEX", tex_s), ("LIT", light_s)]):
            col2 = (0, 220, 100) if val < 0.4 else (0, 160, 255) if val < 0.65 else (0, 60, 255)
            cv2.putText(out, f"{lbl}:{val:.2f}", (x, y - 10 - i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col2, 1)

        self.prev_gray = gray
        self.prev_rect = (x, y, w, h)
        return out, {'dct': dct_s, 'tex': tex_s, 'light': light_s, 'jit': jit_s, 'fs': fs}

    # ── verdict ─────────────────────────────────────────────────────────────

    def verdict(self):
        if not self.scores:
            return None

        face_present = self.with_face / max(self.total, 1)

        # ── KEY FIX: if face barely detected → reject entirely ──
        if face_present < MIN_FACE_PRESENCE:
            return {
                'score':       0,
                'verdict':     'NO VALID SUBJECT',
                'mean_frame':  0.0,
                'hi_risk_pct': 0.0,
                'blink_rpm':   0.0,
                'emotions':    [],
                'face_pct':    round(face_present * 100, 1),
                'frames':      self.total,
                'reason':      'Face detected in fewer than 28% of frames. No subject present or subject too far from camera.'
            }

        mean_s    = np.mean(self.scores)
        hi_risk   = np.mean([s > 0.5 for s in self.scores])
        dur_sec   = self.total / 30.0
        blink_rpm = (self.blink_events / max(dur_sec, 1)) * 60

        blink_s = 0.0
        if LANDMARKS_AVAILABLE and dur_sec > 5:
            blink_s = 0.8 if blink_rpm < 3 else 0.4 if blink_rpm < 8 else 0.0

        emotion_s = max(0.0, 0.3 - len(self.emotions) * 0.1)

        final = float(np.clip(
            0.40 * mean_s +
            0.15 * hi_risk +
            0.20 * blink_s +
            0.10 * emotion_s +
            0.15 * (1.0 - face_present),
            0.0, 1.0
        ))

        label = "SUSPICIOUS" if final > 0.35 else "LIKELY GENUINE"
        return {
            'score':        int(final * 100),
            'verdict':      label,
            'mean_frame':   round(mean_s * 100, 1),
            'hi_risk_pct':  round(hi_risk * 100, 1),
            'blink_rpm':    round(blink_rpm, 1),
            'emotions':     list(self.emotions),
            'face_pct':     round(face_present * 100, 1),
            'frames':       self.total,
            'reason':       None
        }


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS
# ══════════════════════════════════════════════════════════════════════════════

def _video_worker(path, progress_cb, done_cb, stop_ev):
    az  = Analyzer()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        done_cb(None, "Cannot open video file.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    skip  = max(1, int(fps / 10))

    cv2.namedWindow("DeepSecure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepSecure", 720, 480)

    idx = 0
    while not stop_ev.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % skip:
            continue
        frame        = cv2.resize(frame, (640, 480))
        annotated, _ = az.process(frame)
        pct          = int(idx / max(total, 1) * 100)
        progress_cb(pct, f"Analyzing frame {idx}/{total}")
        cv2.imshow("DeepSecure", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    done_cb(az.verdict(), None)


def _webcam_worker(progress_cb, done_cb, stop_ev, duration=15, countdown=3):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        done_cb(None, "Cannot access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("DeepSecure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepSecure", 720, 500)

    # countdown
    t0 = time.time()
    while time.time() - t0 < countdown:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        ov    = frame.copy()
        cv2.rectangle(ov, (0, 0), (640, 480), (8, 8, 22), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        rem = countdown - int(time.time() - t0)
        cv2.putText(frame, str(rem),      (270, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 5,   (0, 245, 196), 10)
        cv2.putText(frame, "GET READY",   (190, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 200), 2)
        cv2.imshow("DeepSecure", frame)
        cv2.waitKey(1)

    az    = Analyzer()
    t_rec = time.time()
    while not stop_ev.is_set():
        elapsed = time.time() - t_rec
        if elapsed > duration:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame        = cv2.flip(frame, 1)
        frame        = cv2.resize(frame, (640, 480))
        annotated, _ = az.process(frame)
        pct          = int(elapsed / duration * 100)
        remaining    = duration - int(elapsed)
        progress_cb(pct, f"Recording… {remaining}s left")

        # REC dot + timer
        cv2.circle(annotated, (618, 22), 10, (0, 0, 255), -1)
        cv2.putText(annotated, f"REC {remaining}s", (560, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("DeepSecure", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    done_cb(az.verdict(), None)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT POPUP
# ══════════════════════════════════════════════════════════════════════════════

def show_result(result, parent):
    score   = result['score']
    verdict = result['verdict']

    COLOR_MAP = {
        "SUSPICIOUS":      DANGER,
        "UNCERTAIN":       WARN,
        "LIKELY GENUINE":  ACCENT,
        "NO VALID SUBJECT": MUTED,
    }
    v_col = COLOR_MAP.get(verdict, MUTED)

    # ── sound ──
    if verdict == "LIKELY GENUINE":
        play_genuine_sound()
    elif verdict in ("SUSPICIOUS", "UNCERTAIN"):
        play_alarm_sound()
    # NO VALID SUBJECT → no sound

    win = tk.Toplevel(parent)
    win.title("DeepSecure — Result")
    win.geometry("520x620")
    win.config(bg=BG)
    win.resizable(False, False)
    win.grab_set()

    # Header
    tk.Label(win, text="ANALYSIS COMPLETE",
             font=("Courier New", 11, "bold"), fg=MUTED, bg=BG).pack(pady=(26, 4))

    # Score circle (simulated with big label)
    score_frame = tk.Frame(win, bg=CARD, width=160, height=160)
    score_frame.pack(pady=8)
    score_frame.pack_propagate(False)
    tk.Label(score_frame, text=f"{score}%",
             font=("Courier New", 52, "bold"), fg=v_col, bg=CARD).pack(expand=True)

    tk.Label(win, text="DEEPFAKE RISK SCORE",
             font=("Courier New", 8), fg=MUTED, bg=BG).pack()
    tk.Label(win, text=verdict,
             font=("Courier New", 18, "bold"), fg=v_col, bg=BG).pack(pady=(6, 4))

    # Reason (for NO VALID SUBJECT)
    reason = result.get('reason')
    if reason:
        tk.Label(win, text=reason, font=("Courier New", 8),
                 fg=MUTED, bg=BG, wraplength=440, justify="center").pack(pady=(0, 6))

    # Sound indicator
    if verdict == "LIKELY GENUINE":
        sound_msg = "🔔 Ding played"
    elif verdict in ("SUSPICIOUS", "UNCERTAIN"):
        sound_msg = "🚨 Alarm sounding…"
    else:
        sound_msg = "🔇 No alert (no subject)"
    tk.Label(win, text=sound_msg, font=("Courier New", 8),
             fg=v_col, bg=BG).pack(pady=(0, 6))

    tk.Frame(win, bg=GRID, height=1).pack(fill="x", padx=36)

    # Metrics grid
    mf = tk.Frame(win, bg=BG)
    mf.pack(pady=14, padx=40, fill="x")
    mf.columnconfigure(0, weight=1)
    mf.columnconfigure(1, weight=1)

    rows = [
        ("Frames Analyzed",   result.get('frames',       'N/A')),
        ("Face Present",      f"{result.get('face_pct',   0)}%"),
        ("High-Risk Frames",  f"{result.get('hi_risk_pct',0)}%"),
        ("Avg Frame Risk",    f"{result.get('mean_frame', 0)}%"),
        ("Blink Rate",        f"{result.get('blink_rpm',  0)}/min"
                              + (" ⚠ needs dlib" if not LANDMARKS_AVAILABLE else "")),
        ("Emotions Seen",     ', '.join(result.get('emotions', []))
                              or ("⚠ needs DeepFace" if not DEEPFACE_AVAILABLE else "none")),
    ]
    for i, (lbl, val) in enumerate(rows):
        tk.Label(mf, text=lbl, font=("Courier New", 9),
                 fg=MUTED, bg=BG, anchor="w").grid(row=i, column=0, sticky="w", pady=3)
        tk.Label(mf, text=str(val), font=("Courier New", 9, "bold"),
                 fg=TEXT, bg=BG, anchor="e").grid(row=i, column=1, sticky="e", pady=3)

    tk.Frame(win, bg=GRID, height=1).pack(fill="x", padx=36, pady=6)

    info = {
        "SUSPICIOUS":      "Multiple deepfake signals detected.\nHigh DCT artifacts + suspicious texture/lighting.",
        "UNCERTAIN":       "Some anomalies found.\nCould be compression, poor lighting, or low-quality camera.",
        "LIKELY GENUINE":  "No significant deepfake artifacts detected.\nVideo appears consistent with genuine footage.",
        "NO VALID SUBJECT":"Face was absent or too small for most of the clip.\nCannot produce a valid deepfake analysis.",
    }
    tk.Label(win, text=info.get(verdict, ""),
             font=("Courier New", 8), fg=MUTED, bg=BG, justify="center").pack(pady=4)

    tk.Button(win, text="  CLOSE  ",
              font=("Courier New", 10, "bold"),
              bg=CARD, fg=TEXT, relief="flat", padx=24, pady=10,
              activebackground=ACCENT, activeforeground=BG,
              command=win.destroy).pack(pady=14)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root    = root
        self.stop_ev = threading.Event()
        self.running = False
        root.title("DeepSecure AI")
        root.geometry("480x480")
        root.config(bg=BG)
        root.resizable(False, False)
        self._ui()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _ui(self):
        # ── header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=32, pady=(36, 0))
        tk.Label(hdr, text="DEEP",   font=("Courier New", 32, "bold"), fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr, text="SECURE", font=("Courier New", 32, "bold"), fg=TEXT,   bg=BG).pack(side="left")

        tk.Label(self.root, text="AI Interview Deepfake Detection",
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(anchor="w", padx=32)
        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill="x", padx=32, pady=16)

        # ── library badges ───────────────────────────────────────────────────
        bf = tk.Frame(self.root, bg=BG)
        bf.pack(fill="x", padx=32, pady=(0, 20))
        for lbl, ok in [("DeepFace", DEEPFACE_AVAILABLE),
                        ("dlib",     DLIB_AVAILABLE),
                        ("Landmarks",LANDMARKS_AVAILABLE)]:
            f = tk.Frame(bf, bg=CARD, padx=8, pady=5)
            f.pack(side="left", padx=(0, 6))
            tk.Label(f, text=("✓ " if ok else "✗ ") + lbl,
                     font=("Courier New", 8),
                     fg=ACCENT if ok else DANGER, bg=CARD).pack()

        # ── hidden progress (needed for worker callbacks) ────────────────────
        self.prog_lbl = tk.Label(self.root, text="",
                                 font=("Courier New", 9), fg=MUTED, bg=BG)
        self.prog_lbl.pack()
        self.prog_var = tk.DoubleVar()
        style = ttk.Style()
        style.theme_use('default')
        style.configure("DS.Horizontal.TProgressbar",
                        troughcolor=BG, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT, bordercolor=BG)
        # Progress bar hidden — only shown while running
        self._prog_bar = ttk.Progressbar(self.root, variable=self.prog_var,
                        style="DS.Horizontal.TProgressbar",
                        length=416, mode='determinate')

        # ── action buttons ───────────────────────────────────────────────────
        bf2 = tk.Frame(self.root, bg=BG)
        bf2.pack(fill="x", padx=32, pady=8)

        self.btn_vid = tk.Button(
            bf2, text="▶   UPLOAD VIDEO",
            font=("Courier New", 12, "bold"),
            bg=ACCENT, fg=BG, relief="flat", pady=16,
            activebackground="#00c9a0",
            command=self._start_video)
        self.btn_vid.pack(fill="x", pady=(0, 10))

        self.btn_cam = tk.Button(
            bf2, text="⬤   LIVE WEBCAM  (15 s)",
            font=("Courier New", 12, "bold"),
            bg=CARD, fg=ACCENT, relief="flat", pady=16,
            activebackground=CARD,
            command=self._start_webcam)
        self.btn_cam.pack(fill="x")

        self.btn_stop = tk.Button(
            bf2, text="■   STOP",
            font=("Courier New", 10),
            bg=DANGER, fg="white", relief="flat", pady=10,
            activebackground="#cc2040",
            command=self._stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=(10, 0))

        tk.Label(self.root,
                 text="Tip: good lighting · face centred · 10–30 s clip for best results",
                 font=("Courier New", 7), fg=MUTED, bg=BG).pack(pady=(14, 0))

    # ── helpers ─────────────────────────────────────────────────────────────

    def _set_busy(self, busy):
        self.running = busy
        s = "disabled" if busy else "normal"
        self.btn_vid.config(state=s)
        self.btn_cam.config(state=s)
        self.btn_stop.config(state="normal" if busy else "disabled")
        # Show/hide progress bar
        if busy:
            self._prog_bar.pack(padx=32, pady=(6, 0))
        else:
            self._prog_bar.pack_forget()

    def _progress(self, pct, msg):
        self.root.after(0, lambda p=pct, m=msg: (
            self.prog_var.set(p),
            self.prog_lbl.config(text=m)
        ))

    def _done(self, result, error):
        def _ui():
            self._set_busy(False)
            self.prog_var.set(0)
            self.prog_lbl.config(text="")
            if error:
                messagebox.showerror("Error", error)
            elif result:
                show_result(result, self.root)
            else:
                messagebox.showinfo("Done", "No analysis data produced.")
        self.root.after(0, _ui)

    def _stop(self):
        self.stop_ev.set()

    def _start_video(self):
        path = filedialog.askopenfilename(
            title="Select Interview Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            return
        self.stop_ev.clear()
        self._set_busy(True)
        threading.Thread(
            target=_video_worker,
            args=(path, self._progress, self._done, self.stop_ev),
            daemon=True).start()

    def _start_webcam(self):
        self.stop_ev.clear()
        self._set_busy(True)
        threading.Thread(
            target=_webcam_worker,
            args=(self._progress, self._done, self.stop_ev),
            daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()