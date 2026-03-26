from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

import mediapipe as mp

app = Flask(__name__)

# ---------------------------------------------------------------------
# Repo-relative path helpers
# ---------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_PATHS = {
    "SLT_REPO_ROOT": PROJECT_ROOT / "SLT" / "external" / "signformer_gcn" / "English" / "slt_how2sign_wicv2023",
    "SLT_CKPT": PROJECT_ROOT / "ckpt" / "checkpoint.best_sacrebleu_6.5700.pt",
    "SLT_SPM_MODEL": PROJECT_ROOT / "ckpt" / "spm" / "spm_bpe_7k.model",
    "SLT_MODEL_PY": PROJECT_ROOT / "SLT" / "signformer_overrides" / "sign2text_transformer_3_gcn.py",
    "SLT_GRAPH_PY": PROJECT_ROOT / "SLT" / "signformer_overrides" / "graph.py",
}

PLACEHOLDER_ENV_VALUES = {
    "",
    "PUT_YOUR_DATA_DIR_HERE",
}


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _get_path(name: str, required: bool = True) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()

    if raw in PLACEHOLDER_ENV_VALUES:
        raw = ""

    if raw:
        path = _resolve_repo_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        return path

    path = DEFAULT_PATHS.get(name)

    if path is None:
        if required:
            raise RuntimeError(f"Missing required path for {name}")
        return None

    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    return path


# ---------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------
DISPLAY_PORT = int(os.environ.get("SLT_APP_PORT", "7860"))
RECORD_MAX_FRAMES = int(os.environ.get("SLT_RECORD_MAX_FRAMES", "180"))
RECORD_EVERY_N = int(os.environ.get("SLT_RECORD_EVERY_N", "2"))
STORE_MAX_SIDE = int(os.environ.get("SLT_STORE_MAX_SIDE", "320"))
PRESTACK_MAX_FRAMES = int(os.environ.get("SLT_PRESTACK_MAX_FRAMES", "64"))
DRAW_LANDMARKS = os.environ.get("SLT_DRAW_LANDMARKS", "1") == "1"

# ---------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------
lock = threading.Lock()
is_recording = False
recorded_frames = deque(maxlen=RECORD_MAX_FRAMES)
last_transcript = "Ready..."
translator = None
translator_error = None
record_counter = 0

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SLT Preview GUI Demo</title>
  <style>
    html, body {
      margin:0; padding:0; width:100%; height:100%;
      background:#0f1115; color:#e8ecf3; font-family:Segoe UI,sans-serif;
    }
    .app {
      width:100vw; height:100vh;
      display:grid;
      grid-template-rows:90px 80px 1fr 120px;
    }
    .topbar {
      display:flex; align-items:center; justify-content:center;
      border-bottom:1px solid #2a3240;
      font-size:1.2rem; font-weight:700;
    }
    .statusbar {
      display:flex; align-items:center; justify-content:center; gap:16px;
      border-bottom:1px solid #2a3240;
    }
    .pill {
      padding:8px 14px; border-radius:999px;
      border:1px solid #2f3a4d; background:#141b29; font-size:0.95rem;
    }
    .pill.rec {
      border-color:#8b1f2f; background:#2a1016; color:#ffb8c1;
    }
    .pill.err {
      border-color:#7a4b00; background:#2b1c00; color:#ffd28a;
    }
    .video-wrap {
      display:flex; align-items:center; justify-content:center;
      padding:18px; box-sizing:border-box;
    }
    .video-card {
      width:min(92vw,1400px); height:min(74vh,780px);
      background:#111722; border:1px solid #2f3a4d; border-radius:16px;
      overflow:hidden; display:flex; align-items:center; justify-content:center;
    }
    .video-card img {
      width:100%; height:100%; object-fit:contain; background:black;
    }
    .transcript {
      border-top:1px solid #2a3240;
      padding:12px 20px;
      display:flex; flex-direction:column; justify-content:center; gap:6px;
    }
    .label { font-size:0.85rem; opacity:0.75; }
    .text { font-size:1.15rem; font-weight:500; min-height:1.5em; }
    .hint { font-size:0.85rem; opacity:0.75; }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">Sign 2 Speech Demo</div>

    <div class="statusbar">
      <div id="recPill" class="pill">Idle</div>
      <div class="pill">Shortcut: Ctrl+R to Start/Stop</div>
    </div>

    <div class="video-wrap">
      <div class="video-card"><img src="/video_feed" alt="Webcam stream"/></div>
    </div>

    <div class="transcript">
      <div class="label">Transcript</div>
      <div id="transcriptText" class="text">Ready...</div>
      <div class="hint">Demo mode: live MediaPipe overlay + MediaPipe-only pseudo-I3D inference.</div>
    </div>
  </div>

  <script>
    let ctrlPressed = false;

    async function toggleRecording() {
      const res = await fetch("/toggle_recording", { method: "POST" });
      const data = await res.json();
      updateStatus(data);
    }

    function updateStatus(data) {
      const pill = document.getElementById("recPill");
      const txt = document.getElementById("transcriptText");

      pill.classList.remove("rec");
      pill.classList.remove("err");

      if (data.is_recording) {
        pill.textContent = "Recording...";
        pill.classList.add("rec");
      } else if (data.error) {
        pill.textContent = "Error";
        pill.classList.add("err");
      } else {
        pill.textContent = "Idle";
      }

      if (data.transcript !== undefined) {
        txt.textContent = data.transcript;
      }
    }

    async function pollState() {
      try {
        const res = await fetch("/state");
        const data = await res.json();
        updateStatus(data);
      } catch (_) {}
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "Control") ctrlPressed = true;
      if (ctrlPressed && (e.key === "r" || e.key === "R")) {
        e.preventDefault();
        toggleRecording();
      }
    });

    document.addEventListener("keyup", (e) => {
      if (e.key === "Control") ctrlPressed = false;
    });

    setInterval(pollState, 500);
    pollState();
  </script>
</body>
</html>
"""


class LiveHolisticOverlay:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        out = frame_bgr.copy()

        if results.pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                out,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                out,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                out,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_styles.get_default_hand_connections_style(),
            )
        return out


overlay_helper = LiveHolisticOverlay() if DRAW_LANDMARKS else None


def _resize_for_recording(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= STORE_MAX_SIDE:
        return frame_bgr.copy()

    scale = STORE_MAX_SIDE / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _subsample_list_uniform(frames, max_frames: int):
    if max_frames is None or len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(np.int64)
    return [frames[int(i)] for i in idx]


def get_translator():
    global translator, translator_error

    if translator is not None:
        return translator
    if translator_error is not None:
        raise RuntimeError(translator_error)

    try:
        repo_root = _get_path("SLT_REPO_ROOT")
        ckpt = _get_path("SLT_CKPT")
        spm_model = _get_path("SLT_SPM_MODEL")
        data_dir = _get_path("SLT_DATA_DIR", required=False)

        model_py = _get_path("SLT_MODEL_PY")
        graph_py = _get_path("SLT_GRAPH_PY")

        device = os.environ.get("SLT_DEVICE", "cpu")
        max_input_frames = int(os.environ.get("SLT_MAX_INPUT_FRAMES", "64"))
        min_input_frames = int(os.environ.get("SLT_MIN_INPUT_FRAMES", "24"))

        print(f"[info] PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
        print(f"[info] SLT_REPO_ROOT: {repo_root}", flush=True)
        print(f"[info] SLT_CKPT: {ckpt}", flush=True)
        print(f"[info] SLT_SPM_MODEL: {spm_model}", flush=True)
        print(f"[info] SLT_DATA_DIR: {data_dir if data_dir is not None else '[auto minimal task dir]'}", flush=True)
        print(f"[info] SLT_MODEL_PY: {model_py}", flush=True)
        print(f"[info] SLT_GRAPH_PY: {graph_py}", flush=True)

        from inference_slt_i3d_3gcn_demo import SignLanguageTranslatorI3D3GCNDemo

        translator = SignLanguageTranslatorI3D3GCNDemo(
            repo_root=repo_root,
            ckpt_path=ckpt,
            spm_model=spm_model,
            data_dir=data_dir,
            device=device,
            model_py=model_py,
            graph_py=graph_py,
            max_input_frames=max_input_frames,
            min_input_frames=min_input_frames,
        )
        return translator

    except Exception as e:
        translator_error = f"{type(e).__name__}: {e}"
        raise


def run_inference_on_frames(frames) -> str:
    try:
        if not frames:
            return "[info] no frames captured."

        frames = _subsample_list_uniform(frames, PRESTACK_MAX_FRAMES)
        arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)

        tr = get_translator()
        beam = int(os.environ.get("SLT_BEAM", "3"))
        max_len_b = int(os.environ.get("SLT_MAX_LEN_B", "32"))
        return tr.translate_frames(arr, beam=beam, max_len_b=max_len_b)

    except Exception as e:
        return f"[error] inference failed: {type(e).__name__}: {e}"


def gen_frames():
    global is_recording, record_counter

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            continue

        display_frame = cv2.flip(raw_frame, 1)

        if overlay_helper is not None:
            try:
                display_frame = overlay_helper.draw(display_frame)
            except Exception:
                pass

        with lock:
            rec = is_recording
            if rec:
                record_counter += 1
                if record_counter % RECORD_EVERY_N == 0:
                    stored = _resize_for_recording(raw_frame)
                    recorded_frames.append(stored)

        if rec:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (50, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ok, buf = cv2.imencode(".jpg", display_frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/toggle_recording", methods=["POST"])
def toggle_recording():
    global is_recording, last_transcript, record_counter

    with lock:
        is_recording = not is_recording

        if is_recording:
            recorded_frames.clear()
            record_counter = 0
            last_transcript = "Recording started..."
            return jsonify({"is_recording": True, "transcript": last_transcript, "error": False})

        frames = list(recorded_frames)

    if len(frames) == 0:
        last_transcript = "[info] no frames captured."
    else:
        last_transcript = run_inference_on_frames(frames)

    has_error = str(last_transcript).startswith("[error]")
    return jsonify({"is_recording": False, "transcript": last_transcript, "error": has_error})


@app.route("/state")
def state():
    with lock:
        err = str(last_transcript).startswith("[error]")
        return jsonify({"is_recording": is_recording, "transcript": last_transcript, "error": err})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=DISPLAY_PORT, debug=False, threaded=True)