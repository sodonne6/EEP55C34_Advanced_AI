from __future__ import annotations

import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# -----------------------
# Shared state
# -----------------------
lock = threading.Lock()
is_recording = False
recorded_frames = deque(maxlen=60 * 30)  # ~30s at ~60fps max cap
last_transcript = "Ready..."

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SLT Preview GUI</title>
  <style>
    html, body { margin:0; padding:0; width:100%; height:100%; background:#0f1115; color:#e8ecf3; font-family:Segoe UI,sans-serif; }
    .app { width:100vw; height:100vh; display:grid; grid-template-rows:90px 80px 1fr 120px; }
    .topbar { display:flex; align-items:center; justify-content:center; border-bottom:1px solid #2a3240; font-size:1.2rem; font-weight:700; }
    .statusbar { display:flex; align-items:center; justify-content:center; gap:16px; border-bottom:1px solid #2a3240; }
    .pill { padding:8px 14px; border-radius:999px; border:1px solid #2f3a4d; background:#141b29; font-size:0.95rem; }
    .pill.rec { border-color:#8b1f2f; background:#2a1016; color:#ffb8c1; }
    .video-wrap { display:flex; align-items:center; justify-content:center; padding:18px; box-sizing:border-box; }
    .video-card { width:min(92vw,1400px); height:min(74vh,780px); background:#111722; border:1px solid #2f3a4d; border-radius:16px; overflow:hidden; display:flex; align-items:center; justify-content:center; }
    .video-card img { width:100%; height:100%; object-fit:contain; background:black; }
    .transcript { border-top:1px solid #2a3240; padding:12px 20px; display:flex; flex-direction:column; justify-content:center; gap:6px; }
    .label { font-size:0.85rem; opacity:0.75; }
    .text { font-size:1.15rem; font-weight:500; min-height: 1.5em; }
    .hint { font-size:0.85rem; opacity:0.75; }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">Sign 2 Speech</div>

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
      <div class="hint">Current mode: webcam + capture flow only. Hook your SLT model in run_inference_on_frames().</div>
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
      if (data.is_recording) {
        pill.textContent = "Recording...";
        pill.classList.add("rec");
      } else {
        pill.textContent = "Idle";
        pill.classList.remove("rec");
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


def run_inference_on_frames(frames) -> str:
    """
    Placeholder for your SLT inference call.
    Replace with your actual model inference pipeline.
    """
    # Example stub result
    n = len(frames)
    return f"[placeholder] captured {n} frames. Replace with SLT model output."


def gen_frames():
    global is_recording

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)

        with lock:
            rec = is_recording
            if rec:
                recorded_frames.append(frame.copy())

        # Visual indicator on frame
        if rec:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/toggle_recording", methods=["POST"])
def toggle_recording():
    global is_recording, last_transcript

    with lock:
        is_recording = not is_recording

        if is_recording:
            recorded_frames.clear()
            last_transcript = "Recording started..."
            return jsonify({"is_recording": True, "transcript": last_transcript})

        # recording just stopped -> run inference
        frames = list(recorded_frames)

    # Run inference outside lock
    if len(frames) == 0:
        last_transcript = "[info] no frames captured."
    else:
        last_transcript = run_inference_on_frames(frames)

    return jsonify({"is_recording": False, "transcript": last_transcript})


@app.route("/state")
def state():
    with lock:
        return jsonify({"is_recording": is_recording, "transcript": last_transcript})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)