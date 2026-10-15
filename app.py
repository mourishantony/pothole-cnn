"""Pothole detection website: upload a road video -> annotated video + verdict.

Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:5000
The classifier works on image tiles (trained on crops), so each sampled frame is
split into overlapping square tiles over the road area; tiles above the threshold are boxed.
"""
import json
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, abort, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).parent
RUNS = ROOT / "web_runs"
RUNS.mkdir(exist_ok=True)
CLASSES = json.loads((ROOT / "models" / "classes.json").read_text())  # ['crack','normal','pothole']
MODEL = tf.keras.models.load_model(ROOT / "models" / "final.keras", compile=False)
IMG = 224
THRESHOLDS = {"pothole": 0.95, "crack": 0.90}   # tile probability needed to flag damage (tuned on unseen RDD2022 frames)
MIN_TILES = 2          # neighbouring tiles that must agree for a pothole
TILE_FRAC = 0.30       # tile side as fraction of the road area (smaller tiles worked best)
MIN_FRAMES = 3         # sampled frames with a pothole needed for a "pothole found" verdict
COLORS = {"pothole": (0, 0, 255), "crack": (0, 165, 255)}
MAX_MB = 300
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def tiles_for(frame):
    """Overlapping square tiles over the lower ~75% of the frame (road area)."""
    H, W = frame.shape[:2]
    y_start = int(H * 0.25)
    side = max(int(min(W, H - y_start) * TILE_FRAC), 64)
    step = max(side // 2, 1)
    boxes = []
    for y in range(y_start, H - side + 1, step) if H - y_start > side else [y_start]:
        for x in range(0, W - side + 1, step) if W > side else [0]:
            boxes.append((x, y, side))
    return boxes


def analyse(frame):
    boxes = tiles_for(frame)
    batch = np.stack([cv2.cvtColor(cv2.resize(frame[y:y + s, x:x + s], (IMG, IMG)), cv2.COLOR_BGR2RGB)
                      for x, y, s in boxes]).astype("float32")
    probs = MODEL.predict(batch, verbose=0, batch_size=64)
    hits = {"pothole": [], "crack": []}
    for (x, y, s), p in zip(boxes, probs):
        k = int(p.argmax())
        if CLASSES[k] != "normal" and p[k] >= THRESHOLDS[CLASSES[k]]:
            hits[CLASSES[k]].append((x, y, s, float(p[k])))
    dets = []
    for kind, hs in hits.items():  # merge overlapping tiles into one box per damaged area
        if not hs:
            continue
        mask = np.zeros(frame.shape[:2], np.uint8)
        for x, y, s, _ in hs:
            mask[y:y + s, x:x + s] = 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            bx, by, bw, bh = st[i][:4]
            inside = [h for h in hs if bx <= h[0] + h[2] / 2 <= bx + bw and by <= h[1] + h[2] / 2 <= by + bh]
            if kind == "pothole" and len(inside) < MIN_TILES:
                continue  # an isolated tile is likely a false alarm
            dets.append((kind, max(h[3] for h in inside), int(bx), int(by), (int(bw), int(bh))))
    return dets


def process(path, out_dir, every_sec=0.5):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot read this video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, 960 / max(W, H))                     # keep processing fast
    W2, H2 = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    raw = out_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2))
    step = max(int(fps * every_sec), 1)
    i, dets, pothole_frames, crack_frames, sampled, best = 0, [], 0, 0, 0, (0.0, None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W2, H2))
        if i % step == 0:
            dets = analyse(frame)
            sampled += 1
            kinds = {d[0] for d in dets}
            pothole_frames += "pothole" in kinds
            crack_frames += "crack" in kinds
            for d in dets:
                if d[0] == "pothole" and d[1] > best[0]:
                    ann = frame.copy()
                    for k, c, x, y, s in dets:
                        cv2.rectangle(ann, (x, y), (x + s[0], y + s[1]), COLORS[k], 3)
                    best = (d[1], ann)
        for k, c, x, y, s in dets:  # keep last detections on screen between samples
            cv2.rectangle(frame, (x, y), (x + s[0], y + s[1]), COLORS[k], 2)
            cv2.putText(frame, f"{k} {c:.0%}", (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[k], 2)
        writer.write(frame)
        i += 1
    cap.release(); writer.release()
    final = out_dir / "annotated.mp4"
    if FFMPEG:  # browsers need H.264
        r = subprocess.run([FFMPEG, "-y", "-i", str(raw), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(final)], capture_output=True)
        if r.returncode != 0:
            raw.replace(final)
        else:
            raw.unlink(missing_ok=True)
    else:
        raw.replace(final)
    if best[1] is not None:
        cv2.imwrite(str(out_dir / "best.jpg"), best[1])
    return {
        "pothole_found": pothole_frames >= MIN_FRAMES,
        "pothole_frames": pothole_frames, "crack_frames": crack_frames, "sampled_frames": sampled,
        "duration_s": round(i / fps, 1), "has_best": best[1] is not None,
    }


PAGE = """<!doctype html><meta charset="utf-8"><meta name="color-scheme" content="light">
<title>Pothole Detector</title>
<style>
html{background:#fff;color-scheme:light}
body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:2rem 1rem;color:#222;background:#fff;min-height:100vh;box-sizing:border-box}
h1{margin-bottom:.2rem;color:#111} .sub{color:#555;margin-top:0}
.box{border:2px dashed #999;border-radius:12px;padding:2rem;text-align:center;background:#fafafa}
button{background:#0a6cff;color:#fff;border:0;border-radius:8px;padding:.7rem 1.4rem;font-size:1rem;cursor:pointer}
button:disabled{background:#999} #res{margin-top:1.5rem} video,img{max-width:100%;border-radius:8px}
.v{padding:1rem;border-radius:10px;font-size:1.3rem;font-weight:600;margin:1rem 0}
.yes{background:#ffe3e3;color:#b00020}.no{background:#e3f6e8;color:#1a7f37} .err{color:#b00020}
.note{color:#666;font-size:.85rem}
</style>
<h1>Pothole &amp; Road Damage Detector</h1>
<p class="sub">Upload a road video; a CNN checks frames for potholes and cracks.</p>
<div class="box"><input type="file" id="f" accept="video/*"><br><br>
<button id="b">Analyse video</button><div id="s"></div></div>
<div id="res"></div>
<p class="note">Decision support only: results need human review. Red = pothole, orange = crack.</p>
<script>
const b=document.getElementById('b'),s=document.getElementById('s'),res=document.getElementById('res');
b.onclick=async()=>{const f=document.getElementById('f').files[0];if(!f){s.textContent='Choose a video first';return}
 b.disabled=true;s.textContent='Analysing... this can take a minute';res.innerHTML='';
 const fd=new FormData();fd.append('video',f);
 try{const r=await fetch('/analyse',{method:'POST',body:fd});const d=await r.json();
  if(!r.ok)throw new Error(d.error||'failed');
  const t=d.pothole_found?'<div class="v yes">Pothole detected</div>':'<div class="v no">No pothole detected</div>';
  res.innerHTML=t+`<p>Frames analysed: ${d.sampled_frames} (${d.duration_s}s video) &middot; with pothole: ${d.pothole_frames} &middot; with crack: ${d.crack_frames}</p>
  <video controls src="/runs/${d.id}/annotated.mp4"></video>`+(d.has_best?`<h3>Strongest pothole frame</h3><img src="/runs/${d.id}/best.jpg">`:'');
  s.textContent=''}catch(e){s.innerHTML='<span class="err">'+e.message+'</span>'}
 b.disabled=false};
</script>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/analyse")
def analyse_route():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No file uploaded"), 400
    if Path(f.filename).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        return jsonify(error="Unsupported file type"), 400
    rid = uuid.uuid4().hex[:12]
    out = RUNS / rid
    out.mkdir()
    src = out / secure_filename(f.filename)
    f.save(src)
    try:
        result = process(src, out)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 422
    finally:
        src.unlink(missing_ok=True)  # do not keep the uploaded original
    return jsonify(id=rid, **result)


@app.get("/runs/<rid>/<name>")
def runs(rid, name):
    if not rid.isalnum() or name not in {"annotated.mp4", "best.jpg"}:
        abort(404)
    return send_from_directory(RUNS / rid, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
