"""Exp3 MAIN app: one Flask page with toggle for Audio / Video / Text.

Feature code lives in separate files:
  audio_features.py, video_features.py, text_features.py
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os

from flask import Flask, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from audio_features import extract_audio_features
from text_features import extract_text_features
from video_features import extract_video_features

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

HTML = """
<!DOCTYPE html><html><head><title>Exp 3: Feature Extraction (Audio/Video/Text)</title>
<style>
body{font-family:'Segoe UI',Tahoma,sans-serif;background:#f4f7f6;margin:0;padding:28px;color:#333}
.container{max-width:1080px;margin:auto}
.card{background:#fff;padding:24px;border-radius:10px;box-shadow:0 4px 10px rgba(0,0,0,.06);margin-bottom:20px;border-top:5px solid #2ecc71}
h1,h2{color:#2c3e50;margin-top:0}
.tabs{display:flex;gap:10px;margin:14px 0}
.tabbtn{padding:10px 22px;border:2px solid #2c3e50;background:#fff;color:#2c3e50;border-radius:8px;cursor:pointer;font-weight:bold}
.tabbtn.active{background:#2c3e50;color:#fff}
.panel{display:none}.panel.active{display:block}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:.88em}
th,td{border:1px solid #eee;padding:9px;text-align:left;vertical-align:top}
th{background:#2c3e50;color:#fff}
tr:nth-child(even){background:#f9f9f9}
button.go{background:#2ecc71;color:#fff;border:none;padding:10px 18px;cursor:pointer;border-radius:5px;font-weight:bold}
audio,video,img.preview{width:100%;margin-top:10px;border-radius:8px}
pre{background:#f7f9f9;border:1px solid #e5e8e8;padding:12px;border-radius:8px;max-height:220px;overflow:auto;white-space:pre-wrap}
code{background:#eef;padding:2px 6px;border-radius:4px}
</style>
<script>
function tab(name){document.querySelectorAll('.tabbtn').forEach(b=>b.classList.toggle('active',b.dataset.t===name));
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='p-'+name));}
window.addEventListener('DOMContentLoaded',()=>tab('{{ atab }}'));
</script></head><body><div class="container">
<div class="card">
<h1>Experiment 3: Feature Extraction over Timestamps</h1>
<p>One page, three input types. Pick a tab, upload a file (it stays embedded in the page),
get <b>&ge;3 features &times; &ge;10 timestamp samples</b> with an application justification.</p>
<div class="tabs">
<button class="tabbtn" data-t="audio" onclick="tab('audio')">Audio</button>
<button class="tabbtn" data-t="video" onclick="tab('video')">Video</button>
<button class="tabbtn" data-t="text" onclick="tab('text')">Text</button>
</div>
<div class="panel" id="p-audio">
<p><b>Objective:</b> podcast speech-activity analysis over time. Features:
<code>RMS</code> (loudness), <code>ZCR</code> (noisiness), <code>Spectral Centroid</code> (brightness).</p>
<form method="POST" enctype="multipart/form-data">
<input type="hidden" name="input_type" value="audio">
<input type="file" name="file" accept=".wav,.mp3,.ogg,.flac,audio/*" required>
<button class="go" type="submit">Extract</button></form></div>
<div class="panel" id="p-video">
<p><b>Objective:</b> lecture-video visual-activity analysis over frames. Features per frame:
<code>Brightness</code>, <code>Contrast</code>, <code>Motion vs prev frame</code> (+ edge, colorfulness).</p>
<form method="POST" enctype="multipart/form-data">
<input type="hidden" name="input_type" value="video">
<input type="file" name="file" accept=".gif,.mp4,.avi,.mov,.mkv,image/gif,video/*" required>
<button class="go" type="submit">Extract</button></form></div>
<div class="panel" id="p-text">
<p><b>Objective:</b> review/transcript sentiment trend over document position. Features per chunk:
<code>Word count</code>, <code>Avg word length</code>, <code>Sentiment polarity</code>.</p>
<form method="POST" enctype="multipart/form-data">
<input type="hidden" name="input_type" value="text">
<input type="file" name="file" accept=".txt,text/plain" required>
<button class="go" type="submit">Extract</button></form></div>
</div>
{% if error %}<div class="card" style="border-top-color:#e74c3c"><b style="color:#e74c3c">{{ error }}</b></div>{% endif %}
{% if audio %}
<div class="card" style="border-top-color:#3498db">
<h2>Audio file (embedded) + {{ audio.rows|length }} timestamp samples</h2>
<p><b>{{ afname }}</b> | {{ audio.sr }} Hz | {{ audio.duration }} s | window {{ audio.window }} s</p>
<audio controls src="{{ audiourl }}"></audio>
<table><tr><th>#</th><th>Timestamp</th><th>RMS</th><th>ZCR</th><th>Centroid (Hz)</th></tr>
{% for r in audio.rows %}<tr><td>{{ loop.index }}</td><td>{{ r.timestamp }}</td><td>{{ r.rms }}</td>
<td>{{ r.zcr }}</td><td>{{ r.centroid }}</td></tr>{% endfor %}</table>
</div>{% endif %}
{% if video %}
<div class="card" style="border-top-color:#9b59b6">
<h2>Video file (embedded) + {{ video.rows|length }} frame samples</h2>
<p><b>{{ vfname }}</b> | {{ video.source }} | total frames {{ video.n_frames_total }} | ~{{ video.duration }} s</p>
{% if vfname.endswith('.gif') %}<img class="preview" src="{{ videourl }}">{% else %}<video class="preview" controls preload="metadata"><source src="{{ videourl }}" type="video/mp4"></video>{% endif %}
<table><tr><th>#</th><th>Frame / Time</th><th>Brightness</th><th>Contrast</th><th>Motion</th><th>Edge</th><th>Color</th></tr>
{% for r in video.rows %}<tr><td>{{ loop.index }}</td><td>{{ r.frame }} / {{ r.timestamp }}</td><td>{{ r.brightness }}</td>
<td>{{ r.contrast }}</td><td>{{ r.motion }}</td><td>{{ r.edge }}</td><td>{{ r.colorfulness }}</td></tr>{% endfor %}</table>
</div>{% endif %}
{% if text %}
<div class="card" style="border-top-color:#e67e22">
<h2>Text file (embedded) + {{ text.rows|length }} samples ({{ text.mode }})</h2>
<p><b>{{ tfname }}</b> | {{ text.n_chars }} chars</p>
<pre>{{ text.full_text[:2000] }}</pre>
<table><tr><th>#</th><th>Sample</th><th>Words</th><th>Avg len</th><th>Polarity</th><th>Preview</th></tr>
{% for r in text.rows %}<tr><td>{{ loop.index }}</td><td>{{ r.sample }}</td><td>{{ r.words }}</td>
<td>{{ r.avg_len }}</td><td>{{ r.polarity }}</td><td>{{ r.preview }}</td></tr>{% endfor %}</table>
</div>{% endif %}
</div></body></html>
"""


@app.route("/uploads/<path:f>")
def up(f):
    return send_from_directory(UPLOAD_DIR, f)


@app.route("/", methods=["GET", "POST"])
def index():
    atab, error = "audio", None
    audio = video = text = None
    afname = vfname = tfname = audiourl = videourl = None
    if request.method == "POST":
        kind = request.form.get("input_type", "audio")
        atab = kind if kind in ("audio", "video", "text") else "audio"
        try:
            if kind == "audio":
                f = request.files.get("file")
                if not f or not f.filename:
                    raise ValueError("No audio file selected.")
                afname = secure_filename(f.filename)
                f.save(os.path.join(UPLOAD_DIR, afname))
                audio = extract_audio_features(os.path.join(UPLOAD_DIR, afname))
                audiourl = url_for("up", f=afname)
                if len(audio["rows"]) < 10:
                    error = f"Only {len(audio['rows'])} samples - upload a longer WAV for >=10."
            elif kind == "video":
                f = request.files.get("file")
                if not f or not f.filename:
                    raise ValueError("No video file selected.")
                vfname = secure_filename(f.filename)
                f.save(os.path.join(UPLOAD_DIR, vfname))
                video = extract_video_features(os.path.join(UPLOAD_DIR, vfname))
                videourl = url_for("up", f=vfname)
                if len(video["rows"]) < 10:
                    error = f"Only {len(video['rows'])} frames - upload a longer clip for >=10."
            else:
                f = request.files.get("file")
                if not f or not f.filename:
                    raise ValueError("No text file selected.")
                tfname = secure_filename(f.filename)
                f.save(os.path.join(UPLOAD_DIR, tfname))
                text = extract_text_features(os.path.join(UPLOAD_DIR, tfname))
                if len(text["rows"]) < 10:
                    error = f"Only {len(text['rows'])} samples - upload a longer document for >=10."
        except Exception as e:
            error = f"Failed: {e}"
    return render_template_string(HTML, atab=atab, error=error,
                                  audio=audio, afname=afname, audiourl=audiourl,
                                  video=video, vfname=vfname, videourl=videourl,
                                  text=text, tfname=tfname)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(debug=True)
