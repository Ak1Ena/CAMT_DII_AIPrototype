from flask import Flask, flash, redirect, request, render_template, make_response, url_for, jsonify
import json
import io
import base64
import os
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        _model_data = pickle.load(f)
    pipeline  = _model_data['pipeline']
    CLASSES   = _model_data['classes']          # ["chihuahua", "muffin"]
    IMG_SIZE  = _model_data['img_size']         # (128, 128)
    print(f"[✓] Model loaded — classes: {CLASSES}, input size: {IMG_SIZE}")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    pipeline = None

# ── Feature extraction functions ──────────────────────────────────────────────

def hog_features(img_arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).convert("L")
    arr = np.array(img, dtype=np.float32)
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi % 180
    cell = 16
    feats = []
    for r in range(0, arr.shape[0], cell):
        for c in range(0, arr.shape[1], cell):
            m = mag[r:r+cell, c:c+cell]
            a = ang[r:r+cell, c:c+cell]
            hist, _ = np.histogram(a, bins=9, range=(0, 180), weights=m)
            feats.extend(hist)
    return np.array(feats)

def color_histogram(img_arr: np.ndarray, bins: int = 32) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_arr[:, :, ch], bins=bins, range=(0, 1))
        feats.extend(hist / (hist.sum() + 1e-8))
    return np.array(feats)

def extract_features(img_arr: np.ndarray) -> np.ndarray:
    return np.concatenate([hog_features(img_arr), color_histogram(img_arr)])

def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0

def predict(pil_image: Image.Image):
    if pipeline is None:
        return "Error", 0, {}
    arr   = preprocess(pil_image)
    feats = extract_features(arr).reshape(1, -1)
    proba = pipeline.predict_proba(feats)[0]
    idx   = int(np.argmax(proba))
    label = CLASSES[idx].capitalize()
    conf  = round(float(proba[idx]) * 100, 1)
    return label, conf, {c.capitalize(): round(float(p) * 100, 1)
                         for c, p in zip(CLASSES, proba)}

# ── Existing Routes ────────────────────────────────────────────────────
@app.route("/") 
def helloworld():
    return "Hello, World!"

@app.route("/name") 
def name():
    return "Siraphop Khatchamat ID672110163"


@app.route('/receive_get',methods=['GET']) 
def web_service_API_GET():

    msg = request.args.get('msg')
    name = request.args.get('name')
    
    print(f'the input message from GET is {msg} from {name}.')
    
    return f'{msg} from {name} received!'


@app.route('/request_POST',methods=['POST']) 
def web_service_API_POST():
        payload = request.data.decode("utf-8")
        inmessage = json.loads(payload) # ทำ json
        print(inmessage)
        json_data = json.dumps({'y': 'POST received!'}) # ส่งกลับไปว่าได้รับเเล้ววว
        return json_data
    

@app.route("/resume")
def resume():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume - [Your Full Name]</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            color: #2c3e50;
        }
        .contact-info {
            font-size: 0.95em;
            color: #555;
        }
        .contact-info a {
            color: #2980b9;
            text-decoration: none;
        }
        .contact-info a:hover {
            text-decoration: underline;
        }
        section {
            margin-bottom: 25px;
        }
        h2 {
            font-size: 1.4em;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .item-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 5px;
        }
        .item-title {
            font-weight: bold;
            font-size: 1.1em;
        }
        .item-subtitle {
            font-style: italic;
            color: #555;
        }
        .item-date {
            font-size: 0.9em;
            color: #666;
            text-align: right;
        }
        ul {
            margin-top: 0;
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        .skills-category {
            font-weight: bold;
        }
    </style>
</head>
<body>

    <header>
        <h1>Siraphop Khatchamat</h1>
        <div class="contact-info">
            Chiang Mai, Thailand &nbsp;|&nbsp; 
            [Phone Number] &nbsp;|&nbsp; 
            <a href="mailto:your.email@example.com">[Email Address]</a> &nbsp;|&nbsp; 
            <a href="https://github.com/yourusername" target="_blank">GitHub</a> &nbsp;|&nbsp; 
            <a href="https://linkedin.com/in/yourusername" target="_blank">LinkedIn</a>
        </div>
    </header>

    <section>
        <h2>Professional Summary</h2>
        <p>[Optional: Detail-oriented software developer with hands-on experience in full-stack web development, workflow automation, and local AI deployment. Passionate about building efficient tools and optimizing systems in Arch Linux environments.]</p>
    </section>

    <section>
        <h2>Education</h2>
        <div class="item-header">
            <div>
                <span class="item-title">Chiang Mai University (CMU)</span> | Chiang Mai, Thailand
                <div class="item-subtitle">CAMT, CMU</div>
            </div>
        </div>
    </section>

    <section>
        <h2>Professional Experience</h2>
        <div class="item-header">
            <div>
                <span class="item-title">Appman</span> 
                <div class="item-subtitle">Software Developer Intern</div>
            </div>
            <div class="item-date">1 Dec 2025 – Present</div>
        </div>
    </section>

    <section>
        <h2>Technical Projects</h2>
        
        <div class="item-header">
            <span class="item-title">Workflow Automation & Logic Design</span>
        </div>
        <ul>
            <li>Designed and maintained complex automation workflows using <strong>n8n</strong>, streamlining data processing and internal tasks.</li>
            <li>Developed custom TypeScript clustering algorithms and integrated React-based web SDKs to enhance application functionality.</li>
        </ul>

        <div class="item-header">
            <span class="item-title">Local AI & Audio Deployment</span>
        </div>
        <ul>
            <li>Successfully deployed and ran local Large Language Models (LLMs) using <strong>Ollama</strong> on CPU-limited hardware.</li>
            <li>Integrated Text-to-Speech (TTS) models, specifically <strong>Kokoro-82M (ONNX)</strong>, with Python scripts to enable real-time voice cloning and interaction.</li>
        </ul>
    </section>

    <section>
        <h2>Technical Skills</h2>
        <ul>
            <li><span class="skills-category">Programming Languages:</span> TypeScript, JavaScript, Python</li>
            <li><span class="skills-category">Frontend & Backend:</span> React, Node.js</li>
            <li><span class="skills-category">Tools & Infrastructure:</span> n8n, Git / GitHub, Arch Linux (pacman/yay), Google Chrome dev tools</li>
            <li><span class="skills-category">AI & Machine Learning:</span> Local LLM deployment (Ollama), ONNX models, TTS generation</li>
        </ul>
    </section>

</body>
</html>
    """
@app.route("/classify")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        raw   = file.read()
        image = Image.open(io.BytesIO(raw))

        # Thumbnail for preview
        thumb = image.copy()
        thumb.thumbnail((420, 420))
        buf = io.BytesIO()
        thumb.convert("RGB").save(buf, format="JPEG", quality=85)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        label, confidence, all_proba = predict(image)

        return jsonify({
            "class":      label,
            "confidence": confidence,
            "all_proba":  all_proba,
            "thumbnail":  thumb_b64,
        })

    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":   # run code 
    app.run(host='0.0.0.0',debug=True,port=5002)#host='0.0.0.0' = run on internet ,port=5002 (port บน server เหมือนประตู) / localhost รันบนเครื่องเรายังไม่ใช่ internet

