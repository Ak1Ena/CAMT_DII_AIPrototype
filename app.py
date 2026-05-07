"""
Flask web app that serves the EfficientNet-B0 classifier.

Run:
    export MODEL_PATH=/path/to/efficientnet_b0_YYYYMMDD_HHMMSS.pth
    python app.py

Then open http://localhost:5000 in a browser.
"""

import os
import io
import logging
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError

from model_utils import ImageClassifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "model.pth"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "gif"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("classifier-app")


# ---------------------------------------------------------------------------
# Lazy classifier loader (so the app can boot even if the model is missing,
# and we can show a clear error in the UI instead of crashing)
# ---------------------------------------------------------------------------
_classifier: ImageClassifier | None = None
_classifier_error: str | None = None


def get_classifier() -> ImageClassifier | None:
    global _classifier, _classifier_error
    if _classifier is not None:
        return _classifier
    try:
        log.info(f"Loading model from {MODEL_PATH} ...")
        _classifier = ImageClassifier(MODEL_PATH)
        log.info(
            f"Model ready. {_classifier.num_classes} classes: "
            f"{_classifier.class_names}. Device: {_classifier.device}."
        )
        _classifier_error = None
    except Exception as exc:  # noqa: BLE001
        _classifier_error = str(exc)
        log.error(f"Failed to load model: {exc}")
    return _classifier


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.route("/")
def index():
    clf = get_classifier()
    return render_template(
        "index.html",
        model_loaded=clf is not None,
        model_error=_classifier_error,
        class_names=clf.class_names if clf else [],
        num_classes=clf.num_classes if clf else 0,
        best_val_acc=clf.best_val_acc if clf else None,
        device=str(clf.device) if clf else None,
    )


@app.route("/api/info")
def api_info():
    clf = get_classifier()
    if clf is None:
        return jsonify({"loaded": False, "error": _classifier_error}), 503
    return jsonify({
        "loaded": True,
        "classes": clf.class_names,
        "num_classes": clf.num_classes,
        "img_size": clf.img_size,
        "best_val_acc": clf.best_val_acc,
        "device": str(clf.device),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    clf = get_classifier()
    if clf is None:
        return jsonify({"error": f"Model not loaded: {_classifier_error}"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image' in the request."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        raw = file.read()
        image = Image.open(io.BytesIO(raw))
        results = clf.predict(image, top_k=min(5, clf.num_classes))
    except UnidentifiedImageError:
        return jsonify({"error": "Could not decode image. Is it a valid image file?"}), 400
    except Exception as exc:  # noqa: BLE001
        log.error("Prediction failed:\n" + traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "predicted_class": results[0]["class"],
        "confidence": results[0]["probability"],
        "top_predictions": results,
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "error": f"File too large. Max size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
    }), 413


if __name__ == "__main__":
    # Trigger a load attempt at startup so the first request feels instant
    get_classifier()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
