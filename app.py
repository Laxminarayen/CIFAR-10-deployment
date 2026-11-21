"""Minimal image-classification Flask server using a trained ONNX model."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "model.onnx"
CLASSES_PATH = MODEL_DIR / "classes.json"
IMAGE_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PIXELS = 12_000_000

if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
    raise FileNotFoundError(
        "Missing artifacts/model.onnx or artifacts/classes.json. "
        "Run train.ipynb first, then copy its artifacts directory here."
    )

with CLASSES_PATH.open(encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
SESSION = ort.InferenceSession(
    str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
)
INPUT_NAME = SESSION.get_inputs()[0].name

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB uploads

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #f6f7fb; margin: 0; }
    main { background: white; max-width: 480px; margin: 8vh auto; padding: 28px;
           border-radius: 12px; box-shadow: 0 6px 24px #0001; }
    h1 { margin-top: 0; }
    input { margin: 12px 0 20px; max-width: 100%; }
    button { background: #155eef; color: white; border: 0; padding: 12px 18px;
             border-radius: 7px; cursor: pointer; }
    .error { color: #b42318; }
    .result { margin-top: 24px; padding-top: 16px; border-top: 1px solid #ddd; }
    li { margin: 8px 0; }
    .muted { color: #667085; }
  </style>
</head>
<body><main>
  <h1>Image Classifier</h1>
  <p class="muted">Upload a JPG, PNG, or WEBP image (max 5 MB).</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/jpeg,image/png,image/webp" required>
    <div><button type="submit">Classify image</button></div>
  </form>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  {% if predictions %}
    <div class="result">
      <h2>Prediction: {{ predictions[0].label }}</h2>
      <p>{{ predictions[0].confidence }}% confidence</p>
      <h3>Top predictions</h3>
      <ol>{% for item in predictions %}
        <li>{{ item.label }} — {{ item.confidence }}%</li>
      {% endfor %}</ol>
    </div>
  {% endif %}
</main></body>
</html>
"""


def predict(file_storage):
    """Return up to 3 predictions for an uploaded image (without saving it)."""
    if file_storage is None or not file_storage.filename:
        raise ValueError("Please choose an image file.")

    try:
        with Image.open(file_storage.stream) as raw:
            if raw.width * raw.height > MAX_PIXELS:
                raise ValueError("Image is too large in pixels (12 MP maximum).")
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsupported image file.") from exc

    image = image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    batch = np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)

    logits = SESSION.run(None, {INPUT_NAME: batch})[0][0]
    if len(logits) != len(CLASS_NAMES):
        raise RuntimeError("The ONNX model and classes.json have different class counts.")
    scores = np.exp(logits - np.max(logits))
    scores /= np.sum(scores)
    best = np.argsort(scores)[::-1][:3]
    return [
        {"label": CLASS_NAMES[int(i)], "confidence": round(float(scores[i] * 100), 2)}
        for i in best
    ]


@app.get("/")
def index():
    return render_template_string(PAGE, predictions=None, error=None)


@app.post("/")
def predict_html():
    try:
        predictions = predict(request.files.get("image"))
        return render_template_string(PAGE, predictions=predictions, error=None)
    except ValueError as exc:
        return render_template_string(PAGE, predictions=None, error=str(exc)), 400


@app.post("/api/predict")
def predict_api():
    try:
        predictions = predict(request.files.get("image"))
        return jsonify({"prediction": predictions[0]["label"], "predictions": predictions})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "class_count": len(CLASS_NAMES)})


@app.errorhandler(413)
def too_large(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "File too large (maximum 5 MB)."}), 413
    return render_template_string(PAGE, predictions=None, error="File too large (max 5 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
