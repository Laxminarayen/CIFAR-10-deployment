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
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MAX_PIXELS = 12_000_000

if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
    raise FileNotFoundError(
        "Missing artifacts/model.onnx or artifacts/classes.json. "
        "Run train.ipynb first, then copy its artifacts directory here."
    )

with CLASSES_PATH.open(encoding="utf-8") as f:
    metadata = json.load(f)

# Train.ipynb exports the exact label order and preprocessing configuration.
# Reject stale artifacts (e.g., an older cats/dogs model) rather than silently
# returning mislabeled predictions.
if not isinstance(metadata, dict) or metadata.get("dataset") != "CIFAR-10":
    raise ValueError("Wrong classes.json format. Rerun the updated CIFAR-10 train.ipynb.")
CLASS_NAMES = metadata.get("classes")
if CLASS_NAMES != CIFAR10_CLASSES:
    raise ValueError("Expected the 10 CIFAR-10 classes in classes.json. Retrain and re-export.")
if metadata.get("resize_interpolation") != "bilinear":
    raise ValueError("Only bilinear image resizing is supported.")
MEAN = np.asarray(metadata.get("mean"), dtype=np.float32)
STD = np.asarray(metadata.get("std"), dtype=np.float32)
if MEAN.shape != (3,) or STD.shape != (3,) or not np.all(STD > 0):
    raise ValueError("Invalid RGB normalization values in classes.json.")

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
SESSION = ort.InferenceSession(
    str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
)
model_input = SESSION.get_inputs()[0]
INPUT_NAME = model_input.name
input_dims = model_input.shape
if (len(input_dims) != 4 or input_dims[1] != 3
        or not isinstance(input_dims[2], int)
        or not isinstance(input_dims[3], int)):
    raise ValueError("Expected fixed NCHW ONNX model input, e.g. [1, 3, 128, 128].")
# PIL expects (width, height); automatically use the exported model's size.
IMAGE_SIZE = (input_dims[3], input_dims[2])
if metadata.get("input_size") != [input_dims[2], input_dims[3]]:
    raise ValueError("ONNX input size and classes.json do not match. Copy both from one training run.")

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
  <h1>CIFAR-10 Image Classifier</h1>
  <p class="muted">Classify images into the 10 CIFAR-10 categories: {{ classes|join(', ') }}. Upload a JPG, PNG, or WEBP image (max 5 MB).</p>
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
    return render_template_string(PAGE, classes=CLASS_NAMES, predictions=None, error=None)


@app.post("/")
def predict_html():
    try:
        predictions = predict(request.files.get("image"))
        return render_template_string(PAGE, classes=CLASS_NAMES, predictions=predictions, error=None)
    except ValueError as exc:
        return render_template_string(PAGE, classes=CLASS_NAMES, predictions=None, error=str(exc)), 400


@app.post("/api/predict")
def predict_api():
    try:
        predictions = predict(request.files.get("image"))
        return jsonify({"prediction": predictions[0]["label"], "predictions": predictions})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "dataset": "CIFAR-10",
                    "class_count": len(CLASS_NAMES), "image_size": list(IMAGE_SIZE)})


@app.errorhandler(413)
def too_large(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "File too large (maximum 5 MB)."}), 413
    return render_template_string(PAGE, classes=CLASS_NAMES, predictions=None, error="File too large (max 5 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
