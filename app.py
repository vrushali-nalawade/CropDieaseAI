# ======================
# MEMORY + PERFORMANCE FIXES (VERY IMPORTANT)
# ======================
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# ======================
# IMPORTS
# ======================
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ======================
# FLASK SETUP
# ======================
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

# ======================
# MODEL PATH
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "corn_disease_final_model.h5")

# ======================
# CLASS LABELS
# ======================
CLASS_NAMES = [
    "Leaf Blight",
    "Rust",
    "Healthy",
    "Powdery Mildew"
]

# ======================
# LAZY MODEL LOADING (CRITICAL)
# ======================
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully 🚀")

# ======================
# TEST ROUTE
# ======================
@app.route("/", methods=["GET"])
def home():
    return "Crop Disease Detection API Running 🚜🌱"

# ======================
# PREDICTION ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():

    load_model()  # load only once

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))

        return jsonify({
            "disease": CLASS_NAMES[idx],
            "confidence": float(preds[idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

