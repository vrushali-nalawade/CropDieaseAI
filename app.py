from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# ======================
# MODEL PATH
# ======================
MODEL_PATH = os.path.join("model", "corn_disease_final_model.h5")

# ======================
# CLASS LABELS
# ======================
CLASS_NAMES = [
    "Leaf Blight",
    "Rust",
    "Healthy",
    "Powdery Mildew"
]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Warming model...")
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy)

print("Model ready 🚀")

# ======================
# TEST ROUTE
# ======================
@app.route("/", methods=["GET"])
def home():
    return "Flask server running"

# ======================
# PREDICTION
# ======================


@app.route("/predict", methods=["POST"])
def predict():
    print("📥 Request received")



    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))

        print("✅ Prediction:", CLASS_NAMES[idx])

        return jsonify({
            "disease": CLASS_NAMES[idx],
            "confidence": float(preds[idx])
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=True
    )
