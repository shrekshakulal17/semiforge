import os
import sys
import io

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from PIL import Image
import torch
import torchvision.transforms.functional as TF


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# MODEL
# ============================================================

from models.restoration import RestorationModel


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

model = RestorationModel().to(device)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "best_model.pth"
)

if os.path.exists(MODEL_PATH):

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print("Model loaded:", MODEL_PATH)

else:

    print("WARNING: best_model.pth not found.")
    print("The API will still start, but restoration will not work yet.")


model.eval()


# ============================================================
# RESTORE ENDPOINT
# ============================================================

@app.route("/restore", methods=["POST"])
def restore():

    # Check image
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # Open uploaded image
        image = Image.open(
            file.stream
        ).convert("RGB")

        # ----------------------------------------------------
        # Resize input to 256x256 for our current model
        # ----------------------------------------------------

        image = image.resize(
            (256, 256),
            Image.Resampling.BICUBIC
        )

        # PIL → Tensor
        tensor = TF.to_tensor(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)


        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(tensor)


        # ----------------------------------------------------
        # Tensor → PIL
        # ----------------------------------------------------

        output = output.squeeze(0).cpu()

        output = torch.clamp(
            output,
            0,
            1
        )

        output_image = TF.to_pil_image(
            output
        )


        # ----------------------------------------------------
        # Save image in memory
        # ----------------------------------------------------

        image_bytes = io.BytesIO()

        output_image.save(
            image_bytes,
            format="PNG"
        )

        image_bytes.seek(0)


        return send_file(
            image_bytes,
            mimetype="image/png"
        )


    except Exception as e:

        print("RESTORE ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "device": str(device)
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )