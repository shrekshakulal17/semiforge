import os
import sys
import io
import time
import base64

from flask import Flask, request, jsonify
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

MODEL_LOADED = False

if os.path.exists(MODEL_PATH):

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Supports a normal state_dict
    # and a checkpoint dictionary
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    MODEL_LOADED = True

    print("Model loaded:", MODEL_PATH)

else:

    print("WARNING: best_model.pth not found.")
    print("Restoration endpoint will not work until the model exists.")


model.eval()


# ============================================================
# RESTORE ENDPOINT
# ============================================================

@app.route("/restore", methods=["POST"])
def restore():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_LOADED:

        return jsonify({
            "error": "Trained model not found.",
            "model_path": MODEL_PATH
        }), 503


    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        original_image = Image.open(
            file.stream
        ).convert("RGB")


        original_width, original_height = (
            original_image.size
        )


        input_resolution = (
            f"{original_width} × {original_height}"
        )


        # ----------------------------------------------------
        # Resize for model
        # ----------------------------------------------------

        model_input = original_image.resize(
            (256, 256),
            Image.Resampling.BICUBIC
        )


        # ----------------------------------------------------
        # PIL → Tensor
        # ----------------------------------------------------

        tensor = TF.to_tensor(
            model_input
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)


        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()


        with torch.no_grad():

            output = model(tensor)


        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()


        inference_time = (
            end_time - start_time
        ) * 1000


        # ----------------------------------------------------
        # Tensor → PIL
        # ----------------------------------------------------

        output = output.squeeze(0).cpu()

        output = torch.clamp(
            output,
            0.0,
            1.0
        )


        output_image = TF.to_pil_image(
            output
        )


        # ----------------------------------------------------
        # Output resolution
        # ----------------------------------------------------

        output_width, output_height = (
            output_image.size
        )


        output_resolution = (
            f"{output_width} × {output_height}"
        )


        # ----------------------------------------------------
        # Convert restored image to Base64
        # ----------------------------------------------------

        image_bytes = io.BytesIO()

        output_image.save(
            image_bytes,
            format="PNG"
        )

        image_bytes.seek(0)


        encoded_image = base64.b64encode(
            image_bytes.read()
        ).decode("utf-8")


        restored_image = (
            "data:image/png;base64,"
            + encoded_image
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "restoredImage": restored_image,

            "psnr": None,
            "ssim": None,
            "lpips": None,

            "inferenceTime": round(
                inference_time,
                2
            ),

            "inputResolution":
                input_resolution,

            "modelInputResolution":
                "256 × 256",

            "outputResolution":
                output_resolution,

            "detectedDegradation": [
                "Reduced spatial resolution"
            ],

            "device":
                str(device),

            "model":
                "RestorationModel"

        })


    except Exception as e:

        print(
            "RESTORE ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "ok",

        "modelLoaded":
            MODEL_LOADED,

        "device":
            str(device),

        "model":
            "RestorationModel"

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