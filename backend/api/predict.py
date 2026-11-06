
from pathlib import Path
import uuid

import cv2
import numpy as np
import torch

from torch import nn
from torchvision import models, transforms

from PIL import Image, ImageOps, UnidentifiedImageError

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "weights" / "efficientnet_b0_finetuned.pth"
UPLOAD_FOLDER = BASE_DIR / "uploads"

DEVICE = torch.device("cpu")

DEFAULT_IMAGE_SIZE = 224

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MAX_IMAGE_SIZE = 10 * 1024 * 1024

CONFIDENCE_THRESHOLD = 0.70

# IMPORTANT: Blueprint declaration
predict_bp = Blueprint("predict", __name__)


# =====================================================
# FACE DETECTOR
# =====================================================

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Unable to load face detector")


class FaceNotFoundError(Exception):
    pass


# =====================================================
# LOAD EFFICIENTNET MODEL
# =====================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

CLASSES = checkpoint["classes"]

CLASS_TO_IDX = checkpoint["class_to_idx"]

IMAGE_SIZE = int(
    checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)
)

if set(CLASSES) != {"fake", "real"}:
    raise ValueError(
        f"Incorrect model classes: {CLASSES}"
    )

if (
    CLASS_TO_IDX.get("fake") != 0
    or CLASS_TO_IDX.get("real") != 1
):
    raise ValueError(
        f"Incorrect class mapping: {CLASS_TO_IDX}"
    )


model = models.efficientnet_b0(weights=None)

number_of_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    number_of_features,
    len(CLASSES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("\n================================")
print("VERAFACE EFFICIENTNET MODEL")
print("================================")

print("Model loaded:", MODEL_PATH)
print("Classes:", CLASSES)
print("Class mapping:", CLASS_TO_IDX)
print("Image size:", IMAGE_SIZE)
print("Face detector: OpenCV")

print("================================\n")


# =====================================================
# IMAGE PREPROCESSING
# SAME AS ORIGINAL TRAINING PIPELINE
# =====================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =====================================================
# FACE DETECTION
# =====================================================

def detect_and_crop_face(image):

    image_array = np.array(image)

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:

        raise FaceNotFoundError(
            "No face detected. Please upload a clear face image."
        )

    # Select largest detected face.
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Calculate face crop.
    padding_x = int(width * 0.20)
    padding_y = int(height * 0.20)

    left = max(0, x - padding_x)
    top = max(0, y - padding_y)

    right = min(
        image.width,
        x + width + padding_x
    )

    bottom = min(
        image.height,
        y + height + padding_y
    )

    cropped_face = image.crop(
        (left, top, right, bottom)
    )

    return cropped_face, len(faces)


# =====================================================
# IMAGE PREDICTION
# =====================================================

def predict_image(image_path):

    # Load and correct image orientation.
    with Image.open(image_path) as source:
        source.verify()

    with Image.open(image_path) as source:

        image = ImageOps.exif_transpose(source)

        image = image.convert("RGB")

        image.load()

    original_dimensions = image.size

    # Detect faces.
    # Face detection is separate from classification.
    _, face_count = detect_and_crop_face(image)

    # Use the complete image for classification.
    # This matches the original training input more closely.
    image_tensor = image_transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # Run EfficientNet.
    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    fake_probability = float(
        probabilities[CLASS_TO_IDX["fake"]].item()
    )

    real_probability = float(
        probabilities[CLASS_TO_IDX["real"]].item()
    )

    confidence = max(
        fake_probability,
        real_probability
    )

    # Avoid presenting weak predictions as certain.
    if confidence < CONFIDENCE_THRESHOLD:

        prediction = "INCONCLUSIVE"

    elif fake_probability > real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    print("\n========== VERAFACE PREDICTION ==========")

    print("Original dimensions:", original_dimensions)

    print("Detected faces:", face_count)

    print("Face crop used for classification: False")

    print(
        f"FAKE probability: {fake_probability * 100:.2f}%"
    )

    print(
        f"REAL probability: {real_probability * 100:.2f}%"
    )

    print("Final result:", prediction)

    print(
        f"Model confidence: {confidence * 100:.2f}%"
    )

    print("=========================================\n")

    return {

        "prediction": prediction,

        "confidence": confidence,

        "real_probability": real_probability,

        "fake_probability": fake_probability,

        "face_detected": True,

        "face_count": face_count,

        "face_crop_used": False

    }


# =====================================================
# FILE VALIDATION
# =====================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =====================================================
# API ROUTE
# =====================================================

@predict_bp.route(
    "/api/predict",
    methods=["POST"]
)

def predict():

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "message": "No image was uploaded."

        }), 400

    file = request.files["image"]

    if not file.filename:

        return jsonify({

            "success": False,

            "message": "Please select an image."

        }), 400

    if not allowed_file(file.filename):

        return jsonify({

            "success": False,

            "message": "Only JPG, JPEG and PNG images are allowed."

        }), 400

    file_path = None

    try:

        # Check file size.
        file.seek(0, 2)

        file_size = file.tell()

        file.seek(0)

        if file_size == 0:

            return jsonify({

                "success": False,

                "message": "The uploaded image is empty."

            }), 400

        if file_size > MAX_IMAGE_SIZE:

            return jsonify({

                "success": False,

                "message": "Image size must be below 10 MB."

            }), 400

        # Create upload folder.
        UPLOAD_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = secure_filename(
            file.filename
        )

        if not filename:

            return jsonify({

                "success": False,

                "message": "Invalid filename."

            }), 400

        unique_filename = (
            uuid.uuid4().hex + "_" + filename
        )

        file_path = UPLOAD_FOLDER / unique_filename

        file.save(file_path)

        # Run prediction.
        result = predict_image(file_path)

        return jsonify({

            "success": True,

            "prediction": result["prediction"],

            "confidence": round(
                result["confidence"] * 100, 2
            ),

            "real_probability": round(
                result["real_probability"] * 100, 2
            ),

            "fake_probability": round(
                result["fake_probability"] * 100, 2
            ),

            "trust_score": round(
                result["confidence"] * 100, 2
            ),

            "face_detected": result["face_detected"],

            "face_count": result["face_count"],

            "face_crop_used": result["face_crop_used"],

            "message": (
                "Image analysis completed. "
                "This AI prediction is an estimate, "
                "not proof of authenticity."
            )

        })

    except FaceNotFoundError as error:

        return jsonify({

            "success": False,

            "face_detected": False,

            "message": str(error)

        }), 422

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        print(
            "Image processing error:",
            str(error)
        )

        return jsonify({

            "success": False,

            "message": (
                "Unable to process this image. "
                "Please upload a valid JPG or PNG image."
            )

        }), 400

    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "message": "Prediction failed. Check Flask terminal."

        }), 500

    finally:

        # Delete temporary uploaded image.
        if file_path is not None and file_path.exists():

            try:

                file_path.unlink()

            except OSError as error:

                print(
                    "File cleanup error:",
                    str(error)
                )
