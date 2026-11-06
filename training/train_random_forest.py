import os
import cv2
import joblib
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(r"C:\Users\pooji\Downloads\archive (4)\real_vs_fake\real-vs-fake")

MODEL_DIR = Path(r"E:\VeraFace\backend\weights")
MODEL_PATH = MODEL_DIR / "random_forest.pkl"

IMAGE_SIZE = (32, 32)

# Number of trees in the Random Forest
N_ESTIMATORS = 100


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Convert one RGB image into a compact numerical feature vector.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # OpenCV loads images as BGR.
    # Convert to RGB for color statistics.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # 1. Resize image
    # --------------------------------------------------------

    resized = cv2.resize(image_rgb, IMAGE_SIZE)

    # --------------------------------------------------------
    # 2. Grayscale features
    # --------------------------------------------------------

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Normalize pixel values to 0-1
    gray_normalized = gray.astype(np.float32) / 255.0

    # Flatten the resized grayscale image
    pixel_features = gray_normalized.flatten()

    # --------------------------------------------------------
    # 3. Brightness and contrast
    # --------------------------------------------------------

    brightness = np.mean(gray_normalized)
    contrast = np.std(gray_normalized)

    # --------------------------------------------------------
    # 4. Sharpness
    # --------------------------------------------------------

    sharpness = cv2.Laplacian(gray_normalized, cv2.CV_32F).var()

    # --------------------------------------------------------
    # 5. Edge density
    # --------------------------------------------------------

    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.mean(edges > 0)

    # --------------------------------------------------------
    # 6. Color statistics
    # --------------------------------------------------------

    color_features = []

    for channel in range(3):
        channel_data = resized[:, :, channel].astype(np.float32) / 255.0

        color_features.append(np.mean(channel_data))
        color_features.append(np.std(channel_data))
        color_features.append(np.min(channel_data))
        color_features.append(np.max(channel_data))

    color_features = np.array(color_features, dtype=np.float32)

    # --------------------------------------------------------
    # 7. Combine everything
    # --------------------------------------------------------

    extra_features = np.array(
        [
            brightness,
            contrast,
            sharpness,
            edge_density,
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            pixel_features,
            extra_features,
            color_features,
        ]
    )

    return features.astype(np.float32)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split):
    """
    Load images from:

        split/real
        split/fake

    Returns:
        X = feature matrix
        y = labels
    """

    split_path = DATASET / split

    real_dir = split_path / "real"
    fake_dir = split_path / "fake"

    if not real_dir.exists():
        raise FileNotFoundError(f"Missing directory: {real_dir}")

    if not fake_dir.exists():
        raise FileNotFoundError(f"Missing directory: {fake_dir}")

    X = []
    y = []

    # --------------------------------------------------------
    # REAL = 1
    # --------------------------------------------------------

    real_files = sorted(real_dir.glob("*.jpg"))

    print(f"\nLoading REAL images from {split}...")
    print(f"Found: {len(real_files)}")

    for index, image_path in enumerate(real_files, start=1):

        try:
            features = extract_features(image_path)

            X.append(features)
            y.append(1)

        except Exception as error:
            print(f"Error: {image_path}")
            print(error)

        if index % 5000 == 0:
            print(f"Processed REAL: {index}/{len(real_files)}")

    # --------------------------------------------------------
    # FAKE = 0
    # --------------------------------------------------------

    fake_files = sorted(fake_dir.glob("*.jpg"))

    print(f"\nLoading FAKE images from {split}...")
    print(f"Found: {len(fake_files)}")

    for index, image_path in enumerate(fake_files, start=1):

        try:
            features = extract_features(image_path)

            X.append(features)
            y.append(0)

        except Exception as error:
            print(f"Error: {image_path}")
            print(error)

        if index % 5000 == 0:
            print(f"Processed FAKE: {index}/{len(fake_files)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("VERAFACE - RANDOM FOREST TRAINING")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET)

    print(f"\nFeature image size:")
    print(IMAGE_SIZE)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train, y_train = load_dataset("train")

    print("\nTraining feature matrix:")
    print("Shape:", X_train.shape)

    print("\nTraining labels:")
    print("REAL:", np.sum(y_train == 1))
    print("FAKE:", np.sum(y_train == 0))

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    X_valid, y_valid = load_dataset("valid")

    print("\nValidation feature matrix:")
    print("Shape:", X_valid.shape)

    print("\nValidation labels:")
    print("REAL:", np.sum(y_valid == 1))
    print("FAKE:", np.sum(y_valid == 0))

    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print(f"\nNumber of trees: {N_ESTIMATORS}")
    print("Training started...")

    model.fit(X_train, y_train)

    print("\nTraining completed.")

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_valid,
            y_pred,
            target_names=["FAKE", "REAL"],
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"\nModel path:")
    print(MODEL_PATH)

    print("\nRandom Forest training pipeline completed successfully.")


if __name__ == "__main__":
    main()