
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ---------------- SETTINGS ----------------

DATASET_DIR = Path(
    r"C:\Users\pooji\Downloads\archive (4)\real_vs_fake\real-vs-fake"
)

MODEL_PATH = Path(
    r"E:\VeraFace\backend\weights\efficientnet_b0_finetuned.pth"
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
IMAGES_PER_CLASS = 1000
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------- LOAD MODEL CHECKPOINT ----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

classes = checkpoint["classes"]
class_to_idx = checkpoint["class_to_idx"]
image_size = checkpoint.get("image_size", IMAGE_SIZE)

print("Model:", MODEL_PATH.name)
print("Classes:", classes)
print("Class mapping:", class_to_idx)


# ---------------- LOAD TEST DATA ----------------

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dir = DATASET_DIR / "test"

if not test_dir.exists():
    raise FileNotFoundError(f"Test folder not found: {test_dir}")

test_full = datasets.ImageFolder(
    test_dir,
    transform=test_transform,
)

if test_full.classes != classes:
    raise ValueError(
        f"Class mismatch! Model: {classes}; dataset: {test_full.classes}"
    )

print("Total available test images:", len(test_full))


def balanced_subset(dataset, per_class):
    selected = []

    for class_id in range(len(dataset.classes)):
        indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        generator = torch.Generator().manual_seed(SEED + class_id)
        order = torch.randperm(
            len(indices), generator=generator
        ).tolist()

        selected.extend(
            indices[i]
            for i in order[:min(per_class, len(indices))]
        )

    return Subset(dataset, selected)


test_data = balanced_subset(test_full, IMAGES_PER_CLASS)

print("Test images selected:", len(test_data))

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ---------------- BUILD AND LOAD MODEL ----------------

model = models.efficientnet_b0(weights=None)

number_of_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(
    number_of_features,
    len(classes),
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# ---------------- EVALUATE ----------------

confusion = torch.zeros(
    len(classes), len(classes), dtype=torch.int64
)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item(), predicted.item()] += 1

total = int(confusion.sum().item())
correct = int(confusion.diag().sum().item())
accuracy = 100.0 * correct / max(total, 1)

print("\n========== FINETUNED MODEL TEST RESULTS ==========")
print(f"Test accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {correct}/{total}")

print("\nConfusion matrix")
print("Rows = actual class; columns = predicted class")
print("Class order:", classes)

for class_id, class_name in enumerate(classes):
    row = confusion[class_id]
    class_total = int(row.sum().item())
    class_correct = int(row[class_id].item())
    class_accuracy = (
        100.0 * class_correct / class_total
        if class_total else 0.0
    )

    print(
        f"Actual {class_name}: {row.tolist()} | "
        f"Accuracy: {class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )

print("\nEvaluation completed.")
