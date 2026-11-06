
from pathlib import Path
import json
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ---------------- SETTINGS ----------------

DATASET_DIR = Path(
    r"C:\Users\pooji\Downloads\archive (4)\real_vs_fake\real-vs-fake"
)

OUTPUT_DIR = Path(r"E:\VeraFace\backend\weights")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "efficientnet_b0_finetuned.pth"
CLASSES_PATH = OUTPUT_DIR / "efficientnet_b0_finetuned_classes.json"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
IMAGES_PER_CLASS = 5000
VALIDATION_IMAGES_PER_CLASS = 1000
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------- CHECK DATASET ----------------

for split in ("train", "valid"):
    split_path = DATASET_DIR / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_path}"
        )


# ---------------- IMAGE TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- LOAD DATA ----------------

train_full = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform,
)

valid_full = datasets.ImageFolder(
    DATASET_DIR / "valid",
    transform=valid_transform,
)

print("Classes:", train_full.classes)
print("Class mapping:", train_full.class_to_idx)
print("Available training images:", len(train_full))
print("Available validation images:", len(valid_full))


def balanced_subset(dataset, per_class):
    selected = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        generator = torch.Generator().manual_seed(
            SEED + class_id
        )
        order = torch.randperm(
            len(class_indices),
            generator=generator,
        ).tolist()

        chosen = [
            class_indices[i]
            for i in order[:min(per_class, len(class_indices))]
        ]
        selected.extend(chosen)

    return Subset(dataset, selected)


train_data = balanced_subset(
    train_full, IMAGES_PER_CLASS
)
valid_data = balanced_subset(
    valid_full, VALIDATION_IMAGES_PER_CLASS
)

print("Training sample:", len(train_data))
print("Validation sample:", len(valid_data))

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ---------------- BUILD MODEL ----------------

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Freeze all feature layers first.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Fine-tune the last two feature blocks.
for block in list(model.features.children())[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

number_of_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(
    number_of_features,
    len(train_full.classes),
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

feature_parameters = [
    parameter
    for parameter in model.features.parameters()
    if parameter.requires_grad
]
classifier_parameters = list(model.classifier.parameters())

optimizer = torch.optim.Adam([
    {"params": feature_parameters, "lr": 0.00001},
    {"params": classifier_parameters, "lr": 0.0001},
])


# ---------------- TRAIN AND VALIDATE ----------------

best_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()

    # Keep frozen feature blocks in evaluation mode.
    for block in model.features.children():
        if all(not p.requires_grad for p in block.parameters()):
            block.eval()

    total_loss = 0.0
    total_seen = 0

    for step, (images, labels) in enumerate(
        train_loader, start=1
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total_seen += batch_count

        if step % 50 == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / max(total_seen, 1)
    print(
        f"Epoch {epoch + 1} complete. "
        f"Average training loss: {average_loss:.4f}"
    )

    # Validate after each epoch.
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / max(total, 1)

    print(f"Validation accuracy: {accuracy:.2f}%")
    print(f"Validation correct: {correct}/{total}")

    # Save the best validation checkpoint.
    if accuracy > best_accuracy:
        best_accuracy = accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_full.classes,
                "class_to_idx": train_full.class_to_idx,
                "image_size": IMAGE_SIZE,
                "best_validation_accuracy": best_accuracy,
            },
            MODEL_PATH,
        )

        with open(CLASSES_PATH, "w", encoding="utf-8") as file:
            json.dump(
                train_full.class_to_idx,
                file,
                indent=2,
            )

        print("Best model saved to:", MODEL_PATH)

print("Best validation accuracy:", f"{best_accuracy:.2f}%")
print("Class mapping saved to:", CLASSES_PATH)
print("Training and validation completed.")
