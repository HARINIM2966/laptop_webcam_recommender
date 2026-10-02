import os
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# Paths
# -----------------------------

CSV_PATH = "data/products.csv"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "embeddings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load product data
# -----------------------------

products = pd.read_csv(CSV_PATH)

# -----------------------------
# Load CLIP model
# -----------------------------

print("Loading CLIP model... - create_embeddings.py:29")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()

print("Using device: - create_embeddings.py:44", device)

# -----------------------------
# Create image embeddings
# -----------------------------

image_embeddings = []

for index, product in products.iterrows():

    image_name = product["image"]

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    print("Processing: - create_embeddings.py:61", image_path)

    if not os.path.exists(image_path):
        print("WARNING: Image not found: - create_embeddings.py:64", image_path)

        # Create zero vector if image doesn't exist
        image_embeddings.append(
            np.zeros(512, dtype=np.float32)
        )

        continue

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(
            **inputs
        )

    if hasattr(outputs, "pooler_output"):
        image_features = outputs.pooler_output
    else:
        image_features = outputs

    # Normalize embedding
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    embedding = image_features.cpu().numpy()[0]
    image_embeddings.append(embedding)

# -----------------------------
# Save embeddings
# -----------------------------

image_embeddings = np.array(
    image_embeddings,
    dtype=np.float32
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "image_embeddings.npy"
    ),
    image_embeddings
)

# Save product metadata
with open(
    os.path.join(
        OUTPUT_FOLDER,
        "products.pkl"
    ),
    "wb"
) as file:

    pickle.dump(products, file)

print()
print("=================================== - create_embeddings.py:133")
print("Image embeddings created successfully! - create_embeddings.py:134")
print("Number of products: - create_embeddings.py:135", len(products))
print("Embedding shape: - create_embeddings.py:136", image_embeddings.shape)
print("=================================== - create_embeddings.py:137")