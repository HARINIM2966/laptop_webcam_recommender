import os
import io

import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Webcam Product Discovery",
    page_icon="📷",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "camera_bytes" not in st.session_state:
    st.session_state["camera_bytes"] = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "products.csv"
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "images"
)

EMBEDDING_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "image_embeddings.npy"
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

@st.cache_data
def load_products():

    return pd.read_csv(
        CSV_PATH
    )


products = load_products()


# ============================================================
# LOAD CLIP MODEL
# ============================================================

@st.cache_resource
def load_clip():

    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model = CLIPModel.from_pretrained(
        model_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    model.eval()

    return processor, model, device


processor, model, device = load_clip()


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        image_features = (
            vision_outputs.pooler_output
        )

        image_features = model.visual_projection(
            image_features
        )

    image_features = image_features / (
        image_features.norm(
            dim=-1,
            keepdim=True
        ) + 1e-10
    )

    return image_features.cpu().numpy()[0]


# ============================================================
# TEXT EMBEDDING
# ============================================================

def get_text_embedding(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        text_features = (
            text_outputs.pooler_output
        )

        text_features = model.text_projection(
            text_features
        )

    text_features = text_features / (
        text_features.norm(
            dim=-1,
            keepdim=True
        ) + 1e-10
    )

    return text_features.cpu().numpy()[0]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding,
    image_embeddings
):

    query_embedding = (
        query_embedding /
        (
            np.linalg.norm(
                query_embedding
            ) + 1e-10
        )
    )

    image_embeddings = (
        image_embeddings /
        (
            np.linalg.norm(
                image_embeddings,
                axis=1,
                keepdims=True
            ) + 1e-10
        )
    )

    scores = np.dot(
        image_embeddings,
        query_embedding
    )

    return scores


# ============================================================
# WEBCAM IMAGE VALIDATION
# ============================================================

def is_webcam_image(image):

    positive_prompts = [
        "a real computer webcam",
        "a real USB webcam",
        "a real laptop webcam",
        "a real external webcam camera",
        "a real webcam device used for video calls",
        "a real webcam mounted on a computer monitor"
    ]

    negative_prompts = [
        "a toy",
        "a toy camera",
        "a toy webcam",
        "a fake webcam",
        "a children's toy camera",
        "a doll",
        "a model or miniature camera",
        "a decorative object",
        "a phone",
        "a mobile phone",
        "a laptop computer",
        "another electronic device"
    ]

    positive_inputs = processor(
        text=positive_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    negative_inputs = processor(
        text=negative_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    positive_inputs = {
        key: value.to(device)
        for key, value in positive_inputs.items()
    }

    negative_inputs = {
        key: value.to(device)
        for key, value in negative_inputs.items()
    }

    with torch.no_grad():

        positive_outputs = model(
            **positive_inputs
        )

        negative_outputs = model(
            **negative_inputs
        )

    positive_score = (
        positive_outputs.logits_per_image
        .mean()
        .item()
    )

    negative_score = (
        negative_outputs.logits_per_image
        .mean()
        .item()
    )

    score_difference = (
    positive_score - negative_score
)

    return score_difference > 3.0


# ============================================================
# TITLE
# ============================================================

st.title(
    "📷 AI Webcam Product Discovery"
)

st.write(
    "Find similar webcam products using "
    "text, images, or your laptop webcam."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Search Options"
)


search_type = st.sidebar.radio(
    "Choose input type",
    [
        "Text Search",
        "Upload Image",
        "Laptop Webcam"
    ]
)


max_price = st.sidebar.number_input(
    "Maximum Price",
    min_value=0,
    value=10000,
    step=500
)


num_results = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# ============================================================
# TEXT SEARCH
# ============================================================

text_query = None

if search_type == "Text Search":

    text_query = st.text_input(
        "What kind of webcam are you looking for?",
        placeholder=(
            "Example: Full HD webcam with autofocus"
        )
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_image = None

if search_type == "Upload Image":

    uploaded_image = st.file_uploader(
        "Upload a webcam image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_image is not None:

        uploaded_image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded image",
            width=300
        )


# ============================================================
# LAPTOP WEBCAM
# ============================================================

if search_type == "Laptop Webcam":

    camera_file = st.camera_input(
        "Take a photo using your laptop webcam"
    )

    if camera_file is not None:

        camera_bytes = (
            camera_file.getvalue()
        )

        st.session_state[
            "camera_bytes"
        ] = camera_bytes

        camera_image = Image.open(
            io.BytesIO(camera_bytes)
        ).convert("RGB")

        st.image(
            camera_image,
            caption="Captured image",
            width=300
        )


# ============================================================
# SEARCH BUTTON
# ============================================================

search_button = st.button(
    "🔍 Search"
)


# ============================================================
# SEARCH PROCESS
# ============================================================

if search_button:

    query_embedding = None


    # ========================================================
    # TEXT SEARCH
    # ========================================================

    if search_type == "Text Search":

        if not text_query:

            st.warning(
                "Please enter a search query."
            )

            st.stop()

        st.info(
            "Searching for matching products..."
        )

        query_embedding = (
            get_text_embedding(
                text_query
            )
        )


    # ========================================================
    # UPLOAD IMAGE SEARCH
    # ========================================================

    elif search_type == "Upload Image":

        if uploaded_image is None:

            st.warning(
                "Please upload an image."
            )

            st.stop()

        if not is_webcam_image(
            uploaded_image
        ):

            st.warning(
                "This image does not appear "
                "to be a webcam. Please upload "
                "a webcam image."
            )

            st.stop()

        st.info(
            "Searching for visually similar products..."
        )

        query_embedding = (
            get_image_embedding(
                uploaded_image
            )
        )


    # ========================================================
    # LAPTOP WEBCAM SEARCH
    # ========================================================

    elif search_type == "Laptop Webcam":


        camera_bytes = (
            st.session_state.get(
                "camera_bytes"
            )
        )

        if camera_bytes is None:

            st.warning(
                "Please take a photo first."
            )

            st.stop()

        camera_image = Image.open(
            io.BytesIO(camera_bytes)
        ).convert("RGB")
        if not is_webcam_image(
            camera_image
        ):

            st.warning(
                "This image does not appear "
                "to be a webcam. Please show "
                "a webcam to the camera."
            )

            st.stop()

        st.info(
            "Searching using your laptop "
            "webcam image..."
        )

        query_embedding = (
            get_image_embedding(
                camera_image
            )
        )


    # ========================================================
    # CHECK EMBEDDINGS
    # ========================================================

    if not os.path.exists(
        EMBEDDING_PATH
    ):

        st.error(
            "Image embeddings not found. "
            "Run create_embeddings.py first."
        )

        st.stop()


    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    image_embeddings = np.load(
        EMBEDDING_PATH
    )


    # ========================================================
    # CALCULATE SIMILARITY
    # ========================================================

    scores = cosine_similarity(
        query_embedding,
        image_embeddings
    )


    # ========================================================
    # ADD SCORES TO PRODUCTS
    # ========================================================

    results = products.copy()

    results["similarity"] = scores


    # ========================================================
    # PRICE FILTER
    # ========================================================

    results = results[
        results["price"] <= max_price
    ]


    # ========================================================
    # SORT RESULTS
    # ========================================================

    results = results.sort_values(
        by="similarity",
        ascending=False
    )


    results = results.head(
        num_results
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader(
        "Recommended Webcams"
    )


    if len(results) == 0:

        st.warning(
            "No webcams found within your "
            "selected price range."
        )

    else:

        for _, product in results.iterrows():

            st.markdown(
                "---"
            )

            col1, col2 = st.columns(
                [1, 2]
            )


            # ------------------------------------------------
            # PRODUCT IMAGE
            # ------------------------------------------------

            with col1:

                image_path = os.path.join(
                    IMAGE_FOLDER,
                    product["image"]
                )

                if os.path.exists(
                    image_path
                ):

                    st.image(
                        image_path,
                        width=250
                    )

                else:

                    st.warning(
                        "Product image not found."
                    )


            # ------------------------------------------------
            # PRODUCT DETAILS
            # ------------------------------------------------

            with col2:

                st.subheader(
                    product["product_name"]
                )

                st.write(
                    "**Brand:**",
                    product["brand"]
                )

                st.write(
                    "**Price:** ₹",
                    product["price"]
                )

                st.write(
                    "**Resolution:**",
                    product["resolution"]
                )

                st.write(
                    "**FPS:**",
                    product["fps"]
                )

                st.write(
                    "**Autofocus:**",
                    product["autofocus"]
                )

                st.write(
                    "**Microphone:**",
                    product["microphone"]
                )

                st.write(
                    "**Connection:**",
                    product["connection"]
                )

                st.write(
                    product["description"]
                )

                st.write(
                    "**Similarity:**",
                    f"{product['similarity']:.2%}"
                )