
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Webcam Recommender",
    page_icon="📷",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.csv"
IMAGE_DIR = BASE_DIR / "images"
EMBEDDING_PATH = BASE_DIR / "embeddings" / "image_embeddings.npy"


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #eef8ff;
        color: #172b4d;
    }

    h1, h2, h3 {
        color: #123c69 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: white !important;
        color: #111111 !important;
    }

    /* Selectbox and multiselect controls */
    div[data-baseweb="select"] > div {
        background-color: #102a43 !important;
        border-color: #486581 !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="select"] input {
        color: white !important;
    }

    /* Dropdown popover and options */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[role="listbox"] {
        background-color: #102a43 !important;
        color: white !important;
    }

    div[role="option"] {
        background-color: #102a43 !important;
        color: white !important;
    }

    div[role="option"]:hover,
    div[aria-selected="true"] {
        background-color: #244766 !important;
        color: white !important;
    }

    div[data-baseweb="tag"] {
        background-color: #244766 !important;
        color: white !important;
    }

    div[data-baseweb="tag"] * {
        color: white !important;
    }

    div[data-baseweb="select"] svg {
        fill: white !important;
        color: white !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 10px;
        padding: 12px;
    }

    /* Upload arrow/icon */
    div[data-testid="stFileUploader"] svg {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }
    div[data-testid="stFileUploader"] button {
    color: white !important;
    }
    /* Camera permission and camera-input text */
    div[data-testid="stCameraInput"] p,
    div[data-testid="stCameraInput"] label,
    div[data-testid="stCameraInput"] span,
    div[data-testid="stCameraInput"] button,
    div[data-testid="stCameraInput"] [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }

    /* Camera consent prompt: selectors vary by Streamlit version */
    div[data-testid="stCameraInput"] [role="dialog"] *,
    div[data-testid="stCameraInput"] [data-testid*="camera"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HEADER
# =========================================================

st.title("📷 AI Laptop Webcam Recommender")

st.write(
    """
    Discover webcams based on your budget and preferences.
    Get random recommendations when you are unsure what to choose,
    compare products, or use CLIP-based AI image matching.
    """
)

st.divider()


# =========================================================
# 4. LOAD PRODUCT DATA
# =========================================================

@st.cache_data
def load_products(csv_path):
    df = pd.read_csv(csv_path)

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    aliases = {
        "product_name": ["name", "webcam_name", "title"],
        "brand": ["manufacturer", "company"],
        "price": ["cost", "price_inr", "price_₹"],
        "image": ["image_path", "image_file", "image_filename"],
        "best_for": ["purpose", "usage", "recommended_for"],
        "resolution": ["video_resolution"],
        "fps": ["frame_rate", "frames_per_second"],
        "field_of_view": ["fov", "field_of_view_degrees"],
        "privacy_shutter": ["privacy_cover", "shutter"],
        "low_light": ["low_light_performance"],
        "features": ["key_features", "description"],
    }

    for standard, alternatives in aliases.items():
        if standard not in df.columns:
            for alternative in alternatives:
                if alternative in df.columns:
                    df.rename(
                        columns={alternative: standard},
                        inplace=True,
                    )
                    break

    if "product_name" not in df.columns:
        raise ValueError(
            "products.csv must contain a product_name or name column."
        )

    if "price" not in df.columns:
        raise ValueError(
            "products.csv must contain a price column."
        )

    defaults = {
        "brand": "Unknown",
        "image": "",
        "best_for": "General use",
        "resolution": "Not specified",
        "fps": "Not specified",
        "field_of_view": "Not specified",
        "privacy_shutter": "Not specified",
        "low_light": "Not specified",
        "features": "Not specified",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["product_name", "price"]).copy()

    for column, default in defaults.items():
        df[column] = df[column].fillna(default).astype(str)

    df["product_name"] = df["product_name"].astype(str)

    return df.reset_index(drop=True)


if not DATA_PATH.exists():
    st.error(f"Product dataset not found: {DATA_PATH}")
    st.stop()

try:
    products = load_products(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load products.csv: {error}")
    st.stop()

if products.empty:
    st.error("No valid products were found in products.csv.")
    st.stop()


# =========================================================
# 5. IMAGE PATH HELPER
# =========================================================

def get_product_image(product):
    image_value = str(product.get("image", "")).strip()

    if image_value.lower() not in ("", "nan", "none", "null"):
        image_path = Path(image_value)

        candidates = [
            image_path if image_path.is_absolute()
            else BASE_DIR / image_path,
            IMAGE_DIR / image_value,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

    if "product_id" in product.index:
        try:
            candidate = IMAGE_DIR / f"webcam{int(product['product_id'])}.jpg"
            if candidate.is_file():
                return str(candidate)
        except (ValueError, TypeError, OverflowError):
            pass

    try:
        candidate = IMAGE_DIR / f"webcam{int(product.name) + 1}.jpg"
        if candidate.is_file():
            return str(candidate)
    except (ValueError, TypeError, OverflowError):
        pass

    return None


# =========================================================
# 6. USER INPUTS
# =========================================================

st.header("🔎 Choose Your Requirements")

price_col1, price_col2 = st.columns(2)

with price_col1:
    min_price = st.number_input(
        "Minimum Price (₹)",
        value=0.0,
        step=100.0,
        format="%.2f",
        key="min_price",
    )

with price_col2:
    max_price = st.number_input(
        "Maximum Price (₹)",
        value=float(max(10000, products["price"].max())),
        step=100.0,
        format="%.2f",
        key="max_price",
    )

num_recommendations = st.number_input(
    "Number of Recommendations (required)",
    value=5,
    step=1,
    key="num_recommendations",
    help="Enter a positive whole number.",
)

st.subheader("Brand")

all_brands = sorted(products["brand"].unique().tolist())

selected_brands = st.multiselect(
    "Choose one or more brands",
    options=all_brands,
    default=[],
    placeholder="All brands",
    key="selected_brands",
    help="Leave empty to include all brands. Select multiple brands if desired.",
)

if not selected_brands:
    st.caption("All brands are included.")

st.subheader("Intended Use")

all_purposes = sorted(products["best_for"].unique().tolist())

selected_purpose = st.selectbox(
    "Where will you use the webcam?",
    options=["All purposes"] + all_purposes,
    index=0,
    key="selected_purpose",
)


# =========================================================
# 7. VALIDATION
# =========================================================

valid_inputs = True

if min_price < 0:
    st.error("❌ Minimum price cannot be negative.")
    valid_inputs = False

if max_price < 0:
    st.error("❌ Maximum price cannot be negative.")
    valid_inputs = False

# Both zero means no price limit.
both_prices_zero = min_price == 0 and max_price == 0

if min_price >= max_price and not both_prices_zero:
    st.error("❌ Minimum price must be less than maximum price.")
    valid_inputs = False

if num_recommendations < 0:
    st.error("❌ Number of recommendations cannot be negative.")
    valid_inputs = False
elif num_recommendations == 0:
    st.error("❌ Number of recommendations is mandatory. Enter at least 1.")
    valid_inputs = False
elif not float(num_recommendations).is_integer():
    st.error("❌ Number of recommendations must be a whole number.")
    valid_inputs = False


# =========================================================
# 8. RECOMMENDATION BUTTON
# =========================================================

st.divider()

if st.button(
    "✨ Recommend Webcams",
    type="primary",
    use_container_width=True,
):
    st.session_state["recommendations"] = pd.DataFrame()
    st.session_state["recommendation_message"] = ""

    if not valid_inputs:
        st.session_state["recommendation_message"] = (
            "Please correct the input errors before requesting recommendations."
        )
    else:
        if both_prices_zero:
            candidates = products.copy()
        else:
            candidates = products[
                (products["price"] >= min_price)
                & (products["price"] <= max_price)
            ].copy()

        if selected_brands:
            candidates = candidates[
                candidates["brand"].isin(selected_brands)
            ].copy()

        if selected_purpose != "All purposes":
            candidates = candidates[
                candidates["best_for"] == selected_purpose
            ].copy()

        if candidates.empty:
            st.session_state["recommendation_message"] = (
                "No webcams match your filters. Try a different price range "
                "or change the selected filters."
            )
        else:
            count = min(int(num_recommendations), len(candidates))

            recommendations = candidates.sample(
                n=count,
                replace=False,
            ).reset_index(drop=True)

            st.session_state["recommendations"] = recommendations


# =========================================================
# 9. DISPLAY RECOMMENDATIONS
# =========================================================

recommendations = st.session_state.get(
    "recommendations",
    pd.DataFrame(),
)

message = st.session_state.get("recommendation_message", "")

if message:
    st.warning(message)

if not recommendations.empty:
    st.divider()
    st.header("⭐ Recommended Webcams")

    st.write(
        f"Showing {len(recommendations)} webcam(s) matching your filters."
    )

    for _, product in recommendations.iterrows():
        with st.container(border=True):
            image_col, details_col = st.columns([1, 2])

            with image_col:
                image_path = get_product_image(product)

                if image_path:
                    st.image(image_path, use_container_width=True)
                else:
                    st.info("Product image not available.")

            with details_col:
                st.subheader(product["product_name"])
                st.write(f"**Brand:** {product['brand']}")
                st.write(f"**Price:** ₹{product['price']:,.2f}")
                st.write(f"**Best for:** {product['best_for']}")
                st.write(f"**Resolution:** {product['resolution']}")
                st.write(f"**Frame rate:** {product['fps']}")
                st.write(f"**Field of view:** {product['field_of_view']}")
                st.write(f"**Privacy shutter:** {product['privacy_shutter']}")
                st.write(f"**Low-light performance:** {product['low_light']}")
                st.write(f"**Features:** {product['features']}")


# =========================================================
# 10. COMPARE UP TO 3 RECOMMENDED WEBCAMS
# =========================================================

st.divider()
st.header("📊 Compare Recommended Webcams")

if recommendations.empty:
    st.info(
        "Click Recommend Webcams first. Then select 2 or 3 cameras to compare."
    )
else:
    comparison_selection = st.multiselect(
        "Select up to 3 recommended webcams",
        options=recommendations["product_name"].tolist(),
        max_selections=3,
        key="comparison_selection",
        placeholder="Choose webcams to compare",
    )

    if st.button(
        "📊 Compare Selected Webcams",
        type="primary",
        use_container_width=True,
    ):
        if len(comparison_selection) < 2:
            st.error("Select at least 2 webcams to compare.")
        elif len(comparison_selection) > 3:
            st.error("You can compare a maximum of 3 webcams.")
        else:
            comparison = recommendations[
                recommendations["product_name"].isin(comparison_selection)
            ].copy()

            comparison_columns = [
                "product_name",
                "brand",
                "price",
                "best_for",
                "features",
                "resolution",
                "fps",
                "field_of_view",
                "privacy_shutter",
                "low_light",
            ]

            comparison_columns = [
                column for column in comparison_columns
                if column in comparison.columns
            ]

            comparison = comparison[comparison_columns].set_index(
                "product_name"
            ).T

            comparison.index = [
                "Brand" if field == "brand"
                else "Price (₹)" if field == "price"
                else "Best For (Intended Use)" if field == "best_for"
                else "Features" if field == "features"
                else "Resolution" if field == "resolution"
                else "Frame Rate" if field == "fps"
                else "Field of View" if field == "field_of_view"
                else "Privacy Shutter" if field == "privacy_shutter"
                else "Low-Light Performance" if field == "low_light"
                else field.replace("_", " ").title()
                for field in comparison.index
            ]

            st.subheader("Side-by-Side Comparison")
            st.dataframe(comparison, use_container_width=True)


# =========================================================
# 11. GENAI: CLIP IMAGE MATCHING
# =========================================================

st.divider()
st.header("🤖 GenAI: Find Similar Webcams from an Image")

st.write(
    """
    This feature uses the pretrained CLIP vision model to extract image
    embeddings and rank webcam products by visual similarity.
    """
)

upload_col, camera_col = st.columns(2)

with upload_col:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="clip_upload",
    )

with camera_col:
    captured_image = st.camera_input(
        "Or capture an image",
        key="clip_camera",
    )

query_file = (
    uploaded_image
    if uploaded_image is not None
    else captured_image
)


@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return processor, model, device


def get_image_embedding(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt")

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    features = features / (
        features.norm(dim=-1, keepdim=True) + 1e-10
    )

    return features.cpu().numpy()[0]


@st.cache_resource
def load_webcam_validation_text_features():
    """Cache CLIP text embeddings used to distinguish webcams from other objects."""
    processor, model, device = load_clip_model()

    webcam_prompts = [
        "a real external USB webcam with a glass camera lens",
        "a computer webcam with a circular lens and monitor clip",
        "a real webcam product designed for computer video calls",
        "a black Logitech style webcam with a lens in the center",
        "a webcam mounted on top of a computer monitor",
        "a standalone external computer webcam product photograph",
        "a small electronic webcam with a mounting bracket",
        "a real USB video conferencing camera for a laptop",
    ]

    non_webcam_prompts = [
        "a plastic toy camera for children",
        "a toy webcam made of plastic, not a real electronic device",
        "a doll or action figure toy",
        "a stuffed animal or teddy bear",
        "a toy object photographed on a table",
        "a drawing or cartoon of a webcam",
        "a smartphone or mobile phone",
        "a laptop computer without an external webcam",
        "a computer mouse or keyboard",
        "a digital camera or DSLR camera, not a computer webcam",
        "a household object unrelated to a webcam",
        "a person, pet, food, or landscape",
    ]

    prompts = webcam_prompts + non_webcam_prompts

    inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features = text_features / (
        text_features.norm(dim=-1, keepdim=True) + 1e-10
    )

    webcam_count = len(webcam_prompts)

    return (
        text_features[:webcam_count].cpu(),
        text_features[webcam_count:].cpu(),
    )


def validate_uploaded_webcam(image, processor, model, device):
    """Reject likely non-webcam images before performing product matching."""
    image_embedding = get_image_embedding(
        image,
        processor,
        model,
        device,
    )

    image_tensor = torch.tensor(
        image_embedding,
        dtype=torch.float32,
    )

    webcam_features, non_webcam_features = (
        load_webcam_validation_text_features()
    )

    webcam_scores = webcam_features @ image_tensor
    non_webcam_scores = non_webcam_features @ image_tensor

    best_webcam_score = float(webcam_scores.max().item())
    best_non_webcam_score = float(non_webcam_scores.max().item())

    margin = best_webcam_score - best_non_webcam_score

    # Heuristic thresholds; image classification is not infallible.
    is_webcam = (
        best_webcam_score >= 0.22
        and margin >= 0.015
    )

    return is_webcam, best_webcam_score, best_non_webcam_score


if query_file is not None:
    try:
        query_image = Image.open(query_file).convert("RGB")

        st.image(
            query_image,
            caption="Image provided for AI matching",
            width=300,
        )

        if not EMBEDDING_PATH.exists():
            st.warning(
                "Image embeddings were not found. Run create_embeddings.py first."
            )
        elif not valid_inputs:
            st.warning("Correct the input errors before AI matching.")
        else:
            if st.button(
                "🤖 Find Similar Webcams with AI",
                type="primary",
                key="clip_recommend_button",
            ):
                with st.spinner(
                    "Checking the image and finding similar webcams..."
                ):
                    processor, model, device = load_clip_model()

                    # Validate before generating any recommendations.
                    is_webcam, webcam_score, non_webcam_score = (
                        validate_uploaded_webcam(
                            query_image,
                            processor,
                            model,
                            device,
                        )
                    )

                    if not is_webcam:
                        st.error(
                            "❌ Invalid image! Please upload or capture a "
                            "real computer webcam. Toy cameras, toys, drawings, "
                            "and unrelated objects cannot be used for recommendations."
                        )
                        st.caption(
                            f"Validation scores — webcam: {webcam_score:.3f}; "
                            f"non-webcam: {non_webcam_score:.3f}."
                        )

                    else:
                        st.success(
                            "✅ A webcam image was detected. Finding similar products..."
                        )

                        embeddings = np.asarray(
                            np.load(EMBEDDING_PATH),
                            dtype=np.float32,
                        )

                        if embeddings.ndim != 2:
                            st.error(
                                "Embeddings must be a 2D array. Regenerate them."
                            )

                        elif len(embeddings) != len(products):
                            st.error(
                                "Embedding count does not match products.csv. "
                                "Regenerate embeddings in the same product order."
                            )

                        else:
                            query_embedding = get_image_embedding(
                                query_image,
                                processor,
                                model,
                                device,
                            )

                            if embeddings.shape[1] != len(query_embedding):
                                st.error(
                                    "Embedding dimensions do not match the CLIP "
                                    "model. Regenerate the embeddings."
                                )

                            else:
                                norms = (
                                    np.linalg.norm(
                                        embeddings,
                                        axis=1,
                                        keepdims=True,
                                    )
                                    + 1e-10
                                )

                                normalized = embeddings / norms
                                scores = normalized @ query_embedding

                                scored = products.copy()
                                scored["similarity"] = scores

                                # Both zero prices mean no price limit.
                                if not both_prices_zero:
                                    scored = scored[
                                        (scored["price"] >= min_price)
                                        & (scored["price"] <= max_price)
                                    ].copy()

                                if selected_brands:
                                    scored = scored[
                                        scored["brand"].isin(selected_brands)
                                    ].copy()

                                if selected_purpose != "All purposes":
                                    scored = scored[
                                        scored["best_for"] == selected_purpose
                                    ].copy()

                                scored = scored.sort_values(
                                    "similarity",
                                    ascending=False,
                                ).head(int(num_recommendations))

                                st.subheader("AI-Matched Webcams")

                                if scored.empty:
                                    st.warning(
                                        "No products match your filters."
                                    )

                                for _, product in scored.iterrows():
                                    with st.container(border=True):
                                        image_col, details_col = st.columns([1, 2])

                                        with image_col:
                                            image_path = get_product_image(product)

                                            if image_path:
                                                st.image(
                                                    image_path,
                                                    use_container_width=True,
                                                )
                                            else:
                                                st.info("Image unavailable.")

                                        with details_col:
                                            st.subheader(
                                                product["product_name"]
                                            )
                                            st.write(
                                                f"**Brand:** {product['brand']}"
                                            )
                                            st.write(
                                                f"**Price:** ₹{product['price']:,.2f}"
                                            )
                                            st.write(
                                                f"**Best for:** {product['best_for']}"
                                            )
                                            st.write(
                                                f"**Features:** {product['features']}"
                                            )
                                            st.write(
                                                f"**CLIP similarity:** "
                                                f"{product['similarity']:.3f}"
                                            )

    except Exception as error:
        st.error(f"AI image matching failed: {error}")


# =========================================================
# 12. PROJECT STATUS
# =========================================================

with st.expander("Dataset and Project Status"):
    st.write(f"**Dataset:** `{DATA_PATH}`")
    st.write(f"**Valid products:** {len(products)}")
    st.write(f"**Images folder:** `{IMAGE_DIR}`")
    st.write(
        "**CLIP embeddings:** "
        + ("Found" if EMBEDDING_PATH.exists() else "Not found")
    )
    st.write("**Dataset columns:** " + ", ".join(products.columns.tolist()))

st.divider()
st.caption("AI Webcam Recommender | Streamlit + PyTorch + CLIP")
