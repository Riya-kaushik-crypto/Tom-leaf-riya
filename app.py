# app.py
# Leaf validation: CLIP zero-shot (openai/clip-vit-base-patch32)
# Fix 4: Image quality pre-check (brightness + blur) before CLIP/model
# Fix 1: Confidence threshold + tiered severity (was hardcoded SEVERE)
# ============================================================

import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models as keras_models
import numpy as np
import pickle
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================
# 1. CONFIG
# ============================================================

IMG_SIZE             = 224
CONFIDENCE_THRESHOLD = 0.70   # below this → "Low Confidence" warning
SEVERE_THRESHOLD     = 0.90   # FIX 1: above this → SEVERE, else MODERATE
WEIGHTS_PATH         = 'model/tomato_weights.weights.h5'
CLASS_INDICES_PATH   = 'model/class_indices.pkl'
FALLBACK_CLASSES     = ['early_blight', 'Healthy', 'late_blight']

# ── FIX 4: Image quality thresholds ─────────────────────────
MIN_BRIGHTNESS  = 50    # 0–255 — reject if image is too dark
MAX_BRIGHTNESS  = 220   # reject if overexposed
MIN_BLUR_SCORE  = 80    # Laplacian variance — reject if too blurry

# ============================================================
# 2. LOAD CLIP VALIDATOR
# ============================================================

print("🔄 Loading CLIP validator...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ CLIP loaded!")

CLIP_LABELS = [
    "a tomato leaf",                           # 0 → leaf ✅
    "a diseased tomato leaf with brown spots", # 1 → leaf ✅
    "a random object or household item",       # 2 → not leaf ❌
    "a human hand or body part",               # 3 → not leaf ❌
    "an indoor background or furniture",       # 4 → not leaf ❌
]
LEAF_INDICES   = {0, 1}
LEAF_THRESHOLD = 0.40

# ============================================================
# 3. BUILD DISEASE MODEL & LOAD WEIGHTS
# ============================================================

def build_model():
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    m = keras_models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    return m

print("🔄 Loading disease model...")
model = build_model()
model.load_weights(WEIGHTS_PATH)
print("✅ Disease model loaded!")

try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    CLASS_NAMES = sorted(class_indices, key=class_indices.get)
    print(f"✅ Classes: {CLASS_NAMES}")
except FileNotFoundError:
    CLASS_NAMES = FALLBACK_CLASSES
    print(f"⚠️ Using fallback classes: {CLASS_NAMES}")

# ============================================================
# 4. DISEASE KNOWLEDGE BASE
# FIX 1: severity removed from here — now computed dynamically
# ============================================================

disease_info = {
    "early_blight": {
        "emoji"   : "🟤",
        "name"    : "Early Blight",
        "symptoms": [
            "Dark brown spots with concentric rings (target-board pattern)",
            "Yellow halo around the spots",
            "Lower / older leaves affected first",
            "Lesions stay dry and crisp",
        ],
        "treatment": [
            "Remove and destroy infected leaves immediately",
            "Spray Mancozeb or Chlorothalonil fungicide every 7–10 days",
            "Maintain plant spacing for good airflow",
            "Water at soil level, not overhead",
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Use certified disease-free seeds",
            "Rotate crops every season",
            "Mulch around plants to reduce soil splash",
        ],
    },
    "late_blight": {
        "emoji"   : "⚫",
        "name"    : "Late Blight",
        "symptoms": [
            "Water-soaked, greasy-looking lesions on leaves",
            "White fuzzy mold on underside of leaf (humid conditions)",
            "Rapid browning — entire leaf dies within days",
            "Dark brown lesions on stems and fruit",
        ],
        "treatment": [
            "Remove and DESTROY infected plants (do not compost)",
            "Spray Copper-based fungicide or Metalaxyl immediately",
            "Apply Ridomil Gold or Revus for severe cases",
            "Isolate affected plants to stop spread",
        ],
        "prevention": [
            "Avoid wet, humid conditions around plants",
            "Plant resistant varieties (e.g., Mountain Magic, Defiant)",
            "Improve air circulation between plants",
            "Apply preventive copper spray during monsoon season",
        ],
    },
    "Healthy": {
        "emoji"   : "✅",
        "name"    : "Healthy",
        "symptoms": [
            "No disease signs detected",
            "Leaves appear normal and green",
        ],
        "treatment" : ["No treatment needed"],
        "prevention": [
            "Continue regular watering and fertilization",
            "Monitor weekly for early signs of disease",
            "Maintain good field hygiene",
        ],
    },
}

# ============================================================
# FIX 1: Dynamic severity based on label + confidence
# ============================================================

def get_severity(label, confidence):
    """
    Returns a severity string computed from confidence score.
    Prevents healthy/uncertain images being labelled SEVERE.

    Thresholds:
      >= SEVERE_THRESHOLD (0.90) → SEVERE
      >= CONFIDENCE_THRESHOLD (0.70) → MODERATE
      < CONFIDENCE_THRESHOLD → LOW (should already be caught above)
    """
    if label == "Healthy":
        return "No disease detected — plant looks healthy!"
    if confidence >= SEVERE_THRESHOLD:
        if label == "late_blight":
            return "SEVERE — can destroy entire crop within days"
        return "SEVERE — act immediately"
    if confidence >= CONFIDENCE_THRESHOLD:
        return "MODERATE — monitor closely and apply treatment"
    return "LOW — confidence too low for a reliable assessment"

# ============================================================
# FIX 4: Image quality pre-check
# ============================================================

def check_image_quality(pil_image):
    """
    Runs BEFORE CLIP and the disease model.
    Checks brightness and sharpness using OpenCV.
    Returns (is_ok: bool, reason: str).

    Thresholds (all tunable in CONFIG section):
      brightness < MIN_BRIGHTNESS (50)  → too dark
      brightness > MAX_BRIGHTNESS (220) → overexposed
      blur_score < MIN_BLUR_SCORE (80)  → too blurry
    """
    img  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = float(gray.mean())
    if brightness < MIN_BRIGHTNESS:
        return False, (
            f"Image is too dark (brightness {brightness:.0f}/255).\n"
            "Please retake in natural daylight on a plain background."
        )
    if brightness > MAX_BRIGHTNESS:
        return False, (
            f"Image is overexposed (brightness {brightness:.0f}/255).\n"
            "Avoid direct flash or harsh sunlight on the leaf."
        )

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < MIN_BLUR_SCORE:
        return False, (
            f"Image is too blurry (sharpness score {blur_score:.0f}).\n"
            "Hold the camera steady and tap to focus on the leaf."
        )

    return True, ""

# ============================================================
# 5. PREPROCESS
# ============================================================

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================================
# 6. CLIP LEAF VALIDATOR
# ============================================================

def is_tomato_leaf(image: Image.Image) -> tuple:
    """
    Uses CLIP zero-shot classification to verify the image is a tomato leaf.
    Returns (True, "") if leaf, or (False, reason) if not.
    """
    try:
        inputs = clip_processor(
            text=CLIP_LABELS,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = clip_model(**inputs)

        probs      = outputs.logits_per_image.softmax(dim=1)[0]
        leaf_prob  = sum(probs[i].item() for i in LEAF_INDICES)
        top_idx    = probs.argmax().item()
        top_label  = CLIP_LABELS[top_idx]
        top_prob   = probs[top_idx].item()

        print(f"CLIP scores: { {CLIP_LABELS[i]: f'{probs[i]:.2%}' for i in range(len(CLIP_LABELS))} }")
        print(f"Leaf probability: {leaf_prob:.2%}")

        if leaf_prob < LEAF_THRESHOLD:
            return False, (
                f"Image looks like: '{top_label}' ({top_prob:.0%} confidence).\n"
                f"Leaf confidence: {leaf_prob:.0%} (minimum required: {LEAF_THRESHOLD:.0%})"
            )
        return True, ""

    except Exception as e:
        print(f"⚠️ CLIP error: {e} — allowing image through")
        return True, ""   # fail open so genuine errors don't block users

# ============================================================
# 7. PREDICT
# ============================================================

def predict(image):
    if image is None:
        return "⚠️ Please upload an image first."
    try:
        # ── FIX 4: Quality gate — runs before CLIP and model ────
        ok, quality_reason = check_image_quality(image)
        if not ok:
            return (
                "📸 Poor Image Quality — Cannot Diagnose\n\n"
                f"Reason: {quality_reason}\n\n"
                "Tips for a good photo:\n"
                " • Shoot in natural daylight (not indoors / at night)\n"
                " • Use a plain white or blue background\n"
                " • Hold phone steady — tap screen to focus\n"
                " • Capture one leaf close-up, filling the frame"
            )

        # ── Step 1: CLIP leaf check ──────────────────────────────
        is_leaf, reason = is_tomato_leaf(image)
        if not is_leaf:
            return (
                "🚫 Not a tomato leaf image.\n\n"
                f"Reason: {reason}\n\n"
                "Please upload a clear photo of a tomato leaf.\n\n"
                "📸 Tips for a good photo:\n"
                " • Shoot in natural daylight\n"
                " • Capture one leaf, close-up\n"
                " • Make the leaf fill most of the frame\n"
                " • Use a plain white or blue background\n"
                " • Keep lesions clearly visible and in focus"
            )

        # ── Step 2: Disease classification ──────────────────────
        img_array = preprocess(image)
        preds     = model.predict(img_array, verbose=0)[0]

        results = sorted(
            [{"label": CLASS_NAMES[i], "score": float(preds[i])}
             for i in range(len(CLASS_NAMES))],
            key=lambda x: x["score"], reverse=True
        )

        top        = results[0]
        label      = top["label"]
        confidence = top["score"]
        info       = disease_info.get(label, {})
        emoji      = info.get("emoji", "🍅")
        name       = info.get("name", label)

        # ── FIX 1: Dynamic severity (replaces hardcoded "SEVERE") ─
        severity = get_severity(label, confidence)

        # ── Step 3: Build report ─────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            out  = f"⚠️ Low Confidence ({confidence:.1%}) — Best guess: {name}\n"
            out += "Please retake photo in better lighting for a reliable result.\n"
        else:
            out  = f"{emoji} Detected: {name}\n"
            out += f"Confidence : {confidence:.1%}\n"

        out += f"Severity   : {severity}\n"   # FIX 1: dynamic, not hardcoded
        out += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        out += "\n🔍 Symptoms:\n"
        for s in info.get("symptoms", []):
            out += f"  • {s}\n"

        out += "\n💊 Treatment:\n"
        for t in info.get("treatment", []):
            out += f"  • {t}\n"

        out += "\n🛡️ Prevention:\n"
        for p in info.get("prevention", []):
            out += f"  • {p}\n"

        out += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        out += "\nAll scores:\n"
        for r in results:
            name_r = disease_info.get(r["label"], {}).get("name", r["label"])
            bar    = "█" * int(r["score"] * 25)
            out   += f"  • {name_r:15s} {r['score']:.1%} {bar}\n"

        return out

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================
# 8. GRADIO UI  (unchanged)
# ============================================================

with gr.Blocks(title="🍅 TomLeafVision", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🍅 TomLeafVision — Tomato Leaf Disease Detector
    Powered by **MobileNetV2** + **CLIP** leaf validation.
    > **Tip:** Clear photo in **natural daylight** on a **plain background** gives best accuracy.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Tomato Leaf Image", height=320)
            with gr.Row():
                submit_btn = gr.Button("🔍 Detect Disease", variant="primary",  size="lg")
                clear_btn  = gr.Button("🗑️ Clear",          variant="secondary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="📋 Diagnosis Report", lines=24, show_copy_button=True)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text, api_name="predict")
    clear_btn.click(fn=lambda: (None, ""), outputs=[image_input, output_text])

    gr.Markdown("""
    ---
    ### 📸 Photo Guide
    | ✅ Do this | ❌ Avoid this |
    |---|---|
    | Natural daylight | Dark / indoor lighting |
    | Plain white or blue background | Cluttered backgrounds |
    | One leaf, close-up | Multiple plants in frame |
    | Early-to-mid stage symptoms | Fully dried / dead leaves |

    **Detectable classes:** Early Blight · Late Blight · Healthy
    """)

demo.queue()
demo.launch()