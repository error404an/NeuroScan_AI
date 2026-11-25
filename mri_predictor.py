from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "brain_mri_model.keras"
CLASS_NAMES_PATH = APP_DIR / "models" / "class_names.json"
IMAGE_SIZE = (160, 160)

_model_cache = None
_class_names_cache = None

def load_model_and_classes():
    """Load the saved CNN and its exact output class order."""
    global _model_cache, _class_names_cache

    if _model_cache is not None and _class_names_cache is not None:
        return _model_cache, _class_names_cache

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or len(class_names) != 4:
        raise ValueError("class_names.json must contain exactly four class names.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Check output count matches the saved labels.
    output_shape = model.output_shape
    if not isinstance(output_shape, tuple) or output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model output shape {output_shape} does not match "
            f"{len(class_names)} saved class labels."
        )

    _model_cache = model
    _class_names_cache = class_names
    return model, class_names


def predict_mri_image(image):
    """
    Predict one 2D image slice using the same RGB, 160x160 and 1/255
    preprocessing used by scripts/train_model.py.

    Returns class scores and the highest-scoring class. Scores are model
    outputs, not calibrated clinical probabilities or a diagnosis.
    """
    model, class_names = load_model_and_classes()

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(image)

    pil_image = pil_image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(pil_image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    # The training model includes Rescaling(1/255) as its first preprocessing
    # operation after augmentation, so do not divide by 255 here.
    scores = model.predict(batch, verbose=0)[0]
    scores = np.asarray(scores, dtype=float)

    if scores.shape[0] != len(class_names):
        raise ValueError("Prediction output count does not match class names.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned non-finite prediction scores.")

    # Softmax scores are displayed as percentages, with explicit caution.
    score_map = {
        str(name): round(float(score) * 100, 2)
        for name, score in zip(class_names, scores)
    }
    best_index = int(np.argmax(scores))

    return {
        "predicted_class": str(class_names[best_index]),
        "top_score": round(float(scores[best_index]) * 100, 2),
        "class_scores": score_map,
        "model_note": (
            "These are uncalibrated model scores for a single 2D image slice. "
            "They are not diagnostic probabilities and must not guide care."
        )
    }
