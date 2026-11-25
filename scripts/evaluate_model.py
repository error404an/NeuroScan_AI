from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "dataset" / "Brain_Tumor_MRI_Dataset_Final" / "test"
MODEL_PATH = ROOT / "models" / "brain_mri_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"

def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder missing: {TEST_DIR}")
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        raise FileNotFoundError("Train the model first using scripts/train_model.py")

    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=class_names, image_size=(160, 160),
        batch_size=16, label_mode="int", shuffle=False
    )
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("These metrics apply only to this dataset split, not clinical performance.")

if __name__ == "__main__":
    main()
