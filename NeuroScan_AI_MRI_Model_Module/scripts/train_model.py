from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "Brain_Tumor_MRI_Dataset_Final"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 15
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]

def main():
    for folder in (TRAIN_DIR, VAL_DIR):
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder missing: {folder}")
        found = {p.name.lower() for p in folder.iterdir() if p.is_dir()}
        if found != set(CLASS_NAMES):
            raise ValueError(f"{folder}: expected folders {CLASS_NAMES}, found {sorted(found)}")

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=CLASS_NAMES, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="int", shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, class_names=CLASS_NAMES, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="int", shuffle=False
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08)
    ])

    model = models.Sequential([
        layers.Input(shape=(160, 160, 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(192, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(96, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_DIR / "brain_mri_model.keras",
        monitor="val_loss", save_best_only=True
    )

    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )

    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
    )
    print("Training complete. Model saved in:", MODEL_DIR)

if __name__ == "__main__":
    main()
