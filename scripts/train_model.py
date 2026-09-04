import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

POSE_FEATURE_COUNT = 33 * 4  # extract_landmarks() puts pose first: 33 landmarks x (x, y, z, visibility)


def load_dataset(data_dir):
    """Load all saved sequences (hands-only, pose dropped) and their folder names as labels."""
    data_path = Path(data_dir)
    sequences = []
    labels = []

    for label_dir in sorted(path for path in data_path.iterdir() if path.is_dir()):
        for sequence_path in sorted(label_dir.glob("*.npy")):
            sequence = np.load(sequence_path)
            if sequence.ndim != 2:
                raise ValueError(f"{sequence_path} must have shape (frames, features), got {sequence.shape}")
            sequences.append(sequence[:, POSE_FEATURE_COUNT:].astype(np.float32))
            labels.append(label_dir.name)

    if not sequences:
        raise ValueError(f"No .npy sequences found in {data_path.resolve()}")

    sequence_shapes = {sequence.shape for sequence in sequences}
    if len(sequence_shapes) != 1:
        raise ValueError(f"All sequences must have the same shape; found {sorted(sequence_shapes)}")

    return np.asarray(sequences), np.asarray(labels)


def augment_sequence(sequence, rng):
    """Return a lightly perturbed copy: random time-warp (speed) plus small spatial jitter/scale."""
    frames, feature_count = sequence.shape

    speed = rng.uniform(0.85, 1.15)
    source_frames = max(2, int(round(frames * speed)))
    source_x = np.arange(frames)
    warped_source = np.stack(
        [np.interp(np.linspace(0, frames - 1, source_frames), source_x, sequence[:, feature]) for feature in range(feature_count)],
        axis=1,
    )
    target_x = np.arange(source_frames)
    warped = np.stack(
        [np.interp(np.linspace(0, source_frames - 1, frames), target_x, warped_source[:, feature]) for feature in range(feature_count)],
        axis=1,
    )

    reshaped = warped.reshape(frames, -1, 3)
    jitter_x, jitter_y = rng.normal(0, 0.01, size=2)
    scale = rng.uniform(0.97, 1.03)
    reshaped[:, :, 0] = reshaped[:, :, 0] * scale + jitter_x
    reshaped[:, :, 1] = reshaped[:, :, 1] * scale + jitter_y
    return reshaped.reshape(frames, feature_count).astype(np.float32)


def augment_dataset(features, labels, augment_factor, seed=42):
    """Expand a training set with augmented copies of each sequence (originals are always kept)."""
    if augment_factor < 1:
        return features, labels
    rng = np.random.default_rng(seed)
    augmented_features = [features]
    augmented_labels = [labels]
    for _ in range(augment_factor):
        augmented_features.append(np.stack([augment_sequence(sequence, rng) for sequence in features]))
        augmented_labels.append(labels)
    return np.concatenate(augmented_features), np.concatenate(augmented_labels)



def build_model(sequence_shape, class_count):
    """Create a small LSTM classifier for landmark sequences."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=sequence_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def main():
    parser = argparse.ArgumentParser(description="Train an LSTM sign-language classifier.")
    parser.add_argument("--data-dir", default="data", help="Directory containing one folder per sign label")
    parser.add_argument("--model-dir", default="models", help="Directory where the model and labels are saved")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training passes through the dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of sequences processed at once")
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=4,
        help="Extra time-warped/jittered copies to generate per training sequence (0 disables augmentation)",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    try:
        features, text_labels = load_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(text_labels)
    class_names = label_encoder.classes_.tolist()

    if len(class_names) < 2:
        parser.error("At least two sign folders are required before training")

    class_counts = np.bincount(labels)
    if np.min(class_counts) < 2:
        parser.error("Each sign needs at least two saved sequences for a train/test split")

    test_size = max(len(class_names), int(np.ceil(len(features) * 0.2)))
    if test_size >= len(features):
        parser.error("Collect more sequences so both train and test sets contain data")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    if args.augment_factor > 0:
        original_count = len(train_features)
        train_features, train_labels = augment_dataset(train_features, train_labels, args.augment_factor)
        print(f"Augmented training set: {original_count} -> {len(train_features)} sequences")

    model = build_model(features.shape[1:], len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    model.fit(
        train_features,
        train_labels,
        validation_data=(test_features, test_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Test accuracy: {test_accuracy:.2%}")

    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "sign_language_model.keras")
    with (model_path / "labels.json").open("w", encoding="utf-8") as labels_file:
        json.dump(class_names, labels_file, indent=2)

    print(f"Saved model to {model_path / 'sign_language_model.keras'}")
    print(f"Saved labels to {model_path / 'labels.json'}")


if __name__ == "__main__":
    main()
