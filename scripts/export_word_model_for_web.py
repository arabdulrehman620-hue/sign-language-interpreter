import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

LAYER_TYPE_MAP = {
    "LSTM": "lstm",
    "Dense": "dense",
    "Dropout": "dropout",
}


def describe_layer(layer):
    layer_type = LAYER_TYPE_MAP.get(layer.__class__.__name__)
    if layer_type == "lstm":
        return {"type": "lstm", "units": layer.units, "returnSequences": layer.return_sequences, "activation": layer.activation.__name__}
    if layer_type == "dense":
        return {"type": "dense", "units": layer.units, "activation": layer.activation.__name__}
    return None  # Dropout and other weight-free layers are skipped for inference-only export


def main():
    parser = argparse.ArgumentParser(description="Export a Keras sign-word model to a browser-loadable format.")
    parser.add_argument("--model", default="models/sign_language_model.keras", help="Path to the trained Keras model")
    parser.add_argument("--labels", default="models/labels.json", help="Path to the labels JSON file")
    parser.add_argument("--output-dir", default="models/custom_words_tfjs", help="Directory to write manifest.json + weights.bin")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.labels, encoding="utf-8") as labels_file:
        labels = json.load(labels_file)

    layers = [description for layer in model.layers if (description := describe_layer(layer)) is not None]
    weights = [weight for layer in model.layers if describe_layer(layer) is not None for weight in layer.get_weights()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_bytes = b"".join(weight.astype(np.float32).tobytes() for weight in weights)
    (output_dir / "weights.bin").write_bytes(weights_bytes)

    manifest = {
        "sequenceLength": model.input_shape[1],
        "featureCount": model.input_shape[2],
        "labels": labels,
        "layers": layers,
        "weights": [{"shape": list(weight.shape)} for weight in weights],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported {len(weights)} weight tensors ({len(weights_bytes)} bytes) to {output_dir}")


if __name__ == "__main__":
    main()
