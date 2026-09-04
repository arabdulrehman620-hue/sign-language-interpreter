import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_labels(labels_path):
    """Load the competition's word-to-index map (or a plain list) as an index-ordered word list."""
    labels_file = Path(labels_path)
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found; using numbered placeholder labels.")
        return [f"class_{i}" for i in range(250)]
    with labels_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    words = [None] * (max(data.values()) + 1)
    for word, index in data.items():
        words[index] = word
    return words


def frames_to_array(frames):
    """Convert JSON-decoded nested lists (null for missing landmarks) into a NaN-filled float32 array."""
    return np.array(
        [[[np.nan if value is None else value for value in point] for point in frame] for frame in frames],
        dtype=np.float32,
    )


def make_handler(predict_fn, labels):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self):
            if self.path != "/predict":
                self.send_response(404)
                self._cors()
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                frames = frames_to_array(body["frames"])
                if frames.ndim != 3 or frames.shape[1:] != (543, 3):
                    raise ValueError(f"Expected frames shaped (n, 543, 3), got {frames.shape}")
                probabilities = np.asarray(predict_fn(inputs=frames)["outputs"]).reshape(-1)
                index = int(np.argmax(probabilities))
                word = labels[index] if index < len(labels) else f"class_{index}"
                response = json.dumps({"word": word, "confidence": float(probabilities[index])}).encode("utf-8")
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except Exception as error:  # noqa: BLE001
                message = json.dumps({"error": str(error)}).encode("utf-8")
                self.send_response(400)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)

        def log_message(self, format_string, *args):  # noqa: A002
            print(f"[serve_word_model] {self.address_string()} - {format_string % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve the word-sign TFLite model over a local HTTP API.")
    parser.add_argument("--model", default="models/word_model/word_model.tflite", help="Path to the .tflite model")
    parser.add_argument("--labels", default="models/word_model/labels.json", help="Path to the labels JSON file")
    parser.add_argument("--port", type=int, default=8001, help="Port to serve the /predict endpoint on")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    predict_fn = interpreter.get_signature_runner("serving_default")

    server = ThreadingHTTPServer(("localhost", args.port), make_handler(predict_fn, labels))
    print(f"Word model server ready: POST clips to http://localhost:{args.port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
