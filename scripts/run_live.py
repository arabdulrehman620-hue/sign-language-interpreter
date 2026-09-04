import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic



def extract_landmarks(results):
    """Return landmarks in the same 258-value format used during training."""
    pose = np.array(
        [[landmark.x, landmark.y, landmark.z, landmark.visibility]
         for landmark in results.pose_landmarks.landmark]
        if results.pose_landmarks
        else np.zeros((33, 4)),
        dtype=np.float32,
    ).flatten()
    left_hand = np.array(
        [[landmark.x, landmark.y, landmark.z]
         for landmark in results.left_hand_landmarks.landmark]
        if results.left_hand_landmarks
        else np.zeros((21, 3)),
        dtype=np.float32,
    ).flatten()
    right_hand = np.array(
        [[landmark.x, landmark.y, landmark.z]
         for landmark in results.right_hand_landmarks.landmark]
        if results.right_hand_landmarks
        else np.zeros((21, 3)),
        dtype=np.float32,
    ).flatten()
    return np.concatenate((pose, left_hand, right_hand))



def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)



def load_labels(labels_path):
    with Path(labels_path).open("r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json must contain a non-empty JSON list")
    return labels


def extract_full_holistic_landmarks(results):
    """Return all 543 MediaPipe Holistic landmarks (face+left hand+pose+right hand) as (543, 3), NaN where missing.

    This matches the raw per-frame layout used by Kaggle's "Isolated Sign Language
    Recognition" competition dataset, which the bundled word_model.tflite expects.
    """

    def landmarks_to_array(landmark_list, count):
        if landmark_list is None:
            return np.full((count, 3), np.nan, dtype=np.float32)
        return np.array(
            [[point.x, point.y, point.z] for point in landmark_list.landmark],
            dtype=np.float32,
        )

    face = landmarks_to_array(results.face_landmarks, 468)
    left_hand = landmarks_to_array(results.left_hand_landmarks, 21)
    pose = landmarks_to_array(results.pose_landmarks, 33)
    right_hand = landmarks_to_array(results.right_hand_landmarks, 21)
    return np.concatenate([face, left_hand, pose, right_hand], axis=0)


def load_word_labels(labels_path):
    """Load the competition's word-to-index map (or a plain list) as an index-ordered word list."""
    labels_file = Path(labels_path)
    if not labels_file.exists():
        print(
            f"Warning: {labels_file} not found; using numbered placeholder labels. "
            "See models/word_model/README.md to fetch the real word list."
        )
        return [f"class_{i}" for i in range(250)]

    with labels_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        words = [None] * (max(data.values()) + 1)
        for word, index in data.items():
            words[index] = word
        return words
    raise ValueError(f"{labels_file} must contain a JSON list or a word-to-index object")


def make_word_predictor(signature_runner, labels):
    """Wrap the TFLite signature runner as predict(frames) -> (word, confidence)."""

    def predict(frames):
        outputs = signature_runner(inputs=np.asarray(frames, dtype=np.float32))["outputs"]
        probabilities = np.asarray(outputs).reshape(-1)
        index = int(np.argmax(probabilities))
        word = labels[index] if index < len(labels) else f"class_{index}"
        return word, float(probabilities[index])

    return predict


def load_tfjs_model(model_path):
    """Build the shared 63-input Dense model and load its binary weights."""
    with model_path.open("r", encoding="utf-8") as model_file:
        model_config = json.load(model_file)

    layers = model_config["modelTopology"]["model_config"]["config"]["layers"]
    dense_layers = [layer for layer in layers if layer["class_name"] == "Dense"]
    if len(dense_layers) != 3:
        raise ValueError("Expected three Dense layers in the TensorFlow.js model")

    input_size = layers[0]["config"]["batch_shape"][1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(dense_layers[0]["config"]["units"], activation="relu"),
            tf.keras.layers.Dropout(dense_layers[0].get("config", {}).get("rate", 0.3)),
            tf.keras.layers.Dense(dense_layers[1]["config"]["units"], activation="relu"),
            tf.keras.layers.Dropout(dense_layers[1].get("config", {}).get("rate", 0.3)),
            tf.keras.layers.Dense(dense_layers[2]["config"]["units"], activation="softmax"),
        ]
    )

    weights = []
    with model_path.with_name(model_config["weightsManifest"][0]["paths"][0]).open("rb") as weights_file:
        weight_data = np.frombuffer(weights_file.read(), dtype=np.float32)
    offset = 0
    for weight in model_config["weightsManifest"][0]["weights"]:
        size = int(np.prod(weight["shape"]))
        weights.append(weight_data[offset:offset + size].reshape(weight["shape"]))
        offset += size
    model.set_weights(weights)
    return model


def extract_hand_landmarks(results):
    """Return one hand as 63 values, matching the shared model's input."""
    hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
    if not hand_landmarks:
        return np.zeros(63, dtype=np.float32)
    return np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def get_wrist_x(results):
    """Return the visible wrist x-coordinate for wave detection."""
    hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
    return hand_landmarks.landmark[0].x if hand_landmarks else None


def is_wave(wrist_positions):
    """Detect several substantial left-right wrist direction changes."""
    positions = np.asarray(wrist_positions, dtype=np.float32)
    if len(positions) < 10 or positions.max() - positions.min() < 0.25:
        return False

    movement = np.diff(positions)
    movement = movement[np.abs(movement) > 0.015]
    if len(movement) < 5:
        return False
    direction_changes = np.sum(movement[1:] * movement[:-1] < 0)
    return direction_changes >= 3



def run_word_recognition_loop(cap, holistic, predict_fn, confidence_threshold, speech_engine):
    """Buffer frames while hands are visible, classify each pause-delimited clip as one sign,
    speak it immediately, then speak the whole sentence after a longer pause with no signing."""
    sign_end_pause = 0.4
    sentence_end_pause = 2.0
    min_frames_per_sign = 4

    buffer = []
    last_hand_time = 0.0
    last_speech_time = time.monotonic()
    transcript = ""
    current_label = "Waiting..."
    current_confidence = 0.0
    wrist_positions = deque(maxlen=20)
    last_wave_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_landmarks(frame, results)

        wrist_x = get_wrist_x(results)
        if wrist_x is not None:
            wrist_positions.append(wrist_x)
        else:
            wrist_positions.clear()

        if time.monotonic() - last_wave_time > 2 and is_wave(wrist_positions):
            transcript = ""
            buffer = []
            wrist_positions.clear()
            last_wave_time = time.monotonic()

        hand_visible = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        now = time.monotonic()

        if hand_visible:
            buffer.append(extract_full_holistic_landmarks(results))
            last_hand_time = now
            current_label = "Signing..."
        elif buffer:
            if now - last_hand_time >= sign_end_pause:
                if len(buffer) >= min_frames_per_sign:
                    word, confidence = predict_fn(np.stack(buffer))
                    current_label, current_confidence = word, confidence
                    if confidence >= confidence_threshold:
                        transcript += (" " if transcript else "") + word
                        if speech_engine is not None:
                            speech_engine.say(word)
                            speech_engine.runAndWait()
                        last_speech_time = now
                buffer = []
        elif transcript and now - last_speech_time >= sentence_end_pause:
            if speech_engine is not None:
                speech_engine.say(transcript.strip())
                speech_engine.runAndWait()
            transcript = ""
            current_label = "Waiting..."
            current_confidence = 0.0

        cv2.putText(frame, f"Text: {transcript[-40:]}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Sign: {current_label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(
            frame, f"Confidence: {current_confidence:.0%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Press q to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.imshow("Live Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and speech_engine is not None and transcript.strip():
            speech_engine.say(transcript.strip())
            speech_engine.runAndWait()
        if key == ord("q"):
            break


def run_keras_word_loop(cap, holistic, model, labels, confidence_threshold, speech_engine):
    """Capture exactly sequence_length consecutive frames the moment a hand appears (matching
    collect_data.py's recording convention), classify once, then require a brief cooldown before
    the next capture so trailing motion from the same sign doesn't immediately re-trigger."""
    sequence_length = model.input_shape[1]
    hands_only = model.input_shape[2] == 126
    sentence_end_pause = 2.0

    buffer = []
    capturing = False
    awaiting_hand_drop = False
    last_speech_time = time.monotonic()
    transcript = ""
    current_label = "Waiting..."
    current_confidence = 0.0
    wrist_positions = deque(maxlen=20)
    last_wave_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_landmarks(frame, results)

        wrist_x = get_wrist_x(results)
        if wrist_x is not None:
            wrist_positions.append(wrist_x)
        else:
            wrist_positions.clear()

        if time.monotonic() - last_wave_time > 2 and is_wave(wrist_positions):
            transcript = ""
            buffer = []
            capturing = False
            awaiting_hand_drop = False
            wrist_positions.clear()
            last_wave_time = time.monotonic()

        hand_visible = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        now = time.monotonic()
        landmarks = extract_landmarks(results)
        if hands_only:
            landmarks = landmarks[132:]

        if capturing:
            buffer.append(landmarks)
            current_label = f"Signing... ({len(buffer)}/{sequence_length})"
            if len(buffer) >= sequence_length:
                model_input = np.asarray(buffer, dtype=np.float32)[np.newaxis, ...]
                probabilities = model.predict(model_input, verbose=0)[0]
                label_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[label_index])
                current_label = labels[label_index]
                if current_confidence >= confidence_threshold:
                    transcript += (" " if transcript else "") + current_label
                    if speech_engine is not None:
                        speech_engine.say(current_label)
                        speech_engine.runAndWait()
                    last_speech_time = now
                buffer = []
                capturing = False
                awaiting_hand_drop = True
        elif awaiting_hand_drop:
            if not hand_visible:
                awaiting_hand_drop = False
                current_label = "Show a sign"
            else:
                current_label = "Lower your hand before the next sign"
        elif hand_visible:
            capturing = True
            buffer = [landmarks]
            current_label = "Signing..."
        elif transcript and now - last_speech_time >= sentence_end_pause:
            if speech_engine is not None:
                speech_engine.say(transcript.strip())
                speech_engine.runAndWait()
            transcript = ""
            current_label = "Waiting..."
            current_confidence = 0.0
        else:
            current_label = "Show a sign"

        cv2.putText(frame, f"Text: {transcript[-40:]}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Sign: {current_label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(
            frame, f"Confidence: {current_confidence:.0%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Press q to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.imshow("Live Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and speech_engine is not None and transcript.strip():
            speech_engine.say(transcript.strip())
            speech_engine.runAndWait()
        if key == ord("q"):
            break


def main():
    parser = argparse.ArgumentParser(description="Recognize trained signs from a live webcam feed.")
    parser.add_argument("--model", default="models/sign_language_model.keras", help="Saved Keras model path")
    parser.add_argument("--labels", default="models/labels.json", help="JSON file containing label names")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--confidence", type=float, default=0.8, help="Minimum confidence required for speech")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model not found: {model_path}")

    if model_path.suffix.lower() == ".tflite":
        labels_path = "models/word_model/labels.json" if args.labels == parser.get_default("labels") else args.labels
        labels = load_word_labels(labels_path)
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        predict_fn = make_word_predictor(interpreter.get_signature_runner("serving_default"), labels)

        try:
            speech_engine = pyttsx3.init()
        except Exception as error:
            speech_engine = None
            print(f"Warning: text-to-speech is unavailable ({error})")

        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Try --camera-index 1 or check the camera connection.")

        try:
            with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                run_word_recognition_loop(cap, holistic, predict_fn, args.confidence, speech_engine)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    try:
        labels = load_labels(args.labels)
        if model_path.suffix.lower() == ".json":
            model = load_tfjs_model(model_path)
            uses_sequence = False
        else:
            model = tf.keras.models.load_model(model_path)
            uses_sequence = True
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"Could not load model or labels: {error}")

    sequence_length = model.input_shape[1] if uses_sequence else 1
    feature_count = model.input_shape[2] if uses_sequence else model.input_shape[1]
    if sequence_length is None or feature_count is None:
        parser.error("The model must have fixed input dimensions")
    if uses_sequence and feature_count not in (258, 126):
        parser.error(
            f"This runner expects 258 (pose+hands) or 126 (hands-only) features per frame, "
            f"but the model expects {feature_count}"
        )
    if not uses_sequence and feature_count != 63:
        parser.error(f"The TensorFlow.js model expects 63 hand features, but the model expects {feature_count}")
    if len(labels) != model.output_shape[-1]:
        parser.error("The number of labels does not match the model output size")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")

    try:
        speech_engine = pyttsx3.init()
    except Exception as error:
        speech_engine = None
        print(f"Warning: text-to-speech is unavailable ({error})")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera-index 1 or check the camera connection.")

    if uses_sequence:
        try:
            with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                run_keras_word_loop(cap, holistic, model, labels, args.confidence, speech_engine)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    frames = deque(maxlen=sequence_length)
    wrist_positions = deque(maxlen=20)
    last_wave_time = 0.0
    transcript = ""
    last_processed_label = None
    current_label = "Waiting..."
    current_confidence = 0.0

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to grab frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                wrist_x = get_wrist_x(results)
                if wrist_x is not None:
                    wrist_positions.append(wrist_x)
                else:
                    wrist_positions.clear()

                if time.monotonic() - last_wave_time > 2 and is_wave(wrist_positions):
                    transcript = ""
                    last_processed_label = None
                    wrist_positions.clear()
                    last_wave_time = time.monotonic()

                frames.append(extract_hand_landmarks(results))
                draw_landmarks(frame, results)

                if len(frames) == sequence_length:
                    model_input = np.expand_dims(np.asarray(frames), axis=0)[:, 0, :]
                    probabilities = model.predict(model_input, verbose=0)[0]
                    label_index = int(np.argmax(probabilities))
                    current_confidence = float(probabilities[label_index])
                    current_label = labels[label_index]

                    if current_confidence >= args.confidence:
                        if current_label != last_processed_label:
                            if current_label == "space":
                                transcript += " "
                                completed_word = transcript.strip().split()[-1] if transcript.strip() else ""
                                if speech_engine is not None and completed_word:
                                    speech_engine.say(completed_word)
                                    speech_engine.runAndWait()
                            elif current_label == "del":
                                transcript = transcript[:-1]
                            else:
                                transcript += current_label
                            last_processed_label = current_label
                    else:
                        last_processed_label = None

                cv2.putText(
                    frame,
                    f"Text: {transcript[-40:]}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Sign: {current_label}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {current_confidence:.0%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press q to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("Live Sign Language Recognition", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 13 and speech_engine is not None and transcript.strip():
                    speech_engine.say(transcript.strip())
                    speech_engine.runAndWait()
                if key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
