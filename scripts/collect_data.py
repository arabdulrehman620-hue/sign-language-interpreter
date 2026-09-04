import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """Return pose and hand landmarks as one fixed-size feature vector."""
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


def collect_action(action, output_dir, sequences, sequence_length, camera_index):
    action_dir = Path(output_dir) / action
    action_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera-index 1 or check the camera connection.")

    print(f"Action: {action}")
    print("Press s to start the next sequence, or q to quit.")

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            sequence_number = 0
            while sequence_number < sequences:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"{action}: {sequence_number}/{sequences} | Press s to record",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Sign Language Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return
                if key != ord("s"):
                    continue

                sequence = []
                for frame_number in range(sequence_length):
                    success, frame = cap.read()
                    if not success:
                        break

                    frame = cv2.flip(frame, 1)
                    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    sequence.append(extract_landmarks(results))
                    draw_landmarks(frame, results)
                    cv2.putText(
                        frame,
                        f"Recording {sequence_number + 1}/{sequences}: {frame_number + 1}/{sequence_length}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Sign Language Data Collector", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                if len(sequence) == sequence_length:
                    np.save(action_dir / f"sequence_{sequence_number:04d}.npy", np.asarray(sequence))
                    sequence_number += 1
                    print(f"Saved {action_dir / f'sequence_{sequence_number - 1:04d}.npy'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Collect pose and hand landmark sequences for sign-language actions.")
    parser.add_argument("action", help="Action label, for example hello or thank_you")
    parser.add_argument("--output-dir", default="data", help="Directory where action sequences are saved")
    parser.add_argument("--sequences", type=int, default=30, help="Number of sequences to collect")
    parser.add_argument("--sequence-length", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    args = parser.parse_args()

    if args.sequences < 1 or args.sequence_length < 1:
        parser.error("--sequences and --sequence-length must be positive")

    collect_action(
        args.action,
        args.output_dir,
        args.sequences,
        args.sequence_length,
        args.camera_index,
    )


if __name__ == "__main__":
    main()