import cv2
import mediapipe as mp

# We import MediaPipe drawing utilities so we can draw landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# We import the Holistic solution, which includes body pose + left hand + right hand detection.
mp_holistic = mp.solutions.holistic


def main():
    """Open webcam and draw MediaPipe landmarks in real time."""

    # Open the default webcam (index 0). If your camera is not the default one,
    # you may need to change this number to 1, 2, etc.
    cap = cv2.VideoCapture(0)

    # If the webcam fails to open, tell the user and stop the script.
    if not cap.isOpened():
        print("Error: Could not open webcam. Try changing the camera index or checking your camera connection.")
        return

    # Create a MediaPipe Holistic model.
    # - static_image_mode=False means we process live video frames continuously.
    # - model_complexity=1 is a good default for speed and accuracy.
    # - min_detection_confidence controls how confident the model must be before accepting a detection.
    # - min_tracking_confidence controls how confident tracking stays stable across frames.
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            # Read one frame from the webcam.
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from webcam.")
                break

            # Mirror the image so the user sees a natural camera view.
            frame = cv2.flip(frame, 1)

            # Convert the frame from BGR (OpenCV default) to RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe Holistic on this frame.
            # It returns landmark data for pose, left hand, right hand, and face.
            results = holistic.process(rgb_frame)

            # Draw the detected landmarks over the original BGR frame.
            # This helps us visually confirm that MediaPipe is working.
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
            )
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
            )

            # Show the frame with overlays.
            cv2.imshow("Webcam + MediaPipe Holistic", frame)

            # Press 'q' to quit the app.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
