import streamlit as st
import cv2
import threading
import tempfile
import os
from deepface import DeepFace
from PIL import Image
import numpy as np

# Set page to wide mode
st.set_page_config(layout="wide")


def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)


def check_face(frame, reference):
    try:
        result = DeepFace.verify(frame, reference.copy())
        return result['verified']
    except ValueError:
        return False


def main():
    st.title("Face Recognition System")

    # Create two columns
    left_column, right_column = st.columns([1, 1])  # Right column slightly wider for camera feed

    with left_column:
        # File uploader for reference image
        st.subheader("Upload Reference Image")
        reference_file = st.file_uploader("Choose a reference image", type=['png', 'jpg', 'jpeg'])

        if reference_file is not None:
            # Display reference image
            reference_image = load_image(reference_file)
            st.image(reference_image, caption="Reference Image", use_column_width=True)

            # Controls
            run = st.checkbox('Start/Stop Camera')

    with right_column:
        st.subheader("Live Camera Feed")
        # Status indicator moved here
        status_text = st.empty()
        # Placeholder for webcam feed
        frame_window = st.image([])

        if reference_file is not None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            counter = 0
            match = False

            while run:
                ret, frame = camera.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Check face every 30 frames
                    if counter % 30 == 0:
                        match = check_face(frame, reference_image)

                    counter += 1

                    # Display status
                    if match:
                        status_text.markdown('### Status: MATCH! ✅')
                        # Draw green text on frame
                        frame = cv2.putText(frame, "Match!", (20, 450),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        status_text.markdown('### Status: NO MATCH ❌')
                        # Draw red text on frame
                        frame = cv2.putText(frame, "No Match!", (20, 450),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                    # Update frame in UI - use column width
                    frame_window.image(frame, use_column_width=True)

                else:
                    st.error("Failed to access webcam")
                    break

            if not run:
                camera.release()

        else:
            st.info("Please upload a reference image to start")


if __name__ == '__main__':
    main()