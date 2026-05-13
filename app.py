import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Focus Detection System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Human Focus Detection System")
st.markdown(
    """
    Upload an image and the system will:
    - Detect humans
    - Detect faces
    - Estimate whether each person is looking at the camera
    - Draw:
        - 🟩 Green = Focused
        - 🟥 Red = Not Focused
    """
)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt", task="detect")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    return yolo_model, face_mesh


yolo_model, face_mesh = load_models()

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def is_looking_forward(landmarks, img_w, img_h):
    """
    Estimate if person is looking forward using eye symmetry.
    """

    # Landmark indices
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1

    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    nose = landmarks[NOSE_TIP]

    left_x = left_eye.x * img_w
    right_x = right_eye.x * img_w
    nose_x = nose.x * img_w

    # Eye center
    eye_center = (left_x + right_x) / 2

    # Nose deviation
    deviation = abs(nose_x - eye_center)

    # Threshold
    threshold = (right_x - left_x) * 0.15

    return deviation < threshold


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert RGB -> BGR
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # YOLO HUMAN DETECTION
    # -----------------------------
    results = yolo_model(img)

    person_boxes = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                person_boxes.append((x1, y1, x2, y2))

    # -----------------------------
    # FACE MESH PROCESSING
    # -----------------------------
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_img)

    face_data = []

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            # Bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]

            x_min = int(min(xs) * img.shape[1])
            y_min = int(min(ys) * img.shape[0])
            x_max = int(max(xs) * img.shape[1])
            y_max = int(max(ys) * img.shape[0])

            focused = is_looking_forward(
                face_landmarks.landmark,
                img.shape[1],
                img.shape[0]
            )

            face_data.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "focused": focused
            })

    # -----------------------------
    # MATCH FACE TO PERSON
    # -----------------------------
    for (px1, py1, px2, py2) in person_boxes:

        status = False

        for face in face_data:
            fx1, fy1, fx2, fy2 = face["bbox"]

            # Check if face center inside person box
            cx = (fx1 + fx2) // 2
            cy = (fy1 + fy2) // 2

            if px1 < cx < px2 and py1 < cy < py2:
                status = face["focused"]
                break

        # Green if focused
        if status:
            color = (0, 255, 0)
            label = "Focused"
        else:
            color = (0, 0, 255)
            label = "Not Focused"

        # Draw rectangle
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 3)

        # Label background
        cv2.rectangle(
            img,
            (px1, py1 - 35),
            (px1 + 170, py1),
            color,
            -1
        )

        # Put text
        cv2.putText(
            img,
            label,
            (px1 + 10, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Convert back BGR -> RGB
    final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(
        final_img,
        caption="Processed Image",
        use_container_width=True
    )

    st.success("Processing Complete ✅")
