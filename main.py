# pip install opencv-python mediapipe
import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (Mediapipe Face Mesh, iris-inclusive subset)
# Right eye outer/inner corners and iris center approximation
RIGHT_EYE = [33, 133]         # outer, inner corners (approx)
RIGHT_IRIS = [468, 469, 470, 471]  # iris ring points (iris center ~ mean)
LEFT_EYE = [263, 362]
LEFT_IRIS = [473, 474, 475, 476]

def iris_center(landmarks, idxs, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)  # rough iris center


capture = 0

def draw_grid_points(frame, w, h):
    """Draw 9 predefined points in a 3x3 grid on the frame."""
    grid_rows, grid_cols = 3, 3
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    grid_w = w - 2 * margin_x
    grid_h = h - 2 * margin_y
    for i in range(grid_rows):
        for j in range(grid_cols):
            x = margin_x + int(j * grid_w / (grid_cols - 1))
            y = margin_y + int(i * grid_h / (grid_rows - 1))
            if i == capture // 3 and j == capture % 3:
                cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as fm:
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark

            # Right eye
            r_center = iris_center(lms, RIGHT_IRIS, w, h).astype(int)
            # Left eye
            l_center = iris_center(lms, LEFT_IRIS, w, h).astype(int)

            # Display
            cv2.circle(frame, tuple(r_center), 3, (0,255,0), -1)
            cv2.circle(frame, tuple(l_center), 3, (0,255,0), -1)

            key = cv2.waitKey(1)
            if key == ord('r'):
                print(f"x: {r_center[0]}, y: {r_center[1]}")
                r_raw = np.array([[lms[i].x, lms[i].y] for i in RIGHT_IRIS])
                print(f"Right iris center: {r_raw.mean(axis=0)}")
            if key == ord('c'):
                capture = (capture + 1) % 9
            if key & 0xFF == 27:  # ESC
                break
                
        draw_grid_points(frame, w, h)
        cv2.imshow("Basic Eye Tracking", frame)
        

cap.release()
cv2.destroyAllWindows()
