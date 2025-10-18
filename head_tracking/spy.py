import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

CAM_INDEX = 0
H_FOV_DEG = 60.0  # do zmiany jesli skala sie nie zgadza

# kamera przesunieta o okolo 10 cm w gore wzgledem srodka ekranu
CAMERA_TO_SCREEN_CENTER_M = np.array([0.0, 0.10, 0.0], dtype=np.float64)

# FaceMesh config
FACE_MESH_STATIC_IMAGE_MODE = False
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINEMENT = True
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.6
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.6

# Smoothing
ONE_EURO_MINCUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_DCUT = 1.0
EMA_ALPHA = 0.1

TEXT_SCALE = 0.7
TEXT_THICK = 2

# ---------------------------- One Euro Filter ----------------------------------

class LowPass:
    def __init__(self, alpha: float, init: Optional[np.ndarray] = None):
        self.alpha = float(alpha)
        self.prev = None if init is None else np.array(init, dtype=np.float64)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float64)
        if self.prev is None:
            self.prev = x
            return x
        self.prev = self.prev + self.alpha * (x - self.prev)
        return self.prev

def smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * dt
    return r / (r + 1.0)

class OneEuro:

    def __init__(self, freq: float, mincutoff=1.0, beta=0.0, dcutoff=1.0, init: Optional[np.ndarray] = None):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_filter = LowPass(alpha=0.0, init=init)
        self.dx_filter = LowPass(alpha=0.0, init=np.zeros_like(init) if init is not None else None)
        self.last_time = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.array(x, dtype=np.float64)
        if self.last_time is None:
            self.last_time = t
            self.x_filter.prev = x
            if self.dx_filter.prev is None:
                self.dx_filter.prev = np.zeros_like(x)
            return x

        dt = max(1e-6, t - self.last_time)
        self.last_time = t

        dx = (x - self.x_filter.prev) / dt
        ad = smoothing_factor(dt, self.dcutoff)
        self.dx_filter.alpha = ad
        dx_hat = self.dx_filter.apply(dx)

        cutoff = self.mincutoff + self.beta * np.linalg.norm(dx_hat)
        a = smoothing_factor(dt, cutoff)
        self.x_filter.alpha = a
        return self.x_filter.apply(x)

# ---------------------------- Head Tracker ------------------------------------

@dataclass
class PoseResult:
    pos_m: np.ndarray      # srodek ekranu (metry), +X right, +Y up, +Z toward user
    cam_tvec_m: np.ndarray # kamera (metry), +X right, +Y down, +Z forward
    rvec: np.ndarray
    success: bool

class HeadTracker:
    def __init__(self, cam_index: int = CAM_INDEX, h_fov_deg: float = H_FOV_DEG):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cam_index}")

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read from camera.")
        self.h, self.w = frame.shape[:2]

        self.K = self._estimate_camera_matrix(self.w, self.h, h_fov_deg)
        self.dist = np.zeros((5, 1), dtype=np.float64)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=FACE_MESH_STATIC_IMAGE_MODE,
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=FACE_MESH_REFINEMENT,
            min_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )

        # model 3D  (milimetry): nosek, broda, oczy rogi, usta rogi
        self.model_points_mm = np.array([
            [0.0,     0.0,     0.0],     # Nose tip
            [0.0,   -63.6,   -12.5],     # Chin
            [-43.3,  32.7,   -26.0],     # Left eye outer
            [ 43.3,  32.7,   -26.0],     # Right eye outer
            [-28.9, -28.9,   -24.1],     # Mouth left
            [ 28.9, -28.9,   -24.1],     # Mouth right
        ], dtype=np.float64)

        self.idx = {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_outer": 33,
            "right_eye_outer": 263,
            "mouth_left": 61,
            "mouth_right": 291,
            "left_eye_inner": 133,
            "right_eye_inner": 362,
        }

        self.pos_filter = OneEuro(freq=120.0, mincutoff=ONE_EURO_MINCUTOFF,
                                  beta=ONE_EURO_BETA, dcutoff=ONE_EURO_DCUT, init=np.zeros(3))
        self.ema = LowPass(alpha=EMA_ALPHA, init=np.zeros(3))

        self.start_time = time.perf_counter()
        self.last_pos_screen_m = np.zeros(3)

    def get_head_position_cm(self) -> Tuple[float, float, float]:
        return tuple((self.last_pos_screen_m * 100.0).round(2))

    @staticmethod
    def _estimate_camera_matrix(w: int, h: int, h_fov_deg: float) -> np.ndarray:
        f = (w / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
        cx, cy = w / 2.0, h / 2.0
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0,  1]], dtype=np.float64)

    def _landmarks_to_points(self, landmarks) -> Optional[np.ndarray]:
        if landmarks is None:
            return None
        lm = landmarks.landmark
        try:
            pts = np.array([
                [lm[self.idx["nose_tip"]].x * self.w,        lm[self.idx["nose_tip"]].y * self.h],
                [lm[self.idx["chin"]].x * self.w,            lm[self.idx["chin"]].y * self.h],
                [lm[self.idx["left_eye_outer"]].x * self.w,  lm[self.idx["left_eye_outer"]].y * self.h],
                [lm[self.idx["right_eye_outer"]].x * self.w, lm[self.idx["right_eye_outer"]].y * self.h],
                [lm[self.idx["mouth_left"]].x * self.w,      lm[self.idx["mouth_left"]].y * self.h],
                [lm[self.idx["mouth_right"]].x * self.w,     lm[self.idx["mouth_right"]].y * self.h],
            ], dtype=np.float64)
            return pts
        except Exception:
            return None

    def _between_eyes(self, landmarks) -> Optional[Tuple[int, int]]:
        try:
            lm = landmarks.landmark
            lx, ly = lm[self.idx["left_eye_inner"]].x * self.w, lm[self.idx["left_eye_inner"]].y * self.h
            rx, ry = lm[self.idx["right_eye_inner"]].x * self.w, lm[self.idx["right_eye_inner"]].y * self.h
            return int((lx + rx) / 2.0), int((ly + ry) / 2.0)
        except Exception:
            return None

    def _solve_pose(self, img_pts_px: np.ndarray) -> PoseResult:

        img_pts_px = np.asarray(img_pts_px, dtype=np.float64).reshape(-1, 2)
        model_pts = np.asarray(self.model_points_mm, dtype=np.float64).reshape(-1, 3)
        if img_pts_px.shape[0] != model_pts.shape[0] or img_pts_px.shape[0] < 4:
            return PoseResult(self.last_pos_screen_m, self.last_pos_screen_m, np.zeros(3), False)

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=model_pts,
            imagePoints=img_pts_px,
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return PoseResult(self.last_pos_screen_m, self.last_pos_screen_m, np.zeros(3), False)

        # mm -> m
        tvec_m_cam = (tvec.reshape(3) / 1000.0).astype(np.float64)

        # przeniesienie do srodka ekranu i flip Y
        head_cam_rel_screen = tvec_m_cam - CAMERA_TO_SCREEN_CENTER_M
        head_screen = np.array([head_cam_rel_screen[0], -head_cam_rel_screen[1], head_cam_rel_screen[2]], dtype=np.float64)

        return PoseResult(pos_m=head_screen, cam_tvec_m=tvec_m_cam, rvec=rvec.reshape(3), success=True)

    def _smooth(self, pos_m: np.ndarray, t: float) -> np.ndarray:
        filtered = self.pos_filter(pos_m, t)
        if EMA_ALPHA > 0.0:
            filtered = self.ema.apply(filtered)
        return filtered

    def run(self):
        print("Head tracker running. Press 'q' to quit.")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = self.mesh.process(rgb)
            pose = None
            between = None

            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0]
                img_pts = self._landmarks_to_points(landmarks)
                between = self._between_eyes(landmarks)
                if img_pts is not None:
                    pose = self._solve_pose(img_pts)

            now = time.perf_counter() - self.start_time

            if pose and pose.success:
                pos_m = self._smooth(pose.pos_m, now)
                self.last_pos_screen_m = pos_m
                txt = f"[x,y,z] = {pos_m[0]*100:.1f} cm, {pos_m[1]*100:.1f} cm, {pos_m[2]*100:.1f} cm"
                cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICK, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Face not found...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), TEXT_THICK, cv2.LINE_AA)

            if between is not None:
                cv2.circle(frame, between, 6, (255, 255, 255), 2)
                cv2.circle(frame, between, 2, (0, 255, 255), -1)

            cx, cy = self.w // 2, self.h // 2
            cv2.drawMarker(frame, (cx, cy), (128, 128, 128), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)


            cv2.imshow("Head Tracker (screen-centered)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ---------------------------- Debug -------------------------------

def main():
    tracker = HeadTracker(cam_index=CAM_INDEX, h_fov_deg=H_FOV_DEG)
    tracker.run()
    x_cm, y_cm, z_cm = tracker.get_head_position_cm()
    print(f"Final head position (screen-centered): x={x_cm} cm, y={y_cm} cm, z={z_cm} cm")
