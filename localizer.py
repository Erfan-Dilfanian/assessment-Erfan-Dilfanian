import cv2
import json
import numpy as np

BIN_DIAMETER_M = 0.40
BIN_HEIGHT_M = 0.65


def load_calib(path: str):
    """
    Load camera calibration and mount geometry from calib.json.
    Returns:
        K         (3x3 ndarray)  camera intrinsic matrix
        D         (5,) ndarray   distortion coefficients [k1,k2,p1,p2,k3]
        cam_h     (float)        camera height above ground, metres
        tilt_rad  (float)        downward tilt angle, radians
    """
    with open(path) as f:
        c = json.load(f)

    K = np.array(c["K"], dtype=np.float64)
    D = np.array(c["dist_coeffs"], dtype=np.float64)
    cam_h = float(c["camera_height_m"])
    tilt_rad = float(np.deg2rad(c["camera_tilt_deg"]))

    return K, D, cam_h, tilt_rad


def build_extrinsic(cam_h: float, tilt_rad: float):
    """
    Build camera-to-world extrinsics such that:

        P_world = R @ P_cam + t

    World frame:
        origin at pole base on the ground
        +X = forward on the ground
        +Y = left
        +Z = up

    Camera frame (OpenCV):
        +X = right
        +Y = down
        +Z = forward
    """
    alpha = abs(tilt_rad)

    # Level camera -> chosen world frame
    R0 = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0]
    ], dtype=np.float64)

    c = np.cos(alpha)
    s = np.sin(alpha)

    # Downward pitch about world Y axis
    Ry = np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=np.float64)

    R = Ry @ R0
    t = np.array([0.0, 0.0, cam_h], dtype=np.float64)

    return R, t


def cam_to_world(xyz_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R @ xyz_cam + t


def estimate_3d(bbox: tuple, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Estimate the 3D position of the bin centroid in the CAMERA frame.

    Args:
        bbox  (x1, y1, x2, y2) pixel bounding box
        K     camera intrinsic matrix (3x3)
        D     distortion coefficients

    Returns:
        xyz_cam  (3,) ndarray [x_cam, y_cam, z_cam] in metres

    Derivation:
        Let H be the known physical bin height and h_px the bbox height in pixels.

            h_px / fy = H / Z
            => Z = fy * H / h_px

        Let (u, v) be the bbox center. After undistorting this point, OpenCV
        returns normalized image coordinates (x_n, y_n), where:

            x_n = X / Z
            y_n = Y / Z

        Hence:

            X = x_n * Z
            Y = y_n * Z
    """
    x1, y1, x2, y2 = bbox

    fy = K[1, 1]
    bbox_h = max(float(y2 - y1), 1.0)

    # Depth along optical axis
    Z = fy * BIN_HEIGHT_M / bbox_h

    # Bbox center pixel
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    # Undistort center pixel into normalized image coordinates
    pts = np.array([[[u, v]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, K, D)

    x_n = undist[0, 0, 0]
    y_n = undist[0, 0, 1]

    X = x_n * Z
    Y = y_n * Z

    return np.array([X, Y, Z], dtype=np.float64)


def camera_distance_from_xyz(xyz_cam: np.ndarray) -> float:
    """
    Euclidean distance from camera origin to bin centroid.
    """
    return float(np.linalg.norm(xyz_cam))