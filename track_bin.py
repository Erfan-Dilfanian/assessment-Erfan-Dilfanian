"""
Skyscouter – Computer Vision Engineer Technical Assessment
Skeleton file: track_bin.py
You are free to restructure this entirely — it is a starting point only.
"""

import cv2
import json
import argparse
import time
import numpy as np


# ---------------------------------------------------------------------------
# Known target dimensions
# ---------------------------------------------------------------------------

BIN_DIAMETER_M = 0.40   # standard outdoor garbage bin, metres
BIN_HEIGHT_M   = 0.65


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_calib(path: str):
    """
    Load camera intrinsics and mount geometry from calib.json.
    Returns:
        K         (3x3 ndarray)  camera intrinsic matrix
        D         (5,  ndarray)  distortion coefficients [k1,k2,p1,p2,k3]
        cam_h     (float)        camera height above ground, metres
        tilt_rad  (float)        downward tilt angle, radians
    """
    with open(path) as f:
        c = json.load(f)
    K        = np.array(c["K"], dtype=np.float64)
    D        = np.array(c["dist_coeffs"], dtype=np.float64)
    cam_h    = float(c["camera_height_m"])
    tilt_rad = float(np.deg2rad(c["camera_tilt_deg"]))
    return K, D, cam_h, tilt_rad


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def build_extrinsic(cam_h: float, tilt_rad: float):
    """
    Build the rotation matrix R and translation vector t mapping
    a point in the CAMERA frame to the WORLD frame.

    World frame: origin at pole base on the ground.
    World +X: forward (optical axis projected to ground)
    World +Y: left
    World +Z: up

    Camera frame: standard OpenCV (Z forward, X right, Y down)

    TODO: implement. Show your derivation in the README.

    Returns:
        R  (3x3 ndarray)
        t  (3,  ndarray)  such that P_world = R @ P_cam + t
    """
    raise NotImplementedError("TODO: implement build_extrinsic")


def cam_to_world(xyz_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R @ xyz_cam + t


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

from detector import load_detector, detect_bin


def load_detector_(use_gpu: bool = False):
    """
    Load and return your detector.
    If use_gpu=True, configure your model to run on GPU.

    README requirement: if use_gpu=True, you must document the GPU model,
    VRAM, and CUDA version used during your testing.
    """
    raise NotImplementedError("TODO: load your detector")


def detect_bin_(frame: np.ndarray, model) -> tuple | None:
    """
    Detect the garbage bin in a single BGR frame.

    Returns:
        (x1, y1, x2, y2, confidence) as floats, or None if not detected.
        Coordinates in pixels, top-left to bottom-right.

    Note: standard COCO-pretrained models (YOLOv8, etc.) include a
    'trash can' / 'garbage bin' class — document whether you use it
    directly or fine-tune, and why.
    """
    raise NotImplementedError("TODO: implement detect_bin")


# ---------------------------------------------------------------------------
# 3D localisation (monocular, known object size)
# ---------------------------------------------------------------------------

def estimate_3d(
    bbox: tuple,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Estimate the 3D position of the bin centroid in the CAMERA frame.

    Args:
        bbox  (x1, y1, x2, y2) pixel bounding box
        K     camera intrinsic matrix (3x3)
        D     distortion coefficients

    Returns:
        xyz_cam  (3,) ndarray [x_cam, y_cam, z_cam] in metres

    Approach hint (show this derivation in README):
        Use the known bin height or diameter and the projected pixel span
        to estimate depth Z via the pinhole equation:
            Z = f_y * BIN_HEIGHT_M / bbox_pixel_height
        Then recover X, Y from the undistorted image-plane centroid:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
    """
    raise NotImplementedError("TODO: implement estimate_3d")


# ---------------------------------------------------------------------------
# Optional: Kalman filter wrapper  [bonus task]
# ---------------------------------------------------------------------------

class PositionKalman:
    """
    Constant-velocity Kalman filter over 3D world position.
    State vector: [x, y, z, vx, vy, vz]

    TODO: implement if attempting the Kalman bonus task.
    Explain your Q, R, and state vector choices in the README.
    """

    def __init__(self):
        raise NotImplementedError("TODO: implement PositionKalman")

    def update(self, xyz_meas: np.ndarray) -> np.ndarray:
        """Accept a measurement, return filtered estimate."""
        raise NotImplementedError

    def predict(self) -> np.ndarray:
        """Return predicted position (called during occlusion frames)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Skyscouter garbage bin tracker")
    parser.add_argument("--video",   required=True, help="Path to input.mp4")
    parser.add_argument("--calib",   required=True, help="Path to calib.json")
    parser.add_argument("--output",  default="results/output.csv")
    parser.add_argument("--gpu",     action="store_true",
                        help="Run detector on GPU (document GPU specs in README)")
    parser.add_argument("--kalman",  action="store_true",
                        help="Enable Kalman filter smoothing (bonus)")
    args = parser.parse_args()

    K, D, cam_h, tilt_rad = load_calib(args.calib)
    # R, t = build_extrinsic(cam_h, tilt_rad)
    model = load_detector(use_gpu=args.gpu)

    if args.gpu:
        print("[INFO] Running detector on GPU — ensure GPU model is documented in README")

    kf = PositionKalman() if args.kalman else None

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    trajectory  = []
    last_known  = None
    last_age    = 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    with open(args.output, "w") as csv:
        csv.write("frame_id,timestamp_ms,x_cam,y_cam,z_cam,"
                  "x_world,y_world,z_world,conf\n")

        frame_id = 0
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            det   = detect_bin(frame, model)

            
            if det is not None:
                x1, y1, x2, y2, conf = det
                print(f"[frame {frame_id:04d}] DETECTED bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={conf:.2f}")

                # draw bounding box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # draw confidence text
                cv2.putText(
                    frame,
                    f"bin {conf:.2f}",
                    (int(x1), max(30, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )



                """
                xyz_cam   = estimate_3d((x1, y1, x2, y2), K, D)
                xyz_world = cam_to_world(xyz_cam, R, t)

                if kf is not None:
                    xyz_world = kf.update(xyz_world)

                last_known = xyz_world
                last_age   = 0

                xw, yw, zw = xyz_world
                dt_ms = int((time.perf_counter() - t0) * 1000)
                print(f"[frame {frame_id:04d}] bin @ world "
                      f"({xw:.2f}, {yw:.2f}, {zw:.2f}) m  "
                      f"conf={conf:.2f}  dt={dt_ms}ms")

                csv.write(f"{frame_id},{ts_ms},"
                          f"{xyz_cam[0]:.4f},{xyz_cam[1]:.4f},{xyz_cam[2]:.4f},"
                          f"{xw:.4f},{yw:.4f},{zw:.4f},{conf:.3f}\n")
                trajectory.append((xw, yw))
            """
            else:
                print(f"[frame {frame_id:04d}] NO DETECTION")
            """
                last_age += 1
                predicted = kf.predict() if kf is not None else last_known
                lk = (f"({predicted[0]:.2f}, {predicted[1]:.2f}, {predicted[2]:.2f})"
                      if predicted is not None else "unknown")
                print(f"[frame {frame_id:04d}] OCCLUDED — "
                      f"last known {lk} m  age={last_age}fr")
            """
            # show live video
            cv2.imshow("Bin Detection", frame)

            # allow window to refresh + exit on ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                 break
            frame_id += 1

    
    cap.release()
    cv2.destroyAllWindows()
    """
    _save_trajectory_plot(trajectory)
    print(f"\nDone. Results saved to {args.output} and trajectory.png")
    """


def _save_trajectory_plot(trajectory: list):
    """
    TODO: complete this function.
    Plot the bin trajectory in the world XY plane (top-down).
    Mark the 3 known ground-truth stop positions.
    Save to trajectory.png.
    """
    import matplotlib.pyplot as plt
    if not trajectory:
        return
    xs, ys = zip(*trajectory)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, linewidth=1.2, color="steelblue", label="trajectory")
    ax.scatter(xs[0],  ys[0],  s=80, color="green", zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], s=80, color="red",   zorder=5, label="end")

    # TODO: add ground-truth stop markers
    # gt_stops = [(x1,y1), (x2,y2), (x3,y3)]
    # ax.scatter(*zip(*gt_stops), s=140, marker='*', color='orange',
    #            zorder=6, label='GT stops')

    ax.set_xlabel("X world [m]")
    ax.set_ylabel("Y world [m]")
    ax.set_title("Garbage bin trajectory — world XY plane (top-down view)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.savefig("trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
