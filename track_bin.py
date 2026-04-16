"""
Skyscouter – Computer Vision Engineer Technical Assessment
Skeleton file: track_bin.py  (v2 – updated with geometry guidance)
Restructure freely. This is a starting point only.
"""

import cv2
import json
import argparse
import time
import numpy as np


from detector import load_detector, detect_bin
from localizer import (

    estimate_3d,
    camera_distance_from_xyz,   # only if you want to print camera distance too
)

# ── Known target dimensions ──────────────────────────────────────────────────
BIN_DIAMETER_M = 0.40   # standard outdoor garbage bin
BIN_HEIGHT_M   = 0.65

# ── Calibration loader ────────────────────────────────────────────────────────

def load_calib(path: str):
    """
    Load camera intrinsics and mount geometry from calib.json.
    Returns:
        K         (3×3 ndarray)  intrinsic matrix
        D         (5,  ndarray)  distortion coefficients [k1,k2,p1,p2,k3]
        cam_h     (float)        camera height above ground in metres
        tilt_rad  (float)        downward tilt in radians (negative = down)
    """
    with open(path) as f:
        c = json.load(f)
    K        = np.array(c["K"],           dtype=np.float64)
    D        = np.array(c["dist_coeffs"], dtype=np.float64)
    cam_h    = float(c["camera_height_m"])
    tilt_rad = float(np.deg2rad(c["camera_tilt_deg"]))
    return K, D, cam_h, tilt_rad


# ── Coordinate transforms ─────────────────────────────────────────────────────

def build_extrinsic(cam_h: float, tilt_rad: float):
    """
    Build rotation matrix R and translation vector t so that:
        P_world = R @ P_cam + t

    World frame
    -----------
    Origin : base of the camera pole on the ground
    +X     : forward (away from pole, along optical axis projected to ground)
    +Y     : left
    +Z     : up

    Camera frame (standard OpenCV convention)
    ------------------------------------------
    +Z : forward (optical axis)
    +X : right
    +Y : down

    Derivation
    ----------
    Step 1 — Rotation only (no translation yet)
    The camera is pitched downward by |tilt_rad| around the camera X axis.
    A pitch-down rotation maps camera Z (forward) toward world -Z (down).

    Rotation matrix for pitch angle θ around X axis:
        Rx(θ) = [[1,      0,       0    ],
                 [0,  cos(θ), -sin(θ)  ],
                 [0,  sin(θ),  cos(θ)  ]]

    Here θ = tilt_rad (negative value means pitched down).

    After applying Rx the axes are:
        cam_X  -> world  X (right stays right... wait, we also need axis swap)

    Full axis mapping from camera to world:
        world_X =  cam_Z  (forward)
        world_Y = -cam_X  (left = negative camera-right)
        world_Z = -cam_Y  (up = negative camera-down)

    Combined: first swap axes, then apply pitch rotation.
    The resulting 3×3 rotation matrix R satisfies:
        P_world_rotated = R @ P_cam

    Step 2 — Translation
    The camera optical centre sits at height cam_h above the world origin.
    In world coordinates the camera is at (0, 0, cam_h), so:
        t = np.array([0.0, 0.0, cam_h])

    Full transform:
        P_world = R @ P_cam + t

    TODO: implement the matrix below using the derivation above.
    Hint: np.array([[...],[...],[...]]) for R, then set t.
    """
    # Axis-swap matrix: maps (cam_X, cam_Y, cam_Z) -> (world_X, world_Y, world_Z)
    # world_X =  cam_Z  -> row 0 = [0,  0,  1]
    # world_Y = -cam_X  -> row 1 = [-1, 0,  0]
    # world_Z = -cam_Y  -> row 2 = [0, -1,  0]
    axis_swap = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0],
    ], dtype=np.float64)

    # Pitch rotation around world Y axis by tilt_rad
    # (camera tilts down, so scene points in front appear below optical centre)
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)

    # TODO: combine Ry and axis_swap into a single R, then set t
    # R = ...
    # t = ...
    R = Ry @ axis_swap
    t = np.array([0.0, 0.0, cam_h], dtype=np.float64)
    return R, t
    # raise NotImplementedError(
    #     "TODO: combine Ry @ axis_swap into R, set t = [0, 0, cam_h]"
    # )


def cam_to_world(xyz_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R @ xyz_cam + t


# ── Detection ────────────────────────────────────────────────────────────────

def load_detector_(use_gpu: bool = False):
    """
    Load and return your detector.
    If use_gpu=True, configure the model to use GPU.
    Document GPU model, VRAM, and CUDA version in README if use_gpu=True.
    """
    raise NotImplementedError("TODO: load detector")


def detect_bin_(frame: np.ndarray, model) -> tuple | None:
    """
    Detect the garbage bin in a single BGR frame.
    Returns (x1, y1, x2, y2, confidence) or None if not detected.

    Note: standard COCO-pretrained YOLOv8 includes a 'trash can' class.
    Document in README whether you use it directly or fine-tune, and why.
    Fine-tuning on frames from input.mp4 is NOT permitted (test-set leakage).
    """
    raise NotImplementedError("TODO: implement detect_bin")


# ── 3D localisation ───────────────────────────────────────────────────────────

def estimate_3d_(bbox: tuple, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Estimate the bin centroid position in the CAMERA frame.

    Pinhole depth estimate (show this derivation in README):
        Z = f_y * BIN_HEIGHT_M / bbox_pixel_height
        X = (u_centre - cx) * Z / fx
        Y = (v_centre - cy) * Z / fy

    where (u_centre, v_centre) is the pixel centroid of the bounding box,
    and (cx, cy), fx, fy come from K.

    Args:
        bbox  (x1, y1, x2, y2) in pixels
        K     intrinsic matrix (3×3)
        D     distortion coefficients (undistort centroid before projecting)

    Returns:
        xyz_cam  (3,) ndarray [x_cam, y_cam, z_cam] in metres
    """
    raise NotImplementedError("TODO: implement estimate_3d")


# ── Optional: Kalman filter ───────────────────────────────────────────────────

class PositionKalman:
    """
    Constant-velocity Kalman filter over 3D world position.
    State vector: [x, y, z, vx, vy, vz]

    Implement this for the bonus task (3c).
    Explain Q, R, and state vector choices in README.
    """
    def __init__(self):
        raise NotImplementedError("TODO: implement PositionKalman")

    def update(self, xyz_meas: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self) -> np.ndarray:
        raise NotImplementedError


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skyscouter bin tracker")
    parser.add_argument("--video",  required=True, help="Path to input.mp4")
    parser.add_argument("--calib",  required=True, help="Path to calib.json")
    parser.add_argument("--output", default="results/output.csv")
    parser.add_argument("--gpu",    action="store_true",
                        help="Run detector on GPU (document specs in README)")
    parser.add_argument("--kalman", action="store_true",
                        help="Enable Kalman filter smoothing (bonus 3c)")
    args = parser.parse_args()

    K, D, cam_h, tilt_rad = load_calib(args.calib)
    R, t   = build_extrinsic(cam_h, tilt_rad)
    model  = load_detector(use_gpu=args.gpu)
    kf     = PositionKalman() if args.kalman else None

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    trajectory = []
    last_known = None
    last_age   = 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    with open(args.output, "w") as csv:
        csv.write("frame_id,timestamp_ms,x1,y1,x2,y2,"
          "x_cam,y_cam,z_cam,x_world,y_world,z_world,conf\n")

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

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"bin {conf:.2f}",
                    (int(x1), max(30, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                xyz_cam   = estimate_3d((x1, y1, x2, y2), K, D)

                dist_cam = camera_distance_from_xyz(xyz_cam)

                xyz_world = cam_to_world(xyz_cam, R, t)

                if kf is not None:
                    xyz_world = kf.update(xyz_world)

                last_known = xyz_world
                last_age   = 0
                xw, yw, zw = xyz_world
                dt_ms = int((time.perf_counter() - t0) * 1000)
                print(f"[frame {frame_id:04d}] "
                    f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}) "
                    f"bin @ world ({xw:.2f}, {yw:.2f}, {zw:.2f}) m  "
                    f"conf={conf:.2f}  dt={dt_ms}ms")
                csv.write(f"{frame_id},{ts_ms},"
                        f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},"
                        f"{xyz_cam[0]:.4f},{xyz_cam[1]:.4f},{xyz_cam[2]:.4f},"
                        f"{xw:.4f},{yw:.4f},{zw:.4f},{conf:.3f}\n")
                trajectory.append((xw, yw))
            else:
                last_age += 1
                predicted = kf.predict() if kf is not None else last_known
                lk = (f"({predicted[0]:.2f}, {predicted[1]:.2f}, {predicted[2]:.2f})"
                      if predicted is not None else "unknown")
                print(f"[frame {frame_id:04d}] OCCLUDED — "
                      f"last known {lk} m  age={last_age}fr")
                
            
            cv2.imshow("Bin Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
                
            frame_id += 1





    cap.release()

    cv2.destroyAllWindows()

    _save_trajectory_plot(trajectory)
    print(f"\nDone. Results: {args.output}  |  trajectory.png")


def _save_trajectory_plot(trajectory: list):
    """
    Generate top-down 2D plot of bin trajectory in world XY plane.
    Mark the 3 stop positions. Save as trajectory.png.
    Load waypoints.json if available to overlay tape marker positions.
    """
    import matplotlib.pyplot as plt, os, json

    if not trajectory:
        print("No trajectory data — skipping plot.")
        return

    xs, ys = zip(*trajectory)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, linewidth=1.2, color="steelblue", label="trajectory")
    ax.scatter(xs[0],  ys[0],  s=80, color="green", zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], s=80, color="red",   zorder=5, label="end")

    # If you have estimated waypoint world coordinates, plot them here:
    # estimated_stops = [(x1,y1), (x2,y2), (x3,y3)]
    # ax.scatter(*zip(*estimated_stops), s=140, marker='*',
    #            color='orange', zorder=6, label='estimated GT stops')

    ax.set_xlabel("X world [m]")
    ax.set_ylabel("Y world [m]")
    ax.set_title("Garbage bin trajectory — world XY plane (top-down)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.savefig("trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("trajectory.png saved.")


if __name__ == "__main__":
    main()
