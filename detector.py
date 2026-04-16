# detector.py
from ultralytics import YOLO
import torch
import numpy as np


def load_detector(use_gpu: bool = False, weights_path: str = None):
    """
    Load YOLO detector from trained weights.
    """

    if weights_path is None:
        weights_path = "./best.pt"

    model = YOLO(weights_path)

    if use_gpu and torch.cuda.is_available():
        model.to("cuda")

        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda

        print("[INFO] YOLO loaded on CUDA")
        print(f"[INFO] GPU: {gpu_name}")
        print(f"[INFO] VRAM: {total_mem:.1f} GB")
        print(f"[INFO] CUDA: {cuda_version}")

    else:
        model.to("cpu")
        print("[INFO] YOLO loaded on CPU")

    return model


def detect_bin(frame: np.ndarray, model, conf_thres: float = 0.25):
    """
    Detect the bin and return the best bbox:
    (x1, y1, x2, y2, confidence)
    """
    results = model.predict(
        source=frame,
        conf=conf_thres,
        device="cuda" if next(model.model.parameters()).is_cuda else "cpu",
        verbose=False
    )

    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes

    best_det = None
    best_conf = -1.0

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2]
        conf = float(box.conf[0].cpu().numpy())

        if conf > best_conf:
            best_conf = conf
            best_det = (*xyxy, conf)

    return best_det