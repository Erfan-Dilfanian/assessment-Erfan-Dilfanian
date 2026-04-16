#!/usr/bin/env bash
# =============================================================================
# Skyscouter – CV Engineer Assessment
# Entry point. Run as: bash run.sh --video input.mp4 --calib calib.json
# Add --gpu flag to enable GPU inference (document GPU specs in README).
# =============================================================================
set -e

# --- parse arguments ---------------------------------------------------------
VIDEO=""
CALIB=""
GPU_FLAG=""
KALMAN_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --video)   VIDEO="$2";       shift 2 ;;
    --calib)   CALIB="$2";       shift 2 ;;
    --gpu)     GPU_FLAG="--gpu"; shift   ;;
    --kalman)  KALMAN_FLAG="--kalman"; shift ;;
    *)         echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" || -z "$CALIB" ]]; then
  echo "Usage: bash run.sh --video <path> --calib <path> [--gpu] [--kalman]"
  exit 1
fi

# --- environment setup -------------------------------------------------------

echo "[run.sh] Installing system dependencies..."
#sudo apt update
#sudo apt install -y python3.10 python3-venv
#echo "[INFO] Installing system dependencies..."


#sudo apt install -y \
#    libxcb-xinerama0 \
#    libxkbcommon-x11-0 \
#    libxcb-cursor0 \
#    libgl1-mesa-glx

#if [[ -d ".venv" ]]; then
#  echo "[run.sh] Removing existing .venv ..."
#  rm -rf .venv
#fi

if [[ ! -d ".venv" ]]; then
  echo "[run.sh] Creating virtual environment..."
  python3.10 -m venv .venv
else
  echo "[run.sh] Reusing existing .venv ..."
fi

source .venv/bin/activate

# --- OpenCV Qt GUI (WSL / Linux) ---
export QT_QPA_PLATFORM_PLUGIN_PATH="$PWD/.venv/lib/python3.10/site-packages/cv2/qt/plugins"
export QT_QPA_PLATFORM=xcb

echo "[run.sh] Using $(python --version)"

echo "[run.sh] Installing Python dependencies..."
pip install --upgrade pip
pip install -v -r requirements.txt

# --- run pipeline ------------------------------------------------------------
echo "[run.sh] Starting tracker..."
echo "[run.sh] Video : $VIDEO"
echo "[run.sh] Calib : $CALIB"
[[ -n "$GPU_FLAG"    ]] && echo "[run.sh] Mode  : GPU"    || echo "[run.sh] Mode  : CPU"
[[ -n "$KALMAN_FLAG" ]] && echo "[run.sh] Kalman: enabled" || echo "[run.sh] Kalman: disabled"
echo ""

python track_bin.py \
  --video  "$VIDEO"  \
  --calib  "$CALIB"  \
  $GPU_FLAG          \
  $KALMAN_FLAG
