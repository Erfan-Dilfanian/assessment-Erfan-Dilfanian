command: bash run.sh --video "Input sample.mp4" --calib calib.json --kalman

COCO does not have trash can / garbage bin

training.
occluded

augmentation

help from external roboflow dataset


resize for edge employment

Tuning of Q and R:
I tuned `process_var`(Q) and `meas_var`(R) in Klaman filter. With the prior knowledge that the motion itself would be smooth, I tend to decrease process_var as lower Q means less process noise and the motion is smooth. As localization's source is from detection, the measuremtns are noisy. Hence, I tend to increase meas_var as higher R means measuremetns are noisy


ground truth comparison
# Skyscouter Assessment

The impleemntation is tested on WSL 

# dependencies
the python dependencies and packages are installed via run.sh file.
run.sh should install these:
```bash
sudo apt update
sudo apt install -y python3.10 python3-venv \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libgl1-mesa-glx
```

I used some publlic datasets from roboflow, and added osme black trash bin images and labeled them myself too.
some public datasets I used:
https://universe.roboflow.com/test-fg7sa/bin-detection-test2



data augmentation. flip. resize

mAP curve
# Usage

```bash

bash run.sh --video "Input sample.mp4" --calib calib.json --gpu
```

GPU spec:
```bash
Thu Apr 16 12:31:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.51                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2070        On  |   00000000:01:00.0 Off |                  N/A |
| N/A   60C    P8              6W /  115W |      14MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A        32      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+
```
---
---

# 1. Detection

## pipeline
I trained a YOLOv8 model for detection. 

I used some of the public dataset, such as the following:

https://universe.roboflow.com/test-fg7sa/bin-detection-test2


I found th emdoel trained solely on this public dataset struggles for black garbage bin, especially in grey background. So I gathered some images fomr the internet and also created some of my own dataset and added to that public dataset. My own dataset can be seen here:

https://app.roboflow.com/registration-mfqu6/bin-735gq/1

## occlusion handling

To maintain detection continuity during temporary occlusions (e.g., a person walking in front of the bin), the system combines motion-based bounding box prediction with Kalman-filter-based state estimation.


A frame is considered valid measurement only if:

* A detection exists, and
* Confidence ≥ `MIN_CONF_FOR_MEAS = 0.20`

Otherwise, the frame is treated as occluded.


An Image-space motion model is develoepd. When detections are available, we estimate the image-plane velocity of the bin:

v_x = c_x(t) - c_x(t-1)

v_y = c_y(t) - c_y(t-1)

where ((c_x, c_y)) is the bounding box center.

During occlusion, we propagate the last known bounding box:

c_x_pred = c_x + v_x

c_y_pred = c_y + v_y

The width and height are kept constant, producing a predicted bounding box.

This allows:

* Continuous visualization
* Continued 3D estimation even without detections




There is a Temporal constraint for the prediction. Prediction is only trusted for a limited duration:

age <= MAX_OCCLUSION_FRAMES (= 12)

After that, the target is considered lost.


We also have 3D prediction using Kalman filter if enabled. A constant-velocity Kalman filter is used in world coordinates:

**State vector:**
[
x = [x, y, z, v_x, v_y, v_z]^T
]

**Prediction step:**
[
x_{t+1} = F x_t
]

During occlusion:

* No measurement update is applied
* Only the prediction step is used

This provides:

* Smooth trajectory continuation
* Robust handling of missing detections





If Kalman is disabled:

* The system uses the last known world position
* Still outputs continuous coordinates (no frame drops)





## justification of model selection
YOLOv8 is efficient for real-time detection due to its short inference time. At my best epoch, epoch 41, I have these evaluation criteria:
mAP50-95: 0.9399
Precision: 0.9762
Recall: 0.999

# 2. 3D localization




This section describes how the 3D position of the garbage bin is estimated from monocular images using the pinhole camera model and known object dimensions, and how it is transformed into a world coordinate frame.



## **2a. Distance estimation from bounding box**

We estimate the distance from the camera origin to the bin centroid using a pinhole camera model and the known physical height of the bin.

Known quantities

* Bin height:
  
  H = 0.65 m
  
* Camera intrinsics:

K = [[fx, 0, cx],
     [0, fy, cy],
     [0,  0,  1]]
* Bounding box pixel height:
  
  h_px = y_2 - y_1
  



for Depth derivation (pinhole model)

Using similar triangles:


h_px/f_y = H/Z


Rearranging:


Z = (f_y*H)/h_px


This gives the depth of the bin centroid along the optical axis (zcam).

This method assumes Bin is upright

Once the 3D coordinates are computed (see 2b), the true Euclidean distance (the final distance to camera origin) is:


d = \sqrt{X^2 + Y^2 + Z^2}





## **2b. 3D position in camera frame**

After estimating depth (Z), we recover full 3D coordinates ((X, Y, Z)) in the camera coordinate frame.





First, bounding box center is obtained. The image projection of the bin centroid is approximated by:


u = (x_1 + x_2)/2
v = (y_1 + y_2)/2


The image is affected by lens distortion, so the pixel coordinates are first undistorted using this command:

```python
cv2.undistortPoints(...)
```

This yields normalized image coordinates: x_n, y_n


These satisfy:


x_n = X/Z

y_n = Y/Z



Hence, to recover X and Y, we have:


X = x_n * Z

Y = y_n * Z



And final camera-frame coordinates will be:


P_cam = [X, Y, Z]^T = [x_n * Z, y_n * Z, Z]^T



Note that camera frame coordinate convention (OpenCV) are:

* (X): right
* (Y): down
* (Z): forward (optical axis)

Each frame outputs:

```
frame_id, timestamp_ms, x_cam, y_cam, z_cam, confidence
```


---

## **2c. Transform to world frame**

We convert camera-frame coordinates into a world frame fixed at the base of the pole.

World frame definition is:

* Origin: base of camera pole
* Axes:

  * (+X): forward along ground
  * (+Y): left
  * (+Z): upward


We first align camera axes with world axes (first R matrix):

X_world =  Z_cam
Y_world = -X_cam
Z_world = -Y_cam

This is implemented using the matrix:

R_axis =

[  0   0   1 ]
[ -1   0   0 ]
[  0  -1   0 ]

---

Then we drive R matrix for Camera tilt compensation:

The camera is tilted downward by angle (\theta).
We apply a rotation around the Y-axis:

R_y (rotation around Y-axis):

[  cos(θ)   0   sin(θ) ]
[    0      1     0    ]
[ -sin(θ)   0   cos(θ) ]





Final combined rotation:

R = R_y * R_axis





The camera is mounted at height (h), so the translation vector is:

t = [ 0, 0, h ]^T





final Camera-to-world transformation:

P_world = R * P_cam + t



### **Interpretation**

* Rotation aligns coordinate systems and accounts for tilt
* Translation places the camera above the ground
* Result:

  * (Z_{world}) represents **height above ground**
  * (X_{world}, Y_{world}) define ground-plane position

---

### **Output format**

```
frame_id, t_ms, x_cam, y_cam, z_cam,
x_world, y_world, z_world, conf
```

This matches Task 2c requirements .

---

## **Summary of pipeline**

For each frame:

1. Detect bounding box
2. Compute:

   * Pixel height → depth (Z)
   * Undistorted centroid → (x_n, y_n)
   * (X, Y) from normalized coordinates
3. Form:
   [
   P_{cam} = (X, Y, Z)
   ]
4. Transform:
   [
   P_{world} = R P_{cam} + t
   ]
5. Output results in CSV + real-time logs

---

## **Why this works well**

* Uses **physical object size → metric scale**
* Uses **camera calibration → accurate projection**
* Uses **full geometry → no black-box solvePnP**
* Meets requirement:

  * explicit derivation
  * interpretable pipeline
  * no disallowed shortcuts

---


