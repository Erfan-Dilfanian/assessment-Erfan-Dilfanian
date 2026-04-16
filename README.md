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

my own added dataset:

https://app.roboflow.com/registration-mfqu6/bin-735gq/1

data augmentation. flip. resize

mAP curve