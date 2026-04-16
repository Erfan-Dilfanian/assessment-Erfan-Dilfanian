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