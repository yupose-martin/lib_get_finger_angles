# lib_get_finger_angles
## NOTICE!!!
* 由於mediapipe handlandmarks不能夠檢測左手右手，所以在使用left_pitch，right_pitch,left_yaw,right_yaw的時候，只能在屏幕中出現該側手掌

## requirement:
1. camera: intel realsense
2. lib needed: mediapipe, pyrealsense2

## functions:
### pitch yaw
* get_abs_left_hand_pitch
* get_abs_left_hand_pitch
* get_abs_left_hand_pitch
* get_abs_left_hand_pitch

## finger angles:
* get_abs_finger_angle: absolute angle
* get_finger_angle: angle relevent to the static angle

## finger curves:
curve = (angle1 + angle2 + angle3) / 3
* get_abs_finger_curve: relevent to the static point
* get_abs_finger_curve: absolute

## mainfile:
### lib_get_finger_angles.py
* the core lib
### demo.py
* shows you how to use it

