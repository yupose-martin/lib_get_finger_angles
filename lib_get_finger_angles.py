import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math, serial, threading, time, cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
now = datetime.now()
stateA = -1 #无移动
stateB = 0  #只有该手指移动
stateC = 1  #其他手指移动

# 手指编号到静态角度的映射
staticAngles = {
    1: 175,  # 大拇指
    2: 155,  # 食指
    3: 155,  # 中指
    4: 155,  # 无名指
    5: 175   # 小拇指
}
setStatic_Angles_Curve = False

staticCurve = {
    1: 175,  # 大拇指
    2: 155,  # 食指
    3: 155,  # 中指
    4: 155,  # 无名指
    5: 175   # 小拇指
}

left_pitch = 0 #手掌的pitch
left_yaw = 0 #手掌的yaw
right_pitch = 0 #手掌的pitch
right_yaw = 0 #手掌的yaw

errorAngleThreshold = 5
angleMove = 6 #判断其他手指是否移动的阈值角度
CurrentAngles = [0.0, 0.0, 0.0, 0.0, 0.0]
current_ema_angles = [0, 0, 0, 0, 0]  # 初始EMA值，可以根据需要调整
CurrentCurveRate = [0,0,0,0,0]# (angle1 + angle2 + angle3) / (3 * 180)
current_ema_curve_rate = [0, 0, 0, 0, 0] #

alpha = 0.2  # EMA平滑系数，可以根据实际需要调整

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=2,
                                smooth_landmarks=True,
                                enable_segmentation=False,
                                smooth_segmentation=True,
                                refine_face_landmarks=False,
                                min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)

def calculate_ema(current_value, previous_ema, alpha=0.3):
    """
    计算单个值的EMA。
    :param current_value: 当前观测值
    :param previous_ema: 上一个EMA值
    :param alpha: 平滑系数,介于0和1之间
    :return: 新的EMA值
    """
    return alpha * current_value + (1 - alpha) * previous_ema


# 配置和启动RealSense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

def update_CurrentAngles():
    """mediapipe code used to update CurrentAngles in the thread
    """
    #mediapipe code used to update CurrentAngles
    global CurrentAngles, current_ema_angles, staticAngles, setStatic_Angles_Curve, CurrentCurveRate, current_ema_curve_rate, left_yaw, right_yaw, left_pitch, right_pitch
    Angle_represent = [0,0,0,0,0] #當前讀取的角度
    Curve_rate_represent = [0,0,0,0,0] #curve _rate
    left_pitch_represent = 0  # 手掌的pitch
    left_yaw_represent = 0  # 手掌的yaw
    right_pitch_represent = 0  # 手掌的pitch
    right_yaw_represent = 0  # 手掌的yaw

    left_forearm_direction = np.array([0, 0, 0])
    right_forearm_direction = np.array([0, 0, 0])

    index_direction = np.array([0, 0, 0])
    middle_direction = np.array([0, 0, 0])
    pinky_direction = np.array([0, 0, 0])
    try:
        while True:
            if(setStatic_Angles_Curve == False):
                if((datetime.now() - now).total_seconds() < 5):#时间小于三秒
                    print("please place your hand in static pose")
                elif((datetime.now() - now).total_seconds() > 5):#时间大于三秒，取现在的角度为静止角度
                    setStatic_Angles_Curve = True
                    for i in range(1,6):
                        staticAngles[i] = CurrentAngles[i - 1]
                        staticCurve[i] = CurrentCurveRate[i - 1]
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!static pose detection finished")
            #print(f"static angles are: {staticAngles[1]},{staticAngles[2]}")
            # 获取帧集
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # 转换图像到NumPy数组
            color_image = np.asanyarray(color_frame.get_data())
            # MediaPipe处理
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            holistic_results = holistic.process(color_image)
            hand_results = hands.process(color_image)

            # 绘制手部关键点并打印3D世界坐标
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, hand_world_landmarks in zip(hand_results.multi_hand_landmarks, hand_results.multi_hand_world_landmarks):
                    # 绘制图像坐标系中的关键点
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # 234 567 9 10 11
                    # process index middle pinky direction for usage in calculating pitch yaw
                    wrist_world_landmark = hand_world_landmarks.landmark[0]
                    index_mcp_world_landmark = hand_world_landmarks.landmark[5]
                    middle_mcp_world_landmark = hand_world_landmarks.landmark[9]
                    pinky_mcp_world_landmark = hand_world_landmarks.landmark[13]

                    index_direction = np.array([index_mcp_world_landmark.x - wrist_world_landmark.x,
                                                index_mcp_world_landmark.y - wrist_world_landmark.y,
                                                index_mcp_world_landmark.z - wrist_world_landmark.z])

                    middle_direction = np.array([middle_mcp_world_landmark.x - wrist_world_landmark.x,
                                                 middle_mcp_world_landmark.y - wrist_world_landmark.y,
                                                 middle_mcp_world_landmark.z - wrist_world_landmark.z])

                    pinky_direction = np.array([pinky_mcp_world_landmark.x - wrist_world_landmark.x,
                                                pinky_mcp_world_landmark.y - wrist_world_landmark.y,
                                                pinky_mcp_world_landmark.z - wrist_world_landmark.z])
                    points = [None] * 21
                    for id, lm in enumerate(hand_world_landmarks.landmark):
                        # 打印世界坐标系中的关键点
                        # if id == 8:
                        #     distance = math.sqrt(lm.x* lm.x + lm.y * lm.y + lm.z * lm.z)
                        #     print(f'distance is :{distance} World Landmark #{id}: x: {lm.x}m, y: {lm.y}m, z: {lm.z}m')
                        for i in range(0,21):
                            if id == i:
                                points[i] = [lm.x, lm.y, lm.z]
                                break

                    if all(point is not None for point in points):
                            
                            Angle_represent[0] = calculate_angle(points[1],points[2],points[3])#thumb拇指
                            Angle_represent[1] = calculate_angle(points[0],points[5],points[6])#食指
                            Angle_represent[2] = calculate_angle(points[0],points[9],points[10])#middle中指
                            Angle_represent[3] = calculate_angle(points[0],points[13],points[14])#无名指
                            Angle_represent[4] = calculate_angle(points[0],points[17],points[18])#小拇指
                            
                            for i in range(0,5):#看mediapipe index
                                index1 = 0
                                index2 = int(4 * i + 1)
                                index3 = int(4 * i + 2)
                                index4 = int(4 * i + 3)
                                index5 = int(4 * i + 4)
                                scale_factor = 180 * 3 #因为这里取的是三个角直接相加，scale到 0-1
                                final_scale = 180 #最终还是映射到180吧
                                angle1 = calculate_angle(points[index1],points[index2],points[index3])
                                angle2 = calculate_angle(points[index2],points[index3],points[index4])
                                angle3 = calculate_angle(points[index3],points[index4],points[index5])
                                Curve_rate_represent[i] = ( (angle1 + angle2 + angle3) / scale_factor ) * final_scale

                            for i in range(0,5):#计算ema数值 for representive angles and curve rate
                                current_ema_angles[i] = calculate_ema(Angle_represent[i], current_ema_angles[i], alpha)
                                current_ema_curve_rate[i] = calculate_ema(Curve_rate_represent[i],current_ema_curve_rate[i], alpha)
                            
                            CurrentAngles = current_ema_angles.copy()
                            CurrentCurveRate = current_ema_curve_rate.copy()
                            # #display both angle and curve
                            # cv2.putText(color_image, f"Thumb:{(-CurrentAngles[0] + staticAngles[1]):.1f} curve:{(-CurrentCurveRate[0] + staticCurve[1]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Index:{(-CurrentAngles[1] + staticAngles[2]):.1f} curve:{(-CurrentCurveRate[1] + staticCurve[2]):.1f}" , (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Middle:{(-CurrentAngles[2] + staticAngles[3]):.1f} curve:{(-CurrentCurveRate[2] + staticCurve[3]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Ring:{(-CurrentAngles[3] + staticAngles[4]):.1f} curve:{(-CurrentCurveRate[3] + staticCurve[4]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Pinky:{(-CurrentAngles[4] + staticAngles[5]):.1f} curve:{(-CurrentCurveRate[4] + staticCurve[5]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            #diaplay curve
                            cv2.putText(color_image, f"Thumb:{(-CurrentCurveRate[0] + staticCurve[1]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Index:{(-CurrentCurveRate[1] + staticCurve[2]):.1f}" , (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Middle:{(-CurrentCurveRate[2] + staticCurve[3]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Ring:{(-CurrentCurveRate[3] + staticCurve[4]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Pinky:{(-CurrentCurveRate[4] + staticCurve[5]):.1f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            ## 只显示index finger
                            # cv2.putText(color_image, f"Index: {(CurrentAngles[1]):.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"IndexCurve: {(CurrentCurveRate[1]):.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if holistic_results.pose_world_landmarks:
                # 直接获取关键点列表
                pose_world_landmarks = holistic_results.pose_world_landmarks.landmark
                left_forearm_direction = {0, 0, 0}
                right_forearm_direction = {0, 0, 0}
                # 直接通过索引访问具体关键点
                left_wrist = pose_world_landmarks[15]
                right_wrist = pose_world_landmarks[16]
                left_elbow = pose_world_landmarks[13]
                right_elbow = pose_world_landmarks[14]
                # 计算前臂方向
                left_forearm_direction = np.array([left_wrist.x - left_elbow.x,
                                                   left_wrist.y - left_elbow.y,
                                                   left_wrist.z - left_elbow.z])

                right_forearm_direction = np.array([right_wrist.x - right_elbow.x,
                                                    right_wrist.y - right_elbow.y,
                                                    right_wrist.z - right_elbow.z])
                # print(f"left_forearm_direction = {left_forearm_direction}")
                # print(f"right_forearm_direction = {right_forearm_direction}")
                # left_length = math.pow(left_forearm_direction[0], 2) + math.pow(left_forearm_direction[1],2) + math.pow(left_forearm_direction[2],2)
                # left_length = math.sqrt(left_length)

                normal_vector_of_plane = np.cross(pinky_direction, index_direction)
                # normalization
                normal_vector_of_plane = normal_vector_of_plane / np.linalg.norm(normal_vector_of_plane)
                middle_cross_normal = np.cross(middle_direction, normal_vector_of_plane)

                #calculate left pitch and yaw
                dot_product_plane_left_forearm = np.dot(left_forearm_direction, normal_vector_of_plane)
                left_pitch_represent = 90 - np.degrees(np.arccos(dot_product_plane_left_forearm
                                                       / (np.linalg.norm(normal_vector_of_plane) * np.linalg.norm(
                                                            left_forearm_direction))))
                # print(f"left_pitch = {left_pitch}")

                left_project = left_forearm_direction - (np.linalg.norm(left_forearm_direction) * np.sin(
                    np.deg2rad(left_pitch)) * normal_vector_of_plane)
                dot_product_of_project_middle = np.dot(left_project, middle_cross_normal)
                left_yaw_represent = 90 - np.degrees(np.arccos(dot_product_of_project_middle
                                                     / (np.linalg.norm(left_project) * np.linalg.norm(
                    middle_cross_normal))))
                #print(f"left_yaw = {left_yaw}")


                # calculate right pitch and yaw
                dot_product_plane_right_forearm = np.dot(right_forearm_direction, normal_vector_of_plane)
                right_pitch_represent = 90 - np.degrees(np.arccos(dot_product_plane_right_forearm
                                                       / (np.linalg.norm(normal_vector_of_plane) * np.linalg.norm(
                    right_forearm_direction))))
                # print(f"left_pitch = {right_pitch}")

                right_project = right_forearm_direction - (np.linalg.norm(right_forearm_direction) * np.sin(
                    np.deg2rad(right_pitch)) * normal_vector_of_plane)
                dot_product_of_project_middle = np.dot(right_project, middle_cross_normal)
                right_yaw_represent = 90 - np.degrees(np.arccos(dot_product_of_project_middle
                                                     / (np.linalg.norm(right_project) * np.linalg.norm(
                    middle_cross_normal))))
                # print(f"left_yaw = {right_yaw}")

                #usa ema
                left_pitch = calculate_ema(current_value=left_pitch_represent,previous_ema=left_pitch,alpha=alpha)
                right_pitch = calculate_ema(current_value=right_pitch_represent, previous_ema=right_pitch,alpha=alpha)
                left_yaw = calculate_ema(current_value=left_yaw_represent,previous_ema=left_yaw,alpha=alpha)
                right_yaw = calculate_ema(current_value=right_yaw_represent, previous_ema=right_yaw,alpha=alpha)

                cv2.putText(color_image, f"left_pitch: {left_pitch:1f} left_yaw: {left_yaw}",
                            (int(0.05 * color_image.shape[1]),
                             int(0.05 * color_image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.putText(color_image, f"right_pitch: {right_pitch:1f} right_yaw: {right_yaw}",
                            (int(0.05 * color_image.shape[1]),
                             int(0.1 * color_image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if holistic_results.pose_landmarks:
                    pose_landmarks = holistic_results.pose_landmarks
                    # cv2.putText(color_image, f"l: {left_length:2f} x:{left_forearm_direction[0]:2f} y:{left_forearm_direction[1]:2f} z:{left_forearm_direction[2]:2f}",
                    #             (int(pose_landmarks.landmark[13].x * color_image.shape[1]),
                    #             int(pose_landmarks.landmark[13].y * color_image.shape[0])),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # print(f"results = {results}")
                    # Draw landmark annotation on the image.
                    # color_image.flags.writeable = True
                    # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    # help(mp_drawing.draw_landmarks)
                    # mp_drawing.draw_landmarks(
                    #     color_image,
                    #     results.face_landmarks,
                    #     mp_holistic.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        color_image,
                        holistic_results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Hands', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == 27:
               break
            
    finally:
        pipeline.stop()
    return

#启动后台一直更新每个角度的angle
thread = threading.Thread(target=update_CurrentAngles)
thread.start()

def get_abs_left_hand_pitch()->"float":
    """
    得到左手的pitch
    """
    return left_pitch

def get_abs_left_hand_pitch()->"float":
    """
    得到右手的pitch
    """
    return right_pitch

def get_abs_left_hand_pitch()->"float":
    """
    得到左手的yaw
    """
    return left_yaw

def get_abs_right_hand_yaw()->"float":
    """
    得到右手的yaw
    """
    return right_yaw
def get_abs_finger_angle(finger_num:"int") -> "float":
    """得到指定手指的絕對角度

    Args:
        finger_num (double): 1:大拇指 2:食指。。。。。

    Returns:
        double: 其余手指的最大角度
    """
    return CurrentAngles[finger_num - 1]

def get_finger_angle(finger_num:"int") -> "float":
    """得到指定手指的角度

    Args:
        finger_num (double): 1:大拇指 2:食指。。。。。

    Returns:
        double: 其余手指的最大角度
    """
    angle = CurrentAngles[finger_num - 1] - staticAngles[finger_num]
    return angle

def get_other_finger_angle(finger_num:"int") -> "float":
    """得到其他手指的最大角度

    Args:
        finger_num (int): 给定的手指 1: thumb...

    Returns:
        double: angle
    """
    other_finger_angle = 0.0
    for i in range(1, 6):
        if i != finger_num:
            this_finger_angle = get_finger_angle(i)
            if this_finger_angle > other_finger_angle:
                other_finger_angle = this_finger_angle
    return other_finger_angle

def get_abs_finger_curve(finger_num:"int") -> "float":
    """得到指定手指的絕對curve

    Args:
        finger_num (int): 1:大拇指 2:食指。。。。。

    Returns:
        double: 手指的最大curve
    """
    # Implement interface to get normalized finger angle
    return CurrentCurveRate[finger_num - 1]

def get_abs_finger_curve(finger_num:"int") -> "float":
    """得到指定手指的curve

    Args:
        finger_num (int): 1:大拇指 2:食指。。。。。

    Returns:
        double: 手指的curve
    """
    # Implement interface to get normalized finger angle
    angle = CurrentCurveRate[finger_num - 1] - staticCurve[finger_num]
    return angle

def get_other_finger_curve(finger_num:"int") -> "float":
    """得到其他手指的最大curve

    Args:
        finger_num (int): 给定的手指 1: thumb...

    Returns:
        double: angle
    """
    other_finger_curve = 0.0
    for i in range(1, 6):
        if i != finger_num:
            this_finger_curve = get_finger_angle(i)
            if this_finger_curve > other_finger_curve:
                other_finger_curve = this_finger_curve
    return other_finger_curve

def get_state(finger_num:"int") -> "float":
    other_finger_move = False
    for i in range(1, 6):
        if i != finger_num and get_finger_angle(i) > angleMove:
            other_finger_move = True
    
    finger_angle = get_finger_angle(finger_num)
    if finger_angle > angleMove and other_finger_move:
        return stateC
    if finger_angle < angleMove:
        return stateA
    if finger_angle > angleMove and not other_finger_move:
        return stateB
    print("!!!Wrong situation happened in getState function")
    return None

def calculate_angle(point_a, point_b, point_c):
    """计算由三个点形成的角度,其中point_b是中心点"""
    # 构建向量
    a = [point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2]]
    b = [point_c[0] - point_b[0], point_c[1] - point_b[1], point_c[2] - point_b[2]]

    # 计算点积
    dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    # 计算模
    mag_a = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    # 计算角度
    angle = math.acos(dot_product / (mag_a * mag_b))

    # 将弧度转换为度
    angle_degrees = math.degrees(angle)

    return angle_degrees

