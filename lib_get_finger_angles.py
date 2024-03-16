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
setStaticAngles = False

errorAngleThreshold = 5
CurrentAngles = [0.0, 0.0, 0.0, 0.0, 0.0]
current_ema_angles = [0, 0, 0, 0, 0]  # 初始EMA值，可以根据需要调整
alpha = 0.2  # EMA平滑系数，可以根据实际需要调整
def calculate_ema(current_value, previous_ema, alpha=0.3):
    """
    计算单个值的EMA。
    :param current_value: 当前观测值
    :param previous_ema: 上一个EMA值
    :param alpha: 平滑系数,介于0和1之间
    :return: 新的EMA值
    """
    return alpha * current_value + (1 - alpha) * previous_ema

angleMove = 6 #判断其他手指是否移动的阈值角度

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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
    global CurrentAngles, staticAngles, setStaticAngles, current_ema_angles
    raw_angles = [0,0,0,0,0]
    try:
        while True:
            if(setStaticAngles == False):
                if((datetime.now() - now).total_seconds() < 3):#时间小于三秒
                    print("please place your hand in static pose")
                elif((datetime.now() - now).total_seconds() > 3):#时间大于三秒，取现在的角度为静止角度
                    setStaticAngles = True
                    for i in range(1,6):
                        staticAngles[i] = CurrentAngles[i - 1]
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
            results = hands.process(color_image)

            # 绘制手部关键点并打印3D世界坐标
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    # 绘制图像坐标系中的关键点
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # 234 567 9 10 11
                    point2 = point3 = point1 = None
                    point5 = point6 = point0 = None
                    point9 = point10 = point11 = None
                    point13 = point14 = point15 = None#4
                    point17 = point18 = point19 = None#5
                    for id, lm in enumerate(hand_world_landmarks.landmark):
                        # 打印世界坐标系中的关键点
                        # if id == 8:
                        #     distance = math.sqrt(lm.x* lm.x + lm.y * lm.y + lm.z * lm.z)
                        #     print(f'distance is :{distance} World Landmark #{id}: x: {lm.x}m, y: {lm.y}m, z: {lm.z}m')
                        if id == 2:
                            point2 = [lm.x, lm.y, lm.z]
                            #print(f"point2 is : {point2}")
                        if id == 3:
                            point3 = [lm.x, lm.y, lm.z]
                            #print(f"point3 is : {point3}")
                        if id == 1:
                            point1 = [lm.x, lm.y, lm.z]
                            #print(f"point4 is : {point4}")
                            
                        if id == 5:
                            point5 = [lm.x, lm.y, lm.z]
                            #print(f"point5 is : {point5}")
                        if id == 6:
                            point6 = [lm.x, lm.y, lm.z]
                            #print(f"point6 is : {point6}")
                        if id == 0:
                            point0 = [lm.x, lm.y, lm.z]
                            #print(f"point7 is : {point7}")
                            
                        if id == 9:
                            point9 = [lm.x, lm.y, lm.z]
                            #print(f"point9 is : {point9}")
                        if id == 10:
                            point10 = [lm.x, lm.y, lm.z]
                            #print(f"point10 is : {point10}")
                        if id == 11:
                            point11 = [lm.x, lm.y, lm.z]
                            #print(f"point11 is : {point11}")
                            
                        if id == 13:
                            point13 = [lm.x, lm.y, lm.z]
                            #print(f"point9 is : {point9}")
                        if id == 14:
                            point14 = [lm.x, lm.y, lm.z]
                            #print(f"point10 is : {point10}")
                        if id == 15:
                            point15 = [lm.x, lm.y, lm.z]
                            #print(f"point11 is : {point11}")
                            
                        if id == 17:
                            point17 = [lm.x, lm.y, lm.z]
                            #print(f"point9 is : {point9}")
                        if id == 18:
                            point18 = [lm.x, lm.y, lm.z]
                            #print(f"point10 is : {point10}")
                        if id == 19:
                            point19 = [lm.x, lm.y, lm.z]
                            #print(f"point11 is : {point11}")

                    if all(point is not None for point in [point2, point3, point1, point5, point6, point0, point9, point10, point11,point13,point14,point15,point17,point18,point19]):
                            
                            # CurrentAngles[0] = miniLib.calculate_angle(point0,point5,point6)#食指
                            # CurrentAngles[1] = miniLib.calculate_angle(point0,point9,point10)#middle中指
                            # CurrentAngles[2] = miniLib.calculate_angle(point1,point2,point3)#thumb拇指
                            # CurrentAngles[3] = miniLib.calculate_angle(point0,point13,point14)#无名指
                            # CurrentAngles[4] = miniLib.calculate_angle(point0,point17,point18)#小拇指
                            # #print(f"index finger(食指) angle is: {CurrentAngles[0]}; middle finger(中指) angle is: {CurrentAngles[1]}; thumb finger(大拇指) angle is: {CurrentAngles[2]}; ring finger(无名指) angle is: {CurrentAngles[3]}; pinky finger(小指) angle is: {CurrentAngles[4]}")
                            # cv2.putText(color_image, f"Index: {CurrentAngles[0]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Middle: {CurrentAngles[1]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Thumb: {CurrentAngles[2]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Ring: {CurrentAngles[3]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # cv2.putText(color_image, f"Pinky: {CurrentAngles[4]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            # #time.sleep(0.1)
                            raw_angles[0] = calculate_angle(point1,point2,point3)#thumb拇指
                            raw_angles[1] = calculate_angle(point0,point5,point6)#食指
                            raw_angles[2] = calculate_angle(point0,point9,point10)#middle中指
                            raw_angles[3] = calculate_angle(point0,point13,point14)#无名指
                            raw_angles[4] = calculate_angle(point0,point17,point18)#小拇指
                            
                            for i in range(0,5):#计算ema数值
                                current_ema_angles[i] = calculate_ema(raw_angles[i], current_ema_angles[i], alpha)
                            
                            CurrentAngles = current_ema_angles.copy()
                            cv2.putText(color_image, f"Thumb: {CurrentAngles[0]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Index: {CurrentAngles[1]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Middle: {CurrentAngles[2]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Ring: {CurrentAngles[3]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(color_image, f"Pinky: {CurrentAngles[4]:.2f}", (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * color_image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * color_image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #显示图像
            cv2.imshow('MediaPipe Hands', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == 27:
               break
            
    finally:
        pipeline.stop()
    return

#启动后台一直更新每个角度的angle
thread = threading.Thread(target=update_CurrentAngles)
thread.start()

def get_finger_angle(finger_num):
    """得到指定手指的角度

    Args:
        finger_num (double): 1:大拇指 2:食指。。。。。

    Returns:
        double: 其余手指的最大角度
    """
    # Implement interface to get normalized finger angle
    angle = CurrentAngles[finger_num - 1] - staticAngles[finger_num]
    return angle

def get_other_finger_angle(finger_num):
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

def get_state(finger_num):
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
    print("Wrong situation happened in getState function")
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

