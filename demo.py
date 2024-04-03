import lib_get_finger_angles as gfa

def main():
    while(True):
        for i in range(1,6):
            pass
            #print(f"{i}'s finger's angle is {lib_get_finger_angles.get_finger_angle(finger_num=i):.1f}")
        print(f"left_pitch = {gfa.get_abs_left_hand_pitch()} left_yaw = {gfa.get_abs_left_hand_yaw()}\
              right_pitch = {gfa.get_abs_right_hand_pitch()} right_yaw = {gfa.get_abs_right_hand_yaw()}")
        
if __name__ == "__main__":
    main()
