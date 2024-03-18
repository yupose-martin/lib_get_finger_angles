import lib_get_finger_angles

def main():
    while(True):
        for i in range(1,6):
            print(f"{i}'s finger's angle is {lib_get_finger_angles.get_finger_angle(finger_num=i):.1f}")

if __name__ == "__main__":
    main()
