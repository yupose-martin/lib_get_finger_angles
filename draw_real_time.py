"""實時更新畫圖"""
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import count
import threading

# 使用itertools.count生成一个无限的迭代器，用于模拟时间或数据点的索引

# 初始化角度数据列表和时间（或索引）列表
list_mA = []

list_thumb_angle = []
list_index_angle = []
list_middle_angle = []
list_ring_angle = []
list_pinky_angle = []

list_angles = {1: list_thumb_angle,
               2: list_index_angle,
               3: list_middle_angle,
               4: list_ring_angle,
               5: list_pinky_angle}

def plot_angles(list_mA:"list",list_angles:"list[list]"):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)  # 设置初始x轴范围
    ax.set_ylim(0, 180)  # 假设角度在0到180度之间

    while True:
        # 这里可以添加一个条件来优雅地停止循环
        plt.pause(1)  # 稍微暂停以便更新图表
        if len(list_mA) > 30:  # 如果超过10个数据点，则动态调整x轴的范围
            ax.set_xlim(list_mA[-30], list_mA[-1] + 1)
            pass
        if list_angles:
            ax.plot(list_mA, list_angles[1], 'r-')  # 'r-'表示红色实线
            ax.plot(list_mA, list_angles[2], 'g-')
            ax.plot(list_mA, list_angles[3], 'b-')
            ax.plot(list_mA, list_angles[4], 'b-')
            ax.plot(list_mA, list_angles[5], 'b-')
            
            fig.canvas.draw()
            fig.canvas.flush_events()

# 创建并启动绘图线程
plot_thread = threading.Thread(target=plot_angles,args=(list_mA,list_angles))
plot_thread.start()

# 更新角度数据
def main():
    global list_mA,list_angles
    for i in range(1,100):
        time.sleep(2)
        list_mA.append(i)
        for j in range(1,6):
            list_angles[j].append(np.random.randint(100))
            print(f"list_ma:{list_mA}")
            print(f"list_angles:{list_angles}")
    return

if __name__ == "__main__":
    main()
