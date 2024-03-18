import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import count
import threading

# 使用itertools.count生成一个无限的迭代器，用于模拟时间或数据点的索引
index = count()

# 初始化角度数据列表和时间（或索引）列表
angles = []
indices = []

def plot_angles():
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)  # 设置初始x轴范围
    ax.set_ylim(0, 180)  # 假设角度在0到180度之间

    while True:
        # 这里可以添加一个条件来优雅地停止循环
        plt.pause(0.1)  # 稍微暂停以便更新图表
        if len(indices) > 10:  # 如果超过10个数据点，则动态调整x轴的范围
            ax.set_xlim(indices[-10], indices[-1] + 1)
        if angles:
            ax.plot(indices, angles, 'r-')  # 'r-'表示红色实线
            fig.canvas.draw()
            fig.canvas.flush_events()

def main():
    for _ in range(100):  # 模拟100次数据更新
        new_index = next(index)
        new_angle = np.random.randint(0, 180)  # 随机生成一个新的角度值，模拟数据更新

        angles.append(new_angle)
        indices.append(new_index)

        time.sleep(0.1)  # 模拟数据更新间隔

# 创建并启动绘图线程
plot_thread = threading.Thread(target=plot_angles)
plot_thread.start()

if __name__ == "__main__":
    main()
