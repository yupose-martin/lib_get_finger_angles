"""結合collect angle data while performing stimulation"""
import socket, threading, time
import numpy as np
import lib_get_finger_angles as gfa
import matplotlib.pyplot as plt

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#ipv4,udp
sock.bind(('10.26.196.17',54377))#UDP服务器端口和IP绑定
print('等待客户端发送请求...')
buf, addr = sock.recvfrom(40960)#等待matlab发送请求，这样就能获取matlab client的ip和端口号
print(f"addr is {addr}")
print(f"buf is {buf}")

mAStart = 25
mAEnd = 45
list_mA = []

list_thumb_angle = []
list_index_angle = []
list_middle_angle = []
list_ring_angle = []
list_pinky_angle = []

list_thumb_curve = []
list_index_curve = []
list_middle_curve = []
list_ring_curve = []
list_pinky_curve = []



list_angles = {1: list_thumb_angle,
               2: list_index_angle,
               3: list_middle_angle,
               4: list_ring_angle,
               5: list_pinky_angle}

list_curve = {1: list_thumb_curve,
               2: list_index_curve,
               3: list_middle_curve,
               4: list_ring_curve,
               5: list_pinky_curve}

def plot_angles(list_mA:"list",list_angles:"list[list]",list_curve:"list[list]"):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_xlim(mAStart, mAEnd)  # 设置初始x轴范围
    ax.set_ylim(-60,60)  # 假设角度在0到180度之间

    while True:
        # 这里可以添加一个条件来优雅地停止循环
        plt.pause(0.2)  # 稍微暂停以便更新图表
        if len(list_mA) > (mAEnd - mAStart):  # 如果超过10个数据点，则动态调整x轴的范围
            ax.set_xlim(list_mA[-(mAEnd - mAStart)], list_mA[-1] + 1)
            pass
        if list_angles:
            #ax.plot(list_mA, list_angles[1], 'r-')  # 'r-'表示红色实线
            ax.plot(list_mA, list_angles[2], 'g-',label = "index angle")
            ax.plot(list_mA, list_curve[2],'r-', label = "index curve")
            #ax.plot(list_mA, list_angles[3], 'b-')
            #ax.plot(list_mA, list_angles[4], 'b-')
            #ax.plot(list_mA, list_angles[5], 'b-')
            
            fig.canvas.draw()
            fig.canvas.flush_events()

# 创建并启动绘图线程
plot_thread = threading.Thread(target=plot_angles,args=(list_mA,list_angles,list_curve))
plot_thread.start()


def apply_stimulation(mA:"float"):
    global sock, buf, addr
    data = [mA]
    s = str(data)
    print(s)
    sock.sendto(bytes(s, encoding = "utf8") ,addr)#将数据转为bytes发送给matlab的client
    print('服务器端已发送')
    # print('正在等待接收客户端信息...')
    # buf, addr = sock.recvfrom(40960)
    # msg = buf.split()
    # print([np.double(i) for i in msg])
    pass
    
def main():
    time.sleep(5)
    mA = mAStart
    while (mA < mAEnd):
        time.sleep(3)
        apply_stimulation(mA=mA)
        time.sleep(9)
        list_mA.append(mA)
        for i in range(1,6):
            list_angles[i].append(gfa.get_finger_angle(i))
            list_curve[i].append(gfa.get_finger_curve(i))
        apply_stimulation(0)
        
        print(f"current mA: {mA}")
        mA += 1
    print(f"list_mA: {list_mA}")
    print(f"list_angles: {list_angles}")
    print(f"list_curve: {list_curve}")
    sock.close()
    return

if __name__ == "__main__":
    main()
