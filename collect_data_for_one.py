import socket, threading, time
import numpy as np
import lib_get_finger_angles as gfa
import matplotlib.pyplot as plt

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#ipv4,udp
sock.bind(('10.32.65.164',54377))#UDP服务器端口和IP绑定
print('等待客户端发送请求...')
buf, addr = sock.recvfrom(40960)#等待matlab发送请求，这样就能获取matlab client的ip和端口号
print(addr)

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
    ax.set_xlim(7, 15)  # 设置初始x轴范围
    ax.set_ylim(-60,60)  # 假设角度在0到180度之间

    while True:
        # 这里可以添加一个条件来优雅地停止循环
        plt.pause(0.2)  # 稍微暂停以便更新图表
        if len(list_mA) > 30:  # 如果超过10个数据点，则动态调整x轴的范围
            ax.set_xlim(list_mA[-30], list_mA[-1] + 1)
            pass
        if list_angles:
            #ax.plot(list_mA, list_angles[1], 'r-')  # 'r-'表示红色实线
            ax.plot(list_mA, list_angles[2], 'g-')
            #ax.plot(list_mA, list_angles[3], 'b-')
            #ax.plot(list_mA, list_angles[4], 'b-')
            #ax.plot(list_mA, list_angles[5], 'b-')
            
            fig.canvas.draw()
            fig.canvas.flush_events()

# 创建并启动绘图线程
plot_thread = threading.Thread(target=plot_angles,args=(list_mA,list_angles))
plot_thread.start()


def apply_stimulation(mA:"float"):
    global sock, buf, addr
    data = [mA]
    s = str(data)
    print(s)
    sock.sendto(bytes(s, encoding = "utf8") ,addr)#将数据转为bytes发送给matlab的client
    print('服务器端已发送')
    print('正在等待接收客户端信息...')
    buf, addr = sock.recvfrom(40960)
    msg = buf.split()
    print([np.double(i) for i in msg])
    pass
    
def main():
    time.sleep(5)
    mA = 7.0
    while (mA < 10.5):
        time.sleep(4)
        apply_stimulation(mA=mA)
        time.sleep(4)
        list_mA.append(mA)
        for i in range(1,6):
            list_angles[i].append(gfa.get_finger_angle(i))
        apply_stimulation(0)
        
        print(f"current mA: {mA}")
        print(f"list_angles: {list_angles}")
        mA += 0.5
    print(f"list_mA: {list_mA}")
    print(f"list_angles: {list_angles}")
    sock.close()
    return

if __name__ == "__main__":
    main()
