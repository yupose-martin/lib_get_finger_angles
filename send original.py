import socket
import time
import numpy as np

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#ipv4,udp
sock.bind(('10.32.65.164',54377))#UDP服务器端口和IP绑定
print('等待客户端发送请求...')
buf, addr = sock.recvfrom(40960)#等待matlab发送请求，这样就能获取matlab client的ip和端口号
print(addr)
count = 0

while True:
    for i in range (0,8): #9 从9.5开始 0.3 0.3 -> 12
        time.sleep(5) # 为了方便调试，这里是每隔一秒发送一次
        #a = np.random.randint(15)
        mA = 9.5 + (0.3*i)
        data = [mA]
        s = str(data)
        print(s)
        sock.sendto(bytes(s, encoding = "utf8") ,addr)#将数据转为bytes发送给matlab的client
        print('服务器端已发送')
        print('正在等待接收客户端信息...')
        buf, addr = sock.recvfrom(40960)
        msg = buf.split()
        print([np.double(i) for i in msg])
        count = count +2
        # msg = buf.split()
        # final=[np.double(i) for i in msg]
        
        #time.sleep(5) # 为了方便调试，这里是每隔一秒发送一次
        #a = np.random.randint(15)
        
        time.sleep(3)
        data = [0]
        s = str(data)
        print(s)
        sock.sendto(bytes(s, encoding = "utf8") ,addr)#将数据转为bytes发送给matlab的client
        print('服务器端已发送')
        print('正在等待接收客户端信息...')
        buf, addr = sock.recvfrom(40960)
        msg = buf.split()
        print([np.double(i) for i in msg])
        count = count +2
        # msg = buf.split()
        # final=[np.double(i) for i in msg]
sock.close()

