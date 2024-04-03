clc
clear all;
fes = FES('COM14', 38400, 0);
setFreqGlobal(fes, 99);

pulse_width = 40;
%count = 0;
data_all=[];
u = udp('10.26.196.17', 54377, 'Timeout', 60,'InputBufferSize',10240);
fopen(u);
fwrite(u,'get');
%[1]
while(1)
    disp('正在等待服务器发送数据...');
    receive = fread(u, 40960);
    data=str2num(char(receive(2:end-1)'));
    data_all = [data_all,data];
    r = data(1);
    if r < 45
        setAmpPwidthSingle(fes, 6, r, pulse_width);
    end
    pause(0.0001); 
    data
end
fclose(u);
delete(u);

