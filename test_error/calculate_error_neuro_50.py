import matplotlib.pyplot as plt
import math

# list_mA = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# list_thumb_curve = [-3.5344114015615276, -3.492737588040626, -3.4571285070294664, -3.107891076735797, -2.4044780990377888, -1.6597574047864612, -0.6329304379364373, 0.25068799258141894, 1.943674282694758, -1.073826155033828, 0.435478492281959, 0.6407536829906348, -1.3693247929187464, 0.826051577296937, -1.0183049657326535]

# list_index_curve = [1.4201054528491426, 1.1812093964209538, 1.1667572928549674, 1.1870136009705305, 5.147593614668978, 4.615831365346125, 9.434748781670024, 7.9186550164705665, 11.490477005759814, 20.20485601855033, 31.780246728544753, 31.3854069982127, 32.717439590643764, 28.987750674452812, 28.30054802891958]

# plt.figure()
# plt.plot(list_mA,list_index_curve,color='r')
# plt.plot(list_mA, list_thumb_curve, color="b")
# plt.show()


import numpy as np
import numpy.random as rd

# 初始化权重和偏置
w = 2.7
b = -2.7
number_of_iterations = 10000

# 输入数据和对应的输出数据
inputs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])  # 用你的输入数据替换，输入数据应该归一化到[-1, 1]
inputs = inputs / 25
outputs = np.array([1.4201054528491426, 1.1812093964209538, 1.1667572928549674, 1.1870136009705305, 5.147593614668978, 4.615831365346125, 9.434748781670024, 7.9186550164705665, 11.490477005759814, 20.20485601855033, 31.780246728544753, 31.3854069982127, 32.717439590643764, 28.987750674452812, 28.30054802891958])  # 用你的输出数据替换，应该归一化到[-1, 1]


def extract_data(list):#list：mA数字
    output_data = np.array([])
    for i in list:
        print(f"i is:{i}")
        index = i - 10
        print(f"output is: {outputs[index]}")
        output_data = np.append(output_data,outputs[index])
    print(f"output_data is: {output_data}")
    return output_data
        
list = np.array([10, 12, 16, 20])        

def get_mean_error(list1:'list', list2:'list'):
    error = 0
    for i in range(12):
        temp = (list2[i] - list1[i])
        error += temp * temp 
    error = error/ 12
    error = math.sqrt(error)
    return error
        
# for i in range(25,40):
#     inputs =  np.append(inputs,i/25)
#     outputs = np.append(outputs,30)
# # 学习率
learning_rate = 0.01

def get_neuro_output(inputs, w, b, outputs, allinputs):
    # 激活函数
    def tanh(x):
        return np.tanh(x)

    # 激活函数的导数
    def tanh_derivative(x):
        return 1.0 - np.tanh(x)**2

    # 输出转换函数
    def scale_output(tanh_output):
        return (tanh_output + 1) * 45  # 将(-1, 1)映射到(0, 90)
    loss = 0
    # 训练过程
    for i in range(number_of_iterations):
        # 正向传播，计算预测值
        weighted_sum = w * inputs + b
        predictions = scale_output(tanh(weighted_sum))
        
        # 计算损失，这里直接使用MSE
        loss = np.mean((predictions - outputs) ** 2)
        
        # 反向传播，计算梯度
        # 对于tanh激活函数，需要使用它的导数
        derivative_predictions = tanh_derivative(weighted_sum)
        # 注意: 这里需要对错误的梯度进行缩放
        error = (predictions - outputs) / 45
        d_loss_predictions = error * derivative_predictions
        dw = np.dot(d_loss_predictions, inputs) / len(inputs)
        db = np.mean(d_loss_predictions)
        
        # 更新权重和偏置
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 可选：打印过程中的损失值
        if i % 100 == 0:  # 每100次迭代打印一次
            print(f"Iteration {i}, Loss: {loss}")

    # 最终权重和偏置
    print(f"Trained weight: {w}, Trained bias: {b}")

    neuro_output = []
    for i in range(len(allinputs)):
        temp = allinputs[i] * w + b
        output = scale_output(tanh(temp))
        neuro_output.append(output)
        
    return neuro_output,loss

#3
list_3 = np.array([13,16,20])
output_3 = extract_data(list_3)
list_3 = list_3 / 25
neuro_output_3,loss3 = get_neuro_output(list_3 ,w, b, output_3, inputs)   

#4
list_4 = np.array([11,13,16,20])
output_4 = extract_data(list_4)
list_4 = list_4 / 25
neuro_output_4,loss4 = get_neuro_output(list_4,w,b,output_4, inputs)

#5
list_5 = np.array([11,13,16,18,20])
output_5 = extract_data(list_5)
list_5 = list_5 / 25
neuro_output_5,loss5 = get_neuro_output(list_5,w,b,output_5, inputs) 

#6
list_6 = np.array([11,13,16,18,20,22])
output_6 = extract_data(list_6)
list_6 = list_6 / 25
neuro_output_6,loss6 = get_neuro_output(list_6,w,b,output_6, inputs)  

neuro_output_all,lossall = get_neuro_output(inputs,w,b,outputs,inputs)
plt.figure()
inputs = inputs * 25 #画图的时候就回来
plt.plot(inputs, outputs,label=f'origin')
plt.plot(inputs, neuro_output_all,label='all')
plt.plot(inputs, neuro_output_3,label='3')
plt.plot(inputs,neuro_output_4,label='4')
plt.plot(inputs,neuro_output_5,label='5')
plt.plot(inputs,neuro_output_6,label='6')

plt.legend()
plt.show()

list_loss_index = [3, 4, 5, 6, 14]
list_loss = [loss3,loss4,loss5, loss6,lossall]

list_loss[0] = get_mean_error(outputs,neuro_output_3)
list_loss[1] = get_mean_error(outputs,neuro_output_4)
list_loss[2] = get_mean_error(outputs,neuro_output_5)
list_loss[3] = get_mean_error(outputs,neuro_output_6)
list_loss[4] = get_mean_error(outputs,neuro_output_all)

plt.figure()
plt.plot(list_loss_index, list_loss, label ='loss')  
plt.legend()
plt.show()


