import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import math

# 确保我们使用的是TensorFlow 2.x
assert tf.__version__.startswith('2.')

def extract_data(list):#list：mA数字
    output_data = np.array([])
    for i in list:
        print(f"i is:{i}")
        index = i - 10
        print(f"output is: {outputs[index]}")
        output_data = np.append(output_data,outputs[index])
    print(f"output_data is: {output_data}")
    return output_data

# 输入数据和对应的输出数据
# inputs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=np.float32)
# outputs = np.array([1.4201054528491426, 1.1812093964209538, 1.1667572928549674, 1.1870136009705305, 5.147593614668978,
#                     4.615831365346125, 9.434748781670024, 7.9186550164705665, 11.490477005759814, 20.20485601855033,
#                     31.780246728544753, 31.3854069982127, 32.717439590643764, 28.987750674452812,
#                     28.30054802891958], dtype=np.float32)

inputs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=np.float32)
outputs = np.array([1.4201054528491426, 1.1812093964209538, 1.1667572928549674, 1.1870136009705305, 5.147593614668978,
                    4.615831365346125, 9.434748781670024, 7.9186550164705665, 11.490477005759814, 20.20485601855033,
                    31.780246728544753, 31.3854069982127, 32.717439590643764, 28.987750674452812,
                    28.30054802891958], dtype=np.float32)

inputs = inputs / 25
outputs = outputs / 35

#3
list_3 = np.array([13,16,20])
output_3 = extract_data(list_3)
list_3 = list_3 / 25
output_3 = output_3 / 35

#4
list_4 = np.array([11,13,16,20])
output_4 = extract_data(list_4)
list_4 = list_4 / 25
output_4 = output_4 / 35

#5
list_5 = np.array([11,13,16,18,20])
output_5 = extract_data(list_5)
list_5 = list_5 / 25
output_5 = output_5 / 35

#6
list_6 = np.array([11,13,16,18,20,22])
output_6 = extract_data(list_6)
list_6 = list_6 / 25
output_6 = output_6 / 35

list_input = {
    3:list_3,
    4:list_4,
    5:list_5,
    6:list_6,
}

list_output = {
    3:output_3,
    4:output_4,
    5:output_5,
    6:output_6,
}

weights1 = np.array([[-0.52573574,  0.400464,   -1.0179603,  -0.8203471,   0.9289895 ]])
bias1 = np.array([[-0.00150385,  0.14876403, -0.30792114,  0.47067845, -0.71107167]])
weights2 = np.array([[ 0.257242  ,  0.0515947 ],
 [ 0.5432388 ,  0.1188819 ],
 [-0.33547947 ,-0.42779607],
 [-0.8150204 ,  1.1678075 ],
 [-0.6459112 , -1.2526146 ]])
bias2 = np.array([0.30867076, 0.09970608])
weight3 = np.array([[ 0.84576285],
 [-1.5881265 ]])
bias3 = np.array([0.55021346])
# 模型定义：一个输入层，两个隐藏层神经元，一个输出层
model = keras.Sequential([
    keras.layers.Dense(units=5, input_shape=(1,), activation='tanh',
                       kernel_initializer=keras.initializers.Constant(weights1),
                       bias_initializer=keras.initializers.Constant(bias1)),
    keras.layers.Dense(units=2, activation='tanh',
                       kernel_initializer=keras.initializers.Constant(weights2),
                       bias_initializer=keras.initializers.Constant(bias2)),
    keras.layers.Dense(units=1, activation='tanh',
    kernel_initializer=keras.initializers.Constant(weight3),
                       bias_initializer=keras.initializers.Constant(bias3),)
])

# 编译模型：使用均方误差作为损失函数，随机梯度下降作为优化器
model.compile(optimizer='sgd', loss='mse')

# 训练模型
history = model.fit(list_3, output_3, epochs=1000, verbose=0)

# 使用训练后的模型进行预测
predicted = model.predict(inputs)
mA = np.array([0.5])
predicted_angle = model.predict(mA)
print(f"mA cause a angle: {predicted_angle}")

mse_error = 0
for i in range(len(predicted)):
    tmp = outputs[i] - predicted[i]
    mse_error += tmp * tmp
mse_error = mse_error / len(predicted)
mse_error = math.sqrt(mse_error)
print(f"mse_errpr = {mse_error}")

# 获取模型的初始权重和偏置
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"layer = {layer}")
    print("Weights:", weights)
    print("Biases:", biases)

# 绘制实际和预测的输出
plt.figure(figsize=(10, 6))
plt.plot(inputs, outputs, 'ro', label='Actual')
plt.plot(inputs, predicted, 'b-', label='Predicted')
plt.title('Actual vs. Predicted')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
