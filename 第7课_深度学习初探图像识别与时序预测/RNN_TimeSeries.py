import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


df_app = pd.read_csv('AppRNN.csv', index_col='Date', parse_dates=['Date']) #导入数据

# 按照2020年10月1日为界拆分数据集
Train = df_app[:'2020-09-30'].iloc[:,0:1].values #训练集
Test = df_app['2020-10-01':].iloc[:,0:1].values #测试集

from sklearn.preprocessing import MinMaxScaler #导入归一化缩放器
Scaler = MinMaxScaler(feature_range=(0,1)) #创建缩放器
Train = Scaler.fit_transform(Train) #拟合缩放器并对训练集进行归一化

# 对测试数据进行归一化处理
Test = Scaler.transform(Test)



# 创建一个函数，将数据集转化为时间序列格式
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

# 设定窗口大小
seq_length = 4

x_train, y_train = sliding_windows(Train, seq_length)
# 使用滑动窗口为测试数据创建特征和标签
x_test, y_test = sliding_windows(Test, seq_length)

# 将数据转化为torch张量
trainX = Variable(torch.Tensor(np.array(x_train)))
trainY = Variable(torch.Tensor(np.array(y_train)))
# 将数据转化为torch张量
testX = Variable(torch.Tensor(np.array(x_test)))
testY = Variable(torch.Tensor(np.array(y_test)))

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层，用于输出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐状态
        h0 = Variable(torch.zeros(num_layers, x.size(0), self.hidden_size))
        
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        
        # 解码RNN的最后一个隐藏层的输出
        out = self.fc(out[:, -1, :])
        
        return out

# 设置模型参数
input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1

# 创建模型
rnn = RNN(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()    # 均方误差
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)   # Adam优化器

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    outputs = rnn(trainX)
    optimizer.zero_grad()
    
    # 计算损失
    loss = criterion(outputs, trainY)
    loss.backward()
    
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# 使用训练好的模型进行预测
rnn.eval() # 设置模型为评估模式
test_outputs = rnn(testX)

# 将预测结果逆归一化
test_outputs = test_outputs.data.numpy()
test_outputs = Scaler.inverse_transform(test_outputs) # 逆归一化

# 真实测试标签逆归一化
y_test_actual = Scaler.inverse_transform(y_test)

# 输出预测和真实结果
for i in range(len(y_test)):
    print(f"Date: {df_app['2020-10-01':].index[i+seq_length]}, Actual Activation: {y_test_actual[i][0]}, Predicted Activation: {test_outputs[i][0]}")


