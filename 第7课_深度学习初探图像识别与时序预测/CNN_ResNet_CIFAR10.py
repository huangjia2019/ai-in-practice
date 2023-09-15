# 1. 导入所需要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 2. 下载CIFAR-10数据集
# 设置图像预处理: 图像增强 + 转换为张量 + 标准化
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 下载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# 3. 使用ResNet-18作为预训练网络
# 下载预训练的ResNet-18模型
resnet18 = torchvision.models.resnet18(pretrained=True)

# 由于CIFAR-10有10个类，我们需要调整ResNet的最后一个全连接层
num_classes = 10
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# 4. 微调预训练的CNN网络
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

# 迁移到GPU上（如果有的话）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18.to(device)

# 训练网络
for epoch in range(10):  # 就演示训练10个epochs

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data[0].to(device), data[1].to(device)

        # 清零参数梯度
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # 每200批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Training Finished')

# 5. 测试网络性能
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

