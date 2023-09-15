import torch
import torchvision
from torchvision import transforms

# 定义图像的预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 如果使用ResNet，则通常需要这个尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用ImageFolder加载数据
dataset_root = './dataset_root'  # 你的数据集的根目录
dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)

# 创建数据加载器
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据加载器进行迭代
for images, labels in data_loader:
    # 这里进行模型的训练/评估等操作
    pass