import os
import torch
import torchvision
import torchvision.transforms as transforms

# 获取当前工作目录
current_dir = os.getcwd()

# 设置数据集的下载路径
dataset_path = os.path.join(current_dir, 'data1', 'cifar-10-batches-py')

# 定义数据预处理
transform = transforms.Compose([
    # 随机裁剪和随机水平翻转（数据增强）
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 随机裁剪为32x32的图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转

    # 标准化（减去均值，除以标准差）
    transforms.ToTensor(),  # 将图片转换为Tensor格式，值范围[0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化，减去均值，除以标准差
])

# 加载训练集（包含数据增强）并设置 download=True 来自动下载数据集
trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)

# 加载测试集（没有数据增强，只进行标准化）
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 仅标准化
                                       ]))

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 打印数据集的大小和示例
print(f"Training data size: {len(trainset)}")
print(f"Testing data size: {len(testset)}")

# 获取一些训练数据并查看样本
data_iter = iter(trainloader)
images, labels = data_iter.next()

# 打印图像尺寸和标签
print(f"Image batch shape: {images.shape}")
print(f"Labels batch shape: {labels.shape}")
# 确保在主模块中运行
if __name__ == '__main__':
    main()