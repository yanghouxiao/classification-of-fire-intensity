# import os  # 导入操作系统模块，用于文件和目录操作
# import numpy as np  # 导入数值计算库，用于数学运算和数组操作
# import torch  # 导入PyTorch深度学习框架，提供张量计算和神经网络功能
# import torch.nn as nn  # 导入神经网络模块，包含各种神经网络层和函数
# import torch.optim as optim  # 导入优化器模块，包含各种优化算法
# import matplotlib.pyplot as plt  # 导入绘图库，用于数据可视化
# import pandas as pd  # 导入数据分析库，用于保存Excel文件
#
# from PIL import Image  # 导入图像处理库，用于图像打开和格式转换
# from tqdm import tqdm  # 导入进度条显示库，用于训练过程可视化
# from ultralytics import YOLO  # 导入YOLO模型，用于目标检测
# from torchvision import transforms  # 导入图像预处理变换模块，用于图像变换
# from smallest_unet_model import smallest_UNet  # 导入自定义的UNet模型，用于火焰分割
#
# epochs = 50  # 定义训练轮次数
# batch_size = 16  # 定义每个批次的样本数
# learning_rate = 1e-3  # 定义学习率
# threshold_unet = 0.5007  # 定义UNet分割的阈值
# enter_pic_size_width = 640  # 定义输入图像的宽度
# enter_pic_size_length = 640  # 定义输入图像的长度
# save_dir = "pt_yangnet/yangnet_pt"  # 定义模型和结果保存目录
#
# # ===============================
# # 设备配置
# # ===============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU可用性并选择设备
#
# fire_seg_model = smallest_UNet().to(device)  # 创建火焰分割UNet模型并转移到指定设备
# fire_seg_model.load_state_dict(  # 加载预训练模型权重
#     torch.load("pt_yangnet/unet_pt/UNet_student_47.pt", map_location=device)  # 从文件加载权重并映射到设备
# )
# fire_seg_model.eval()  # 设置模型为评估模式（不计算梯度）
#
# # ===============================
# # YOLOv11（场景尺度检测）
# # ===============================
# scene_model = YOLO("pt_yangnet/scene_size_pt/best.pt")  # 加载YOLO场景检测模型
#
# # ===============================
# # 场景尺度判定函数
# # ===============================
# def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 定义场景分类函数，默认天空ID=0，树干ID=1
#     has_sky = False  # 初始化天空检测标志
#     has_trunk = False  # 初始化树干检测标志
#
#     for cid in cls_ids:  # 遍历检测到的所有类别ID
#         if int(cid) == sky_id:  # 如果检测到天空
#             has_sky = True  # 设置天空标志为真
#         elif int(cid) == trunk_id:  # 如果检测到树干
#             has_trunk = True  # 设置树干标志为真
#
#     if has_trunk:  # 优先判断：如果检测到树干
#         return torch.tensor([1, 0, 0], dtype=torch.float32)  # 返回特写尺度编码[1,0,0]
#     if has_sky:  # 其次判断：如果检测到天空
#         return torch.tensor([0, 0, 1], dtype=torch.float32)  # 返回远景尺度编码[0,0,1]
#     return torch.tensor([0, 1, 0], dtype=torch.float32)  # 否则返回中景尺度编码[0,1,0]
#
# # ===============================
# # 火焰面积比例计算函数
# # ===============================
# def get_fire_area_ratio(img_tensor):  # 定义计算火焰面积比例的函数
#     with torch.no_grad():  # 不计算梯度（推理模式）
#         pred = fire_seg_model(img_tensor)[0, 0]  # 使用分割模型预测，取第一张图第一个通道
#         binary_mask = (pred > threshold_unet).float()  # 使用阈值0.5007二值化预测结果
#
#     return (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰像素比例并返回
#
# # ===============================
# # 火势大小分类FCNN模型（4-64-4）
# # ===============================
# class FireSizeFCNN(nn.Module):  # 定义火势大小分类的全连接神经网络
#     def __init__(self):  # 初始化函数
#         super().__init__()  # 调用父类初始化
#         self.net = nn.Sequential(  # 定义网络结构序列
#             nn.Linear(4, 64),  # 第一层：4维输入到64维隐藏层
#             nn.ReLU(),  # 激活函数：ReLU
#             nn.Linear(64, 4)  # 第二层：64维隐藏层到4维输出层
#         )
#
#     def forward(self, x):  # 前向传播函数
#         return self.net(x)  # 返回网络输出
#
# model = FireSizeFCNN().to(device)  # 创建FCNN模型实例并转移到设备
#
# # ===============================
# # 数据集类定义
# # ===============================
# class FireSizeDataset(torch.utils.data.Dataset):  # 定义自定义数据集类
#     def __init__(self, root_dir):  # 初始化函数
#         self.samples = []  # 初始化样本列表
#         self.label_map = {  # 定义标签映射字典
#             "no_fire": 0,  # 无火
#             "small_fire": 1,  # 小火
#             "mid_fire": 2,  # 中火
#             "big_fire": 3  # 大火
#         }
#
#         for cls, label in self.label_map.items():  # 遍历所有类别
#             folder = os.path.join(root_dir, cls)  # 构建类别文件夹路径
#             for f in os.listdir(folder):  # 遍历文件夹中的文件
#                 self.samples.append((os.path.join(folder, f), label))  # 添加（文件路径，标签）到样本列表
#
#         self.transform = transforms.Compose([  # 定义图像预处理流程
#             transforms.Resize((enter_pic_size_length, enter_pic_size_width)),  # 调整大小为640×640
#             transforms.ToTensor()  # 转换为张量
#         ])
#
#     def __len__(self):  # 返回数据集大小
#         return len(self.samples)  # 样本数量
#
#     def __getitem__(self, idx):  # 获取单个样本
#         img_path, label = self.samples[idx]  # 获取图像路径和标签
#         img = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
#
#         img_tensor = self.transform(img).unsqueeze(0).to(device)  # 应用预处理，增加批次维度并转移到设备
#
#         yolo_res = scene_model(img_path, verbose=False)[0]  # 使用YOLO模型进行场景检测
#         cls_ids = (  # 获取检测到的类别ID
#             yolo_res.boxes.cls.cpu().numpy()  # 从检测结果中提取类别并转到CPU
#             if yolo_res.boxes is not None else []  # 如果没有检测到任何物体则返回空列表
#         )
#         scene_onehot = scene_size_classification(cls_ids).to(device)  # 场景分类并转到设备
#
#         fire_ratio = get_fire_area_ratio(img_tensor)  # 计算火焰面积比例
#
#         feature = torch.cat([  # 拼接特征向量
#             scene_onehot,  # 3维场景编码
#             torch.tensor([fire_ratio], device=device)  # 1维火焰比例
#         ])
#
#         return feature, torch.tensor(label, dtype=torch.long)  # 返回特征和标签
#
# # ===============================
# # 训练配置
# # ===============================
# train_dataset = FireSizeDataset(  # 创建训练数据集实例
#     "yangnet/forest_fire_size_classification_dataset/train"  # 训练数据路径
# )
#
# train_dataloader = torch.utils.data.DataLoader(  # 创建数据加载器
#     train_dataset,  # 数据集
#     batch_size=batch_size,  # 批次大小
#     shuffle=True  # 是否打乱数据
# )
#
# criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器：Adam优化器
#
# # ===============================
# # 训练循环
# # ===============================
# train_losses = []  # 初始化训练损失列表
#
# model.train()  # 设置模型为训练模式
#
# for epoch in range(epochs):  # 遍历每个训练轮次
#     print(f"第{epoch + 1}轮训练开始")  # 打印当前训练轮次开始信息
#     epoch_loss = 0.0  # 初始化当前轮次损失
#
#     with tqdm(train_dataloader, unit="batch") as tepoch:  # 使用进度条包装数据加载器
#         for batch_features, batch_labels in tepoch:  # 遍历每个批次
#             batch_features = batch_features.to(device)  # 将特征转移到设备
#             batch_labels = batch_labels.to(device)  # 将标签转移到设备
#
#             optimizer.zero_grad()  # 清空梯度
#             predictions = model(batch_features)  # 前向传播，获取模型预测结果
#             loss = criterion(predictions, batch_labels)  # 计算预测结果与真实标签的损失
#             loss.backward()  # 反向传播，计算梯度
#             optimizer.step()  # 更新模型参数
#
#             epoch_loss += loss.item()  # 累加批次损失
#             tepoch.set_postfix(loss=loss.item())  # 在进度条显示当前批次损失
#
#     avg_loss = epoch_loss / len(train_dataloader)  # 计算当前轮次的平均损失
#     train_losses.append(avg_loss)  # 记录平均损失到列表
#
# # ===============================
# # 保存模型和损失数据（包括Excel格式）
# # ===============================
# os.makedirs(save_dir, exist_ok=True)  # 创建保存目录，如果不存在则创建
#
# torch.save(model.state_dict(), f"{save_dir}/fire_size_fcnn.pt")  # 保存模型权重到文件
#
# np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))  # 保存损失数据为npy格式
#
# df_loss = pd.DataFrame({  # 创建损失数据的数据框
#     "Epoch": np.arange(1, epochs + 1),  # 轮次数列
#     "Training_Loss": train_losses  # 训练损失列
# })
# df_loss.to_excel(  # 保存损失数据为Excel文件
#     f"{save_dir}/train_loss.xlsx",  # Excel文件路径
#     index=False  # 不保存索引列
# )
#
# # ===============================
# # 绘制损失曲线
# # ===============================
# plt.figure()  # 创建新图形
# plt.plot(range(1, epochs + 1), train_losses)  # 绘制损失曲线
# plt.xlabel("Epoch")  # 设置x轴标签
# plt.ylabel("Training Loss")  # 设置y轴标签
# plt.title("FCNN Fire Size Classification Training Loss")  # 设置图形标题
# plt.grid(True)  # 显示网格
# plt.savefig(f"{save_dir}/train_loss_curve.png", dpi=300)  # 保存图形为PNG文件，分辨率300dpi
# plt.close()  # 关闭图形
#
# print("✅ 所有模型输入已统一为 640×640×3，训练完成（loss 已保存为 Excel）")  # 打印完成信息


# import os  # 提供操作系统相关功能，如文件路径操作
# import numpy as np  # 数值计算库，用于处理数组数据
# import torch  # PyTorch深度学习框架
# import torch.nn as nn  # PyTorch神经网络模块
# import torch.optim as optim  # PyTorch优化器模块
# import matplotlib.pyplot as plt  # 绘图库，用于可视化
# import pandas as pd  # 数据处理库，用于读写Excel
#
# from PIL import Image  # 图像处理库
# from tqdm import tqdm  # 进度条库，显示训练进度
# from ultralytics import YOLO  # YOLO目标检测库
# from torchvision import transforms  # 图像预处理模块
# from smallest_unet_model import smallest_UNet  # 导入自定义的UNet模型
#
# # ===============================
# # 超参数配置
# # ===============================
# epochs = 50  # 训练总轮数
# batch_size = 16  # 每批数据大小
# learning_rate = 1e-3  # 学习率
# threshold_unet = 0.5007  # UNet分割阈值
# enter_pic_size_width = 640  # 输入图片宽度
# enter_pic_size_length = 640  # 输入图片高度
# save_dir = "pt_yangnet/yangnet_pt"  # 模型保存目录
#
# # ===============================
# # 设备配置
# # ===============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU可用性，选择设备
# print("Using device:", device)  # 打印使用的设备
#
# # ===============================
# # UNet 火焰分割模型
# # ===============================
# fire_seg_model = smallest_UNet().to(device)  # 初始化UNet模型并移动到设备
# fire_seg_model.load_state_dict(  # 加载预训练权重
#     torch.load("pt_yangnet/unet_pt/UNet_student_47.pt", map_location=device)
# )
# fire_seg_model.eval()  # 设置为评估模式，不更新梯度
#
# # ===============================
# # YOLO 场景尺度模型
# # ===============================
# scene_model = YOLO("pt_yangnet/scene_size_pt/best.pt")  # 加载YOLO场景分类模型
#
# # ===============================
# # 场景尺度判定函数
# # ===============================
# def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 根据检测类别判断场景距离
#     has_sky = False  # 初始化天空标志
#     has_trunk = False  # 初始化树干标志
#
#     for cid in cls_ids:  # 遍历所有检测到的类别
#         if int(cid) == sky_id:  # 如果检测到天空
#             has_sky = True  # 设置天空标志
#         elif int(cid) == trunk_id:  # 如果检测到树干
#             has_trunk = True  # 设置树干标志
#
#     if has_trunk:  # 有树干代表近距离
#         return torch.tensor([1, 0, 0], dtype=torch.float32)  # close编码
#     if has_sky:  # 有天空代表远距离
#         return torch.tensor([0, 0, 1], dtype=torch.float32)  # far编码
#     return torch.tensor([0, 1, 0], dtype=torch.float32)      # 否则为中距离
#
# # ===============================
# # 火焰面积比例计算
# # ===============================
# def get_fire_area_ratio(img_tensor):  # 计算火焰像素比例
#     with torch.no_grad():  # 不计算梯度
#         pred = fire_seg_model(img_tensor)[0, 0]  # 获取UNet预测结果
#         binary_mask = (pred > threshold_unet).float()  # 二值化处理
#     return (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰面积比例
#
# # ===============================
# # 火势大小 FCNN（4-64-4）
# # ===============================
# class FireSizeFCNN(nn.Module):  # 定义全连接神经网络
#     def __init__(self):  # 初始化函数
#         super().__init__()  # 调用父类初始化
#         self.net = nn.Sequential(  # 定义网络结构
#             nn.Linear(4, 64),  # 输入层到隐藏层
#             nn.ReLU(),  # 激活函数
#             nn.Linear(64, 4)  # 隐藏层到输出层
#         )
#
#     def forward(self, x):  # 前向传播
#         return self.net(x)  # 返回网络输出
#
# model = FireSizeFCNN().to(device)  # 实例化模型并移动到设备
#
# # ===============================
# # 数据集定义
# # ===============================
# class FireSizeDataset(torch.utils.data.Dataset):  # 自定义数据集类
#     def __init__(self, root_dir):  # 初始化函数
#         self.samples = []  # 存储样本列表
#         self.label_map = {  # 标签映射字典
#             "no_fire": 0,
#             "small_fire": 1,
#             "mid_fire": 2,
#             "big_fire": 3
#         }
#
#         for cls, label in self.label_map.items():  # 遍历所有类别
#             folder = os.path.join(root_dir, cls)  # 构建类别文件夹路径
#             for f in os.listdir(folder):  # 遍历文件夹内所有文件
#                 self.samples.append((os.path.join(folder, f), label))  # 添加样本路径和标签
#
#         self.transform = transforms.Compose([  # 定义图像预处理流程
#             transforms.Resize((enter_pic_size_length, enter_pic_size_width)),  # 调整大小
#             transforms.ToTensor()  # 转换为张量
#         ])
#
#     def __len__(self):  # 返回数据集大小
#         return len(self.samples)  # 样本数量
#
#     def __getitem__(self, idx):  # 获取单个样本
#         img_path, label = self.samples[idx]  # 获取图片路径和标签
#         img = Image.open(img_path).convert("RGB")  # 打开并转换图片为RGB格式
#
#         img_tensor = self.transform(img).unsqueeze(0).to(device)  # 预处理图片并添加批次维度
#
#         yolo_res = scene_model(img_path, verbose=False)[0]  # 运行YOLO检测
#         cls_ids = (  # 提取检测到的类别ID
#             yolo_res.boxes.cls.cpu().numpy()
#             if yolo_res.boxes is not None else []  # 处理无检测结果情况
#         )
#
#         scene_onehot = scene_size_classification(cls_ids).to(device)  # 获取场景编码
#         fire_ratio = get_fire_area_ratio(img_tensor)  # 计算火焰比例
#
#         feature = torch.cat([  # 拼接特征向量
#             scene_onehot,  # 场景编码
#             torch.tensor([fire_ratio], device=device)  # 火焰比例
#         ])
#
#         return feature, torch.tensor(label, dtype=torch.long)  # 返回特征和标签
#
# # ===============================
# # 数据加载器
# # ===============================
# train_dataset = FireSizeDataset(  # 训练数据集
#     "yangnet/forest_fire_size_classification_dataset/train"
# )
# val_dataset = FireSizeDataset(  # 验证数据集
#     "yangnet/forest_fire_size_classification_dataset/val"
# )
#
# train_dataloader = torch.utils.data.DataLoader(  # 训练数据加载器
#     train_dataset, batch_size=batch_size, shuffle=True  # 批量大小和打乱数据
# )
# val_dataloader = torch.utils.data.DataLoader(  # 验证数据加载器
#     val_dataset, batch_size=batch_size, shuffle=False  # 批量大小，不打乱数据
# )
#
# # ===============================
# # 训练配置
# # ===============================
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
#
# train_losses = []  # 存储训练损失
# val_losses = []  # 存储验证损失
#
# # ===============================
# # 训练 & 验证循环
# # ===============================
# for epoch in range(epochs):  # 遍历所有训练轮次
#     print(f"第{epoch + 1}轮训练开始")  # 打印当前训练轮次开始信息
#
#     # -------- Train --------
#     model.train()  # 设置为训练模式
#     train_loss_sum = 0.0  # 初始化训练损失累计
#
#     with tqdm(train_dataloader, unit="batch", desc="Train") as tepoch:  # 训练进度条
#         for features, labels in tepoch:  # 遍历训练批次
#             features = features.to(device)  # 特征移动到设备
#             labels = labels.to(device)  # 标签移动到设备
#
#             optimizer.zero_grad()  # 清空梯度
#             outputs = model(features)  # 前向传播
#             loss = criterion(outputs, labels)  # 计算损失
#             loss.backward()  # 反向传播
#             optimizer.step()  # 更新参数
#
#             train_loss_sum += loss.item()  # 累计损失
#             tepoch.set_postfix(train_loss=loss.item())  # 更新进度条显示
#
#     avg_train_loss = train_loss_sum / len(train_dataloader)  # 计算平均训练损失
#     train_losses.append(avg_train_loss)  # 保存训练损失
#
#     # -------- Val --------
#     model.eval()  # 设置为评估模式
#     val_loss_sum = 0.0  # 初始化验证损失累计
#
#     with torch.no_grad():  # 不计算梯度
#         with tqdm(val_dataloader, unit="batch", desc="Val") as vepoch:  # 验证进度条
#             for features, labels in vepoch:  # 遍历验证批次
#                 features = features.to(device)  # 特征移动到设备
#                 labels = labels.to(device)  # 标签移动到设备
#
#                 outputs = model(features)  # 前向传播
#                 loss = criterion(outputs, labels)  # 计算损失
#
#                 val_loss_sum += loss.item()  # 累计损失
#                 vepoch.set_postfix(val_loss=loss.item())  # 更新进度条显示
#
#     avg_val_loss = val_loss_sum / len(val_dataloader)  # 计算平均验证损失
#     val_losses.append(avg_val_loss)  # 保存验证损失
#
#     print(  # 打印训练信息
#         f"Epoch {epoch + 1}: "
#         f"Train Loss = {avg_train_loss:.6f}, "
#         f"Val Loss = {avg_val_loss:.6f}"
#     )
#
# # ===============================
# # 保存模型 & Loss（Excel）
# # ===============================
# os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#
# torch.save(model.state_dict(), f"{save_dir}/fire_size_fcnn.pt")  # 保存模型权重
#
# df_loss = pd.DataFrame({  # 创建损失数据框
#     "Epoch": np.arange(1, epochs + 1),  # 训练轮次
#     "Train_Loss": train_losses,  # 训练损失
#     "Val_Loss": val_losses  # 验证损失
# })
#
# df_loss.to_excel(  # 保存为Excel文件
#     f"{save_dir}/train_val_loss.xlsx",
#     index=False  # 不保存索引
# )
#
# # ===============================
# # 绘制 Loss 曲线
# # ===============================
# plt.figure()  # 创建图形
# plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")  # 绘制训练损失曲线
# plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")  # 绘制验证损失曲线
# plt.xlabel("Epoch")  # X轴标签
# plt.ylabel("Loss")  # Y轴标签
# plt.title("FCNN Fire Size Classification - Train & Val Loss")  # 图形标题
# plt.legend()  # 显示图例
# plt.grid(True)  # 显示网格
# plt.savefig(f"{save_dir}/train_val_loss_curve.png", dpi=300)  # 保存图形
# plt.close()  # 关闭图形
#
# print("✅ 训练与验证完成")  # 完成提示
# print("📊 Loss 已保存为 Excel")  # 保存提示
# print("📈 Loss 曲线已保存为 PNG")  # 图形保存提示


# import os  # 提供操作系统相关功能
# import numpy as np  # 数值计算库，用于数组操作
# import torch  # PyTorch深度学习框架主库
# import torch.nn as nn  # PyTorch神经网络模块
# import torch.optim as optim  # PyTorch优化器模块
# import pandas as pd  # 数据处理库，用于处理Excel文件
# import matplotlib.pyplot as plt  # 数据可视化库，用于绘图
#
# from PIL import Image  # Python图像处理库
# from tqdm import tqdm  # 进度条显示库
# from ultralytics import YOLO  # YOLO目标检测模型库
# from torchvision import transforms  # PyTorch图像变换模块
# from smallest_unet_model import smallest_UNet  # 自定义的最小UNet模型
#
# # ===============================
# # 超参数
# # ===============================
# epochs = 50  # 训练总轮数
# learning_rate = 1e-3  # 学习率
# img_size = 640  # 图像尺寸（宽高相同）
# batch_size = 16  # 每个批次的样本数量
# threshold_unet = 0.5007  # UNet分割阈值，用于二值化掩膜）
# save_dir = "pt_yangnet/yangnet_pt"  # 模型权重保存目录
# pt_unet = "pt_yangnet/unet_pt/UNet_student_47.pt"  # UNet模型权重路径
# pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"  # YOLO场景模型权重路径
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备（GPU或CPU）
# print("Using device:", device)  # 打印当前使用的设备
#
# # ===============================
# # UNet（仅离线使用）
# # ===============================
# fire_seg_model = smallest_UNet().to(device)  # 初始化UNet模型并转移到指定设备
# fire_seg_model.load_state_dict(  # 加载预训练的UNet模型权重
#     torch.load(pt_unet, map_location=device)  # 使用变量路径加载权重文件
# )
# fire_seg_model.eval()  # 将模型设置为评估模式（不训练）
#
# unet_transform = transforms.Compose([  # 定义图像预处理流程
#     transforms.Resize((img_size, img_size)),  # 调整图像大小
#     transforms.ToTensor()  # 将图像转换为张量
# ])
#
# # ===============================
# # YOLO 场景尺度模型
# # ===============================
# scene_model = YOLO(pt_scene_size)  # 加载预训练的YOLO场景分类模型，使用变量路径
#
# # ===============================
# # 场景尺度编码
# # ===============================
# def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 定义场景分类函数，参数为检测到的类别ID列表
#     has_sky, has_trunk = False, False  # 初始化标志变量
#     for cid in cls_ids:  # 遍历所有检测到的类别ID
#         if int(cid) == sky_id:  # 如果检测到天空类别
#             has_sky = True  # 设置天空标志为真
#         elif int(cid) == trunk_id:  # 如果检测到树干类别
#             has_trunk = True  # 设置树干标志为真
#
#     if has_trunk:  # 如果存在树干
#         return torch.tensor([1, 0, 0], dtype=torch.float32)  # 返回近景编码向量
#     if has_sky:  # 如果存在天空
#         return torch.tensor([0, 0, 1], dtype=torch.float32)  # 返回远景编码向量
#     return torch.tensor([0, 1, 0], dtype=torch.float32)  # 默认返回中景编码向量
#
# # ===============================
# # 数据路径
# # ===============================
# dataset_roots = {  # 定义训练集和验证集的根目录
#     "train": "yangnet/forest_fire_size_classification_dataset/train",  # 训练集路径
#     "val": "yangnet/forest_fire_size_classification_dataset/val"  # 验证集路径
# }
# classes = ["no_fire", "small_fire", "mid_fire", "big_fire"]  # 类别名称列表
#
# # ===============================
# # 🔥 预计算火焰掩膜 & 面积比例
# # ===============================
# def precompute_fire_ratio_and_mask():  # 定义预计算函数
#     print("\n🔥 Precomputing fire masks & ratios")  # 打印开始预计算的信息
#
#     for split, root_dir in dataset_roots.items():  # 遍历训练集和验证集
#         for cls in classes:  # 遍历每个类别
#             img_dir = os.path.join(root_dir, cls)  # 拼接当前类别的图像目录路径
#             ratio_dir = os.path.join(img_dir, "fire_ratio")  # 拼接火焰比例保存目录路径
#             os.makedirs(ratio_dir, exist_ok=True)  # 创建火焰比例保存目录（如果不存在）
#
#             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls}"):  # 遍历当前类别所有图像文件，显示进度条
#                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
#                     continue  # 如果不是图像文件则跳过
#
#                 img_path = os.path.join(img_dir, img_name)  # 拼接完整图像路径
#                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")  # 拼接火焰比例文件路径
#                 mask_path = os.path.join(  # 拼接掩膜图像保存路径
#                     img_dir,
#                     os.path.splitext(img_name)[0] + "_mask.png"  # 在原文件名后添加"_mask"作为掩膜文件名
#                 )
#
#                 if os.path.exists(ratio_path) and os.path.exists(mask_path):  # 如果已经计算过火焰比例和掩膜
#                     continue  # 跳过已处理的图像
#
#                 image = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
#                 image_tensor = unet_transform(image).unsqueeze(0).to(device)  # 图像预处理并添加批次维度，转移到设备
#
#                 with torch.no_grad():  # 禁用梯度计算（推理阶段）
#                     prediction = fire_seg_model(image_tensor)[0, 0]  # 使用UNet模型进行预测，获取第一个通道的输出
#                     binary_mask = (prediction > threshold_unet).float()  # 根据阈值将预测结果二值化为掩膜
#
#                 fire_ratio = (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰面积比例（火焰像素数/总像素数）
#                 np.save(ratio_path, fire_ratio)  # 保存火焰比例到文件
#
#                 mask_image = (binary_mask.cpu().numpy() * 255).astype(np.uint8)  # 将掩膜转换为0-255范围的图像数组
#                 Image.fromarray(mask_image).save(mask_path)  # 保存掩膜图像为PNG格式
#
#     print("✅ Fire masks & ratios cached\n")  # 打印预计算完成信息
#
# # ===============================
# # FCNN
# # ===============================
# class FireSizeFCNN(nn.Module):  # 定义火焰大小分类的FCNN模型类
#     def __init__(self):  # 模型初始化函数
#         super().__init__()  # 调用父类初始化
#         self.network = nn.Sequential(  # 定义顺序网络结构
#             nn.Linear(4, 64),  # 第一层全连接层：输入4维特征，输出64维
#             nn.ReLU(),  # ReLU激活函数
#             nn.Linear(64, 4)  # 第二层全连接层：输出4维（对应4个类别）
#         )
#
#     def forward(self, features):  # 前向传播函数
#         return self.network(features)  # 将输入特征通过网络计算输出
#
# model = FireSizeFCNN().to(device)  # 创建FCNN模型实例并转移到设备
#
# # ===============================
# # 数据集（只读取缓存）
# # ===============================
# class FireSizeDataset(torch.utils.data.Dataset):  # 定义自定义数据集类
#     def __init__(self, root_dir):  # 数据集初始化函数
#         self.samples = []  # 初始化样本列表
#         self.label_map = {  # 定义类别名称到数字标签的映射
#             "no_fire": 0,
#             "small_fire": 1,
#             "mid_fire": 2,
#             "big_fire": 3
#         }
#
#         for cls, label in self.label_map.items():  # 遍历每个类别
#             cls_dir = os.path.join(root_dir, cls)  # 拼接当前类别的目录路径
#             for img_name in os.listdir(cls_dir):  # 遍历当前类别目录下的所有文件
#                 if img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
#                     self.samples.append(  # 将图像路径和标签添加到样本列表
#                         (os.path.join(cls_dir, img_name), label)
#                     )
#
#     def __len__(self):  # 返回数据集大小的方法
#         return len(self.samples)  # 返回样本数量
#
#     def __getitem__(self, index):  # 获取单个样本的方法
#         img_path, label = self.samples[index]  # 获取指定索引的图像路径和标签
#
#         ratio_path = os.path.join(  # 拼接火焰比例文件路径
#             os.path.dirname(img_path),  # 图像所在目录
#             "fire_ratio",  # 火焰比例子目录
#             os.path.basename(img_path) + ".npy"  # 火焰比例文件名（图像文件名+.npy）
#         )
#         fire_ratio = np.load(ratio_path).item()  # 从文件加载火焰比例值
#
#         yolo_result = scene_model(img_path, verbose=False)[0]  # 使用YOLO模型进行场景检测，获取第一个结果
#         class_ids = (  # 获取检测到的类别ID
#             yolo_result.boxes.cls.cpu().numpy()  # 从YOLO结果中提取类别ID并转换到CPU的numpy数组
#             if yolo_result.boxes is not None else []  # 如果未检测到任何目标，返回空列表
#         )
#
#         scene_feature = scene_size_classification(class_ids)  # 根据检测结果计算场景特征向量
#
#         feature_vector = torch.cat([  # 拼接特征向量
#             scene_feature,  # 3维场景特征
#             torch.tensor([fire_ratio])  # 1维火焰比例特征
#         ])
#
#         return feature_vector, torch.tensor(label, dtype=torch.long)  # 返回特征向量和标签
#
# # ===============================
# # 执行预计算
# # ===============================
# precompute_fire_ratio_and_mask()  # 调用预计算函数
#
# # ===============================
# # DataLoader
# # ===============================
# train_loader = torch.utils.data.DataLoader(  # 创建训练集数据加载器
#     FireSizeDataset(dataset_roots["train"]),  # 训练集数据集实例
#     batch_size=batch_size,  # 批次大小
#     shuffle=True  # 打乱数据顺序
# )
#
# val_loader = torch.utils.data.DataLoader(  # 创建验证集数据加载器
#     FireSizeDataset(dataset_roots["val"]),  # 验证集数据集实例
#     batch_size=batch_size,  # 批次大小
#     shuffle=False  # 不打乱验证集数据顺序
# )
#
# # ===============================
# # 训练与验证
# # ===============================
# criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义Adam优化器
#
# train_losses, val_losses = [], []  # 初始化训练和验证损失记录列表
#
# for epoch in range(epochs):  # 遍历每个训练轮次
#     model.train()  # 将模型设置为训练模式
#     epoch_train_loss = 0.0  # 初始化当前轮次训练损失
#
#     for train_features, train_labels in tqdm(train_loader, desc=f"Train {epoch+1}"):  # 遍历训练集批次，显示进度条
#         train_features = train_features.to(device)  # 将特征数据转移到设备
#         train_labels = train_labels.to(device)  # 将标签数据转移到设备
#
#         optimizer.zero_grad()  # 清空梯度缓存
#         predictions = model(train_features)  # 前向传播，获取模型预测
#         loss = criterion(predictions, train_labels)  # 计算损失
#         loss.backward()  # 反向传播，计算梯度
#         optimizer.step()  # 更新模型参数
#
#         epoch_train_loss += loss.item()  # 累加批次损失
#
#     avg_train_loss = epoch_train_loss / len(train_loader)  # 计算当前轮次平均训练损失
#     train_losses.append(avg_train_loss)  # 记录训练损失
#
#     model.eval()  # 将模型设置为评估模式
#     epoch_val_loss = 0.0  # 初始化当前轮次验证损失
#     with torch.no_grad():  # 禁用梯度计算（验证阶段）
#         for val_features, val_labels in tqdm(val_loader, desc="Val"):  # 遍历验证集批次，显示进度条
#             val_features = val_features.to(device)  # 将特征数据转移到设备
#             val_labels = val_labels.to(device)  # 将标签数据转移到设备
#
#             predictions = model(val_features)  # 前向传播，获取模型预测
#             loss = criterion(predictions, val_labels)  # 计算损失
#             epoch_val_loss += loss.item()  # 累加批次损失
#
#     avg_val_loss = epoch_val_loss / len(val_loader)  # 计算当前轮次平均验证损失
#     val_losses.append(avg_val_loss)  # 记录验证损失
#
#     print(  # 打印当前轮次训练信息
#         f"Epoch {epoch+1}: "
#         f"Train Loss = {avg_train_loss:.6f}, "  # 训练损失（保留6位小数）
#         f"Val Loss = {avg_val_loss:.6f}"  # 验证损失（保留6位小数）
#     )
#
# # ===============================
# # 保存模型 & Loss
# # ===============================
# os.makedirs(save_dir, exist_ok=True)  # 创建模型保存目录（如果不存在）
# torch.save(model.state_dict(), f"{save_dir}/fire_size_fcnn.pt")  # 保存模型权重
#
# pd.DataFrame({  # 创建损失记录的数据框
#     "Epoch": range(1, epochs + 1),  # 轮次数列
#     "Train_Loss": train_losses,  # 训练损失列
#     "Val_Loss": val_losses  # 验证损失列
# }).to_excel(f"{save_dir}/train_val_loss.xlsx", index=False)  # 保存为Excel文件，不包含索引列
#
# # ===============================
# # Loss 曲线
# # ===============================
# plt.figure()  # 创建新图形
# plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")  # 绘制训练损失曲线
# plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")  # 绘制验证损失曲线
# plt.xlabel("Epoch")  # 设置x轴标签
# plt.ylabel("Loss")  # 设置y轴标签
# plt.title("Fire Size FCNN Training & Validation Loss")  # 设置图形标题
# plt.legend()  # 显示图例
# plt.grid(True)  # 显示网格线
# plt.savefig(f"{save_dir}/train_val_loss_curve.png", dpi=300)  # 保存图形为PNG格式，分辨率300dpi
# plt.close()  # 关闭图形










# #原先的简单模型
# import os  # 导入操作系统模块，用于文件和目录操作
# import numpy as np  # 导入NumPy库，用于数值计算
# import torch  # 导入PyTorch深度学习框架
# import torch.nn as nn  # 导入PyTorch神经网络模块
# import torch.optim as optim  # 导入PyTorch优化器模块
# import pandas as pd  # 导入Pandas库，用于数据处理
# import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
#
# from PIL import Image  # 导入PIL库，用于图像处理
# from tqdm import tqdm  # 导入tqdm库，用于显示进度条
# from ultralytics import YOLO  # 导入YOLO模型
# from torchvision import transforms  # 导入torchvision的图像变换模块
# from smallest_unet_model import smallest_UNet  # 导入自定义的UNet模型
#
# # ===============================
# # 超参数
# # ===============================
# epochs = 50  # 训练的总轮数
# learning_rate = 1e-3  # 学习率
# img_size = 640  # 图像尺寸
# batch_size = 16  # 批量大小
# threshold_unet = 0.5007  # UNet分割的阈值
#
# save_dir = "pt_yangnet/yangnet_pt"  # 保存结果的目录
# pt_unet = "pt_yangnet/unet_pt/UNet_student_47.pt"  # UNet模型路径
# pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"  # YOLO场景尺度模型路径
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU可用性并设置设备
# print("Using device:", device)  # 打印使用的设备
#
# # ===============================
# # UNet（离线分割）
# # ===============================
# fire_seg_model = smallest_UNet().to(device)  # 实例化UNet模型并移动到设备
# fire_seg_model.load_state_dict(torch.load(pt_unet, map_location=device))  # 加载预训练权重
# fire_seg_model.eval()  # 设置为评估模式
#
# unet_transform = transforms.Compose([  # 定义图像预处理转换
#     transforms.Resize((img_size, img_size)),  # 调整图像大小
#     transforms.ToTensor()  # 转换为张量
# ])
#
# # ===============================
# # YOLO 场景尺度模型
# # ===============================
# scene_model = YOLO(pt_scene_size)  # 加载YOLO模型
#
# # ===============================
# # 场景尺度编码
# # ===============================
# def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 定义场景尺度分类函数
#     has_sky, has_trunk = False, False  # 初始化天空和树干标志
#     for cid in cls_ids:  # 遍历检测到的类别ID
#         if int(cid) == sky_id:  # 如果检测到天空
#             has_sky = True  # 设置天空标志为True
#         elif int(cid) == trunk_id:  # 如果检测到树干
#             has_trunk = True  # 设置树干标志为True
#
#     if has_trunk:  # 如果检测到树干
#         return torch.tensor([1, 0, 0], dtype=torch.float32)  # 返回[1,0,0]表示近景
#     if has_sky:  # 如果检测到天空
#         return torch.tensor([0, 0, 1], dtype=torch.float32)  # 返回[0,0,1]表示远景
#     return torch.tensor([0, 1, 0], dtype=torch.float32)  # 否则返回[0,1,0]表示中景
#
# # ===============================
# # 数据路径
# # ===============================
# dataset_roots = {  # 定义数据集路径字典
#     "train": "yangnet/forest_fire_size_classification_dataset/train",  # 训练集路径
#     "val": "yangnet/forest_fire_size_classification_dataset/val"  # 验证集路径
# }
#
# classes = ["no_fire", "small_fire", "mid_fire", "big_fire"]  # 类别列表
# idx_to_class = {0: "no_fire", 1: "small_fire", 2: "mid_fire", 3: "big_fire"}  # 索引到类别的映射
#
# # ===============================
# # 预计算火焰比例 & 掩膜
# # ===============================
# def precompute_fire_ratio_and_mask():  # 定义预计算火焰比例和掩膜的函数
#     print("\n🔥 Precomputing fire masks & ratios")  # 打印开始预计算信息
#
#     for split, root_dir in dataset_roots.items():  # 遍历训练集和验证集
#         for cls in classes:  # 遍历每个类别
#             img_dir = os.path.join(root_dir, cls)  # 构建图像目录路径
#             ratio_dir = os.path.join(img_dir, "fire_ratio")  # 构建火焰比例保存目录路径
#             os.makedirs(ratio_dir, exist_ok=True)  # 创建目录（如果不存在）
#
#             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls}"):  # 遍历目录中的图像文件
#                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
#                     continue  # 跳过非图像文件
#
#                 img_path = os.path.join(img_dir, img_name)  # 构建完整图像路径
#                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")  # 构建火焰比例保存路径
#                 mask_path = os.path.join(  # 构建掩膜保存路径
#                     img_dir,
#                     os.path.splitext(img_name)[0] + "_mask.png"
#                 )
#
#                 if os.path.exists(ratio_path) and os.path.exists(mask_path):  # 如果已经计算过
#                     continue  # 跳过已处理的图像
#
#                 image = Image.open(img_path).convert("RGB")  # 打开并转换图像为RGB模式
#                 image_tensor = unet_transform(image).unsqueeze(0).to(device)  # 预处理图像并添加批次维度
#
#                 with torch.no_grad():  # 禁用梯度计算
#                     pred = fire_seg_model(image_tensor)[0, 0]  # 使用UNet进行预测
#                     binary_mask = (pred > threshold_unet).float()  # 应用阈值生成二值掩膜
#
#                 fire_ratio = (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰比例
#                 np.save(ratio_path, fire_ratio)  # 保存火焰比例到文件
#
#                 mask_img = (binary_mask.cpu().numpy() * 255).astype(np.uint8)  # 将掩膜转换为图像格式
#                 Image.fromarray(mask_img).save(mask_path)  # 保存掩膜图像
#
#     print("✅ Fire masks & ratios cached\n")  # 打印预计算完成信息
#
# # ===============================
# # FCNN
# # ===============================
# class FireSizeFCNN(nn.Module):  # 定义全连接神经网络类
#     def __init__(self):  # 初始化函数
#         super().__init__()  # 调用父类初始化
#         self.net = nn.Sequential(  # 定义网络结构
#             nn.Linear(4, 64),  # 输入层到隐藏层（4维输入，64维输出）
#             nn.ReLU(),  # ReLU激活函数
#             nn.Linear(64, 4)  # 隐藏层到输出层（64维输入，4维输出）
#         )
#
#     def forward(self, x):  # 前向传播函数
#         return self.net(x)  # 返回网络输出
#
# model = FireSizeFCNN().to(device)  # 实例化模型并移动到设备
#
# # ===============================
# # Dataset
# # ===============================
# class FireSizeDataset(torch.utils.data.Dataset):  # 定义自定义数据集类
#     def __init__(self, root_dir):  # 初始化函数
#         self.samples = []  # 初始化样本列表
#         self.label_map = {  # 定义标签映射字典
#             "no_fire": 0,
#             "small_fire": 1,
#             "mid_fire": 2,
#             "big_fire": 3
#         }
#
#         for cls, label in self.label_map.items():  # 遍历每个类别
#             cls_dir = os.path.join(root_dir, cls)  # 构建类别目录路径
#             for img_name in os.listdir(cls_dir):  # 遍历类别目录中的文件
#                 if img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
#                     self.samples.append(  # 添加到样本列表
#                         (os.path.join(cls_dir, img_name), label)  # (图像路径, 标签)
#                     )
#
#     def __len__(self):  # 返回数据集大小
#         return len(self.samples)
#
#     def __getitem__(self, idx):  # 获取单个样本
#         img_path, label = self.samples[idx]  # 获取图像路径和标签
#
#         ratio_path = os.path.join(  # 构建火焰比例文件路径
#             os.path.dirname(img_path),
#             "fire_ratio",
#             os.path.basename(img_path) + ".npy"
#         )
#         fire_ratio = np.load(ratio_path).item()  # 加载火焰比例
#
#         yolo_res = scene_model(img_path, verbose=False)[0]  # 使用YOLO进行场景检测
#         class_ids = (  # 获取检测到的类别ID
#             yolo_res.boxes.cls.cpu().numpy()  # 从YOLO结果中提取类别
#             if yolo_res.boxes is not None else []  # 如果没有检测到物体则为空列表
#         )
#
#         scene_feat = scene_size_classification(class_ids)  # 获取场景特征向量
#
#         feature = torch.cat([  # 拼接特征向量
#             scene_feat,  # 场景特征（3维）
#             torch.tensor([fire_ratio])  # 火焰比例（1维）
#         ])
#
#         return feature, torch.tensor(label, dtype=torch.long)  # 返回特征和标签
#
# # ===============================
# # 执行预计算
# # ===============================
# precompute_fire_ratio_and_mask()  # 调用预计算函数
#
# # ===============================
# # DataLoader
# # ===============================
# train_loader = torch.utils.data.DataLoader(  # 创建训练数据加载器
#     FireSizeDataset(dataset_roots["train"]),  # 训练数据集
#     batch_size=batch_size,  # 批量大小
#     shuffle=True  # 打乱数据
# )
#
# val_dataset = FireSizeDataset(dataset_roots["val"])  # 创建验证数据集
# val_loader = torch.utils.data.DataLoader(  # 创建验证数据加载器
#     val_dataset,  # 验证数据集
#     batch_size=batch_size,  # 批量大小
#     shuffle=False  # 不打乱数据
# )
#
# # ===============================
# # 训练与验证
# # ===============================
# criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器（Adam）
#
# train_losses, val_losses = [], []  # 初始化训练和验证损失列表
# val_results = []  # 初始化验证结果列表
#
# for epoch in range(epochs):  # 遍历每个训练轮次
#     # -------- Train --------
#     model.train()  # 设置为训练模式
#     train_loss = 0.0  # 初始化训练损失
#
#     for feats, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):  # 遍历训练批次
#         feats, labels = feats.to(device), labels.to(device)  # 移动数据到设备
#
#         optimizer.zero_grad()  # 清零梯度
#         outputs = model(feats)  # 前向传播
#         loss = criterion(outputs, labels)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#
#         train_loss += loss.item()  # 累加损失
#
#     train_losses.append(train_loss / len(train_loader))  # 计算平均训练损失
#
#     # -------- Val --------
#     model.eval()  # 设置为评估模式
#     val_loss = 0.0  # 初始化验证损失
#
#     with torch.no_grad():  # 禁用梯度计算
#         for bidx, (feats, labels) in enumerate(tqdm(val_loader, desc="Val")):  # 遍历验证批次
#             feats, labels = feats.to(device), labels.to(device)  # 移动数据到设备
#
#             outputs = model(feats)  # 前向传播
#             loss = criterion(outputs, labels)  # 计算损失
#             val_loss += loss.item()  # 累加损失
#
#             preds = torch.argmax(outputs, dim=1)  # 获取预测类别
#
#             start = bidx * batch_size  # 计算批次起始索引
#             samples = val_dataset.samples[start:start + len(labels)]  # 获取当前批次对应的样本
#
#             for i in range(len(labels)):  # 遍历当前批次的每个样本
#                 img_path, true_label = samples[i]  # 获取图像路径和真实标签
#                 pred_label = preds[i].item()  # 获取预测标签
#
#                 val_results.append({  # 添加验证结果
#                     "Image_Path": img_path,  # 图像路径
#                     "True_Label": idx_to_class[true_label],  # 真实标签（类别名）
#                     "Pred_Label": idx_to_class[pred_label],  # 预测标签（类别名）
#                     "Correct": int(true_label == pred_label)  # 是否正确预测（0或1）
#                 })
#
#     val_losses.append(val_loss / len(val_loader))  # 计算平均验证损失
#
#     print(  # 打印训练信息
#         f"Epoch {epoch+1}: "
#         f"Train Loss={train_losses[-1]:.6f}, "  # 当前轮次的训练损失
#         f"Val Loss={val_losses[-1]:.6f}"  # 当前轮次的验证损失
#     )
#
# # ===============================
# # 保存模型 & Excel
# # ===============================
# os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#
# torch.save(model.state_dict(), f"{save_dir}/fire_size_fcnn.pt")  # 保存模型权重
#
# pd.DataFrame({  # 创建损失数据DataFrame
#     "Epoch": range(1, epochs + 1),  # 轮次数
#     "Train_Loss": train_losses,  # 训练损失
#     "Val_Loss": val_losses  # 验证损失
# }).to_excel(f"{save_dir}/train_val_loss.xlsx", index=False)  # 保存为Excel文件
#
# pd.DataFrame(val_results).to_excel(  # 创建验证结果DataFrame并保存为Excel
#     f"{save_dir}/val_prediction_results.xlsx",
#     index=False
# )
#
# # ===============================
# # Loss 曲线
# # ===============================
# plt.figure()  # 创建新图形
# plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")  # 绘制训练损失曲线
# plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")  # 绘制验证损失曲线
# plt.xlabel("Epoch")  # 设置x轴标签
# plt.ylabel("Loss")  # 设置y轴标签
# plt.title("Fire Size FCNN Training & Validation Loss")  # 设置图形标题
# plt.legend()  # 显示图例
# plt.grid(True)  # 显示网格
# plt.savefig(f"{save_dir}/train_val_loss_curve.png", dpi=300)  # 保存图形
# plt.close()  # 关闭图形
























































# #复杂的火势判断模型
# import os  # 导入操作系统模块，用于文件和目录操作
# import cv2  # 导入OpenCV库，用于图像处理（读取、写入、变换等）
# import glob  # 导入 glob，用于批量查找文件
# import numpy as np  # 导入NumPy库，用于数值计算
# import torch  # 导入PyTorch深度学习框架
# import torch.nn as nn  # 导入PyTorch神经网络模块
# import torch.optim as optim  # 导入PyTorch优化器模块
# import torch.nn.functional as F  # 导入 PyTorch 的函数式接口（activation、loss、卷积等），并命名为 F，方便直接调用函数
# import pandas as pd  # 导入Pandas库，用于数据处理
# import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
#
# from PIL import Image  # 导入PIL库，用于图像处理
# from tqdm import tqdm  # 导入tqdm库，用于显示进度条
# from torch.utils.data import Dataset  # 导入 PyTorch Dataset 基类，用于自定义数据集
# from ultralytics import YOLO  # 导入YOLO模型
# from torchvision import transforms  # 导入torchvision的图像变换模块
# from torch.utils.data import DataLoader   # 用于按批次(batch)加载数据
# from smallest_unet_model import smallest_UNet  # 导入自定义的UNet模型
#
# # ===============================
# # 超参数
# # ===============================
# epochs = 50  # 训练的总轮数
# learning_rate = 1e-3  # 学习率
# img_size = 640  # 图像尺寸
# batch_size = 16  # 批量大小
# dropout_p = 0.3  #随机失活
# threshold_unet = 0.5007  # UNet分割的阈值
#
# save_dir = "pt_yangnet/yangnet_pt"  # 保存yangnet模型的路径
# pt_unet = "pt_yangnet/unet_pt/UNet_student_47.pt"  # UNet模型路径
# pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"  # YOLO场景尺度模型路径
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU可用性并设置设备
# print("Using device:", device)  # 打印使用的设备
#
# # ===============================
# # UNet（离线分割）
# # ===============================
# fire_seg_model = smallest_UNet().to(device)  # 实例化UNet模型并移动到设备
# fire_seg_model.load_state_dict(torch.load(pt_unet, map_location=device))  # 加载预训练权重
# fire_seg_model.eval()  # 设置为评估模式
#
# unet_transform = transforms.Compose([  # 定义图像预处理转换
#     transforms.Resize((img_size, img_size)),  # 调整图像大小
#     transforms.ToTensor()  # 转换为张量
# ])
#
# # ===============================
# # YOLO 场景尺度模型
# # ===============================
# scene_model = YOLO(pt_scene_size)  # 加载YOLO模型
#
# # ===============================
# # 场景尺度编码
# # ===============================
# def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 定义场景尺度分类函数
#     has_sky, has_trunk = False, False  # 初始化天空和树干标志
#     for cid in cls_ids:  # 遍历检测到的类别ID
#         if int(cid) == sky_id:  # 如果检测到天空
#             has_sky = True  # 设置天空标志为True
#         elif int(cid) == trunk_id:  # 如果检测到树干
#             has_trunk = True  # 设置树干标志为True
#
#     if has_trunk:  # 如果检测到树干（不管是否检测到天空都为近景）
#         return torch.tensor([1, 0, 0], dtype=torch.float32)  # 返回[1,0,0]表示近景
#     if has_sky:  # 如果检测到天空
#         return torch.tensor([0, 0, 1], dtype=torch.float32)  # 返回[0,0,1]表示远景
#     return torch.tensor([0, 1, 0], dtype=torch.float32)  # 否则返回[0,1,0]表示中景
#
# # ===============================
# # 数据路径
# # ===============================
# dataset_roots = {  # 定义数据集路径字典
#     "train": "yangnet/forest_fire_size_classification_dataset_detail/train",  # 训练集路径
#     "val": "yangnet/forest_fire_size_classification_dataset_detail/val"  # 验证集路径
# }
#
# # classes = ["no_fire", "small_fire", "mid_fire", "big_fire"]  # 类别列表
# # idx_to_class = {0: "no_fire", 1: "small_fire", 2: "mid_fire", 3: "big_fire"}  # 索引到类别的映射
#
# classes = ["smallscene_nofire", "smallscene_smallfire", "smallscene_midfire", "smallscene_bigfire", "midscene_nofire",
#            "midscene_smallfire", "midscene_midfire", "midscene_bigfire", "bigscene_nofire", "bigscene_midfire", "bigscene_bigfire"]  # 类别列表
# idx_to_class = {0: "smallscene_nofire", 1: "smallscene_smallfire", 2: "smallscene_midfire", 3: "smallscene_bigfire",
#                 4: "midscene_nofire", 5: "midscene_smallfire", 6: "midscene_midfire", 7: "midscene_bigfire",
#                 8: "bigscene_nofire", 9: "bigscene_midfire", 10: "bigscene_bigfire"}  # 索引到类别的映射
#
# # # ===============================
# # # 预计算火焰比例 & 掩膜
# # # ===============================
# # def precompute_fire_ratio_and_mask():  # 定义预计算火焰比例和掩膜的函数
# #     print("\n🔥 Precomputing fire masks & ratios")  # 打印开始预计算信息
# #
# #     for split, root_dir in dataset_roots.items():  # 遍历训练集和验证集
# #         for cls in classes:  # 遍历每个类别
# #             img_dir = os.path.join(root_dir, cls)  # 构建图像目录路径
# #             ratio_dir = os.path.join(img_dir, "fire_ratio")  # 构建火焰比例保存目录路径
# #             os.makedirs(ratio_dir, exist_ok=True)  # 创建目录（如果不存在）
# #
# #             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls}"):  # 遍历目录中的图像文件
# #                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
# #                     continue  # 跳过非图像文件
# #
# #                 img_path = os.path.join(img_dir, img_name)  # 构建完整图像路径
# #                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")  # 构建火焰比例保存路径
# #                 mask_path = os.path.join(  # 构建掩膜保存路径
# #                     img_dir,
# #                     os.path.splitext(img_name)[0] + "_mask.png"
# #                 )
# #
# #                 if os.path.exists(ratio_path) and os.path.exists(mask_path):  # 如果已经计算过
# #                     continue  # 跳过已处理的图像
# #
# #                 image = Image.open(img_path).convert("RGB")  # 打开并转换图像为RGB模式
# #                 image_tensor = unet_transform(image).unsqueeze(0).to(device)  # 预处理图像并添加批次维度
# #
# #                 with torch.no_grad():  # 禁用梯度计算
# #                     pred = fire_seg_model(image_tensor)[0, 0]  # 使用UNet进行预测
# #                     binary_mask = (pred > threshold_unet).float()  # 应用阈值生成二值掩膜
# #
# #                 fire_ratio = (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰比例
# #                 np.save(ratio_path, fire_ratio)  # 保存火焰比例到文件
# #
# #                 mask_img = (binary_mask.cpu().numpy() * 255).astype(np.uint8)  # 将掩膜转换为图像格式
# #                 Image.fromarray(mask_img).save(mask_path)  # 保存掩膜图像
# #
# #     print("✅ Fire masks & ratios cached\n")  # 打印预计算完成信息
#
# # # ===============================
# # # 预计算火焰比例 & 掩膜
# # # ===============================
# # def precompute_fire_ratio_and_mask():
# #     print("\n🔥 Precomputing fire masks & ratios")          # 打印开始提示信息
# #
# #     for split, root_dir in dataset_roots.items():           # 遍历数据集划分（train/val）及其根目录
# #         for cls in classes:                                 # 遍历文件夹里的所有类别（如 fire/nonfire）
# #             img_dir = os.path.join(root_dir, cls)           # 构建当前类别的图像目录路径
# #             ratio_dir = os.path.join(img_dir, "fire_ratio") # 构建火焰比例文件的保存目录
# #             os.makedirs(ratio_dir, exist_ok=True)           # 创建目录，如果已存在则忽略
# #
# #             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls}"):   # 遍历该目录下所有文件，并显示进度条
# #                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):    # 过滤非图像文件
# #                     continue                                                    # 跳过非图像文件
# #
# #                 img_path = os.path.join(img_dir, img_name)                      # 构建原始图像的完整路径
# #                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")         # 构建火焰比例文件保存路径（.npy）
# #
# #                 # 0/255 掩膜路径（可视化用）
# #                 mask_path = os.path.join(
# #                     img_dir,
# #                     os.path.splitext(img_name)[0] + "_mask.png"
# #                 )
# #
# #                 # 0/1 掩膜路径（计算用）
# #                 mask01_path = os.path.join(
# #                     img_dir,
# #                     os.path.splitext(img_name)[0] + "_mask01.png"
# #                 )
# #
# #                 # 如果三种产物均已存在，则跳过该图像，避免重复计算
# #                 if os.path.exists(ratio_path) and os.path.exists(mask_path) and os.path.exists(mask01_path):
# #                     continue
# #
# #                 image = Image.open(img_path).convert("RGB")                    # 打开图像并转换为RGB模式
# #                 image_tensor = unet_transform(image).unsqueeze(0).to(device)   # 应用预处理、增加batch维度、移至GPU
# #
# #                 with torch.no_grad():                                          # 禁用梯度计算以加速推理
# #                     pred = fire_seg_model(image_tensor)[0, 0]                  # 模型推理，取第一个batch的第一个通道
# #                     binary_mask = (pred > threshold_unet).float()              # 根据阈值生成二值掩膜（0或1）
# #
# #                 fire_ratio = (binary_mask.sum() / binary_mask.numel()).item()  # 计算火焰像素比例
# #                 np.save(ratio_path, fire_ratio)                                # 保存火焰比例为.npy文件
# #
# #                 # =========================
# #                 # 保存0/255掩膜（可视化用）
# #                 # =========================
# #                 mask_img_255 = (binary_mask.cpu().numpy() * 255).astype(np.uint8)  # 将0/1掩膜转换为0/255的uint8数组
# #                 Image.fromarray(mask_img_255).save(mask_path)                      # 保存为PNG图像（可视）
# #
# #                 # =========================
# #                 # 保存0/1掩膜（计算用）
# #                 # =========================
# #                 mask_img_01 = binary_mask.cpu().numpy().astype(np.uint8)            # 直接转为0/1的uint8数组
# #                 Image.fromarray(mask_img_01).save(mask01_path)                      # 保存为PNG图像（数值）
# #
# #     print("✅ Fire masks & ratios cached\n")                 # 打印完成信息
#
# # # ===============================
# # # FCNN
# # # ===============================
# # class FireSizeFCNN(nn.Module):  # 定义全连接神经网络类
# #     def __init__(self):  # 初始化函数
# #         super().__init__()  # 调用父类初始化
# #         self.net = nn.Sequential(  # 定义网络结构
# #             nn.Linear(4, 64),  # 输入层到隐藏层（4维输入，64维输出）
# #             nn.ReLU(),  # ReLU激活函数
# #             nn.Linear(64, 4)  # 隐藏层到输出层（64维输入，4维输出）
# #         )
# #
# #     def forward(self, x):  # 前向传播函数
# #         return self.net(x)  # 返回网络输出
# #
# # model = FireSizeFCNN().to(device)  # 实例化模型并移动到设备
#
# # ===============================
# # 预计算火焰比例 & 掩膜
# # ===============================
# def get_trunk_top_y(image, yolo_model):     # 获取所有trunk中最小的y1（只看class=1）
#     """
#     输入：PIL图像
#     输出：树干最上边界y1（忽略sky=0，只使用trunk=1）
#     """
#     results = yolo_model(image)             # YOLO模型推理
#
#     if results is None or len(results) == 0:    # 如果没有检测结果
#         return None                         # 返回None
#
#     if results[0].boxes is None:            # 如果没有检测框
#         return None                         # 返回None
#
#     boxes = results[0].boxes                # 获取检测框对象
#
#     xyxy = boxes.xyxy.cpu().numpy()         # 获取坐标 (N,4)
#     cls = boxes.cls.cpu().numpy()           # 获取类别 (N,)
#     conf = boxes.conf.cpu().numpy()         # 获取置信度 (N,)
#
#     trunk_y1_list = []                      # 用于存储所有trunk的y1
#
#     for i in range(len(cls)):               # 遍历所有检测框
#
#         if int(cls[i]) != 1:                # 只保留 trunk（class=1）
#             continue                       # 跳过 sky
#
#         if conf[i] < 0.5:                  # 过滤低置信度检测（可调）
#             continue
#
#         x1, y1, x2, y2 = xyxy[i]           # 获取坐标
#         trunk_y1_list.append(y1)           # 收集y1
#
#     if len(trunk_y1_list) == 0:            # 如果没有检测到trunk
#         return None                        # 返回None
#
#     return int(min(trunk_y1_list))       # 传统方法：取最小
#
# # dataset_root/
# # │
# # ├── train/
# # │   ├── fire/
# # │   │   ├── image1.jpg
# # │   │   ├── image1_mask.png
# # │   │   ├── image1_mask01.png
# # │   │   │
# # │   │   ├── fire_ratio/
# # │   │   │   ├── image1.jpg.npy
# # │   │   │
# # │   │   ├── fire_trunk_mask/
# # │   │   │   ├── image1_mask_trunk.png
# # │   │   │   ├── image1_mask_trunk_vis.png
# # │   │   │
# # │   │   ├── fire_edge/
# # │   │   │   ├── image1_sobelx.png
# # │   │   │   ├── image1_sobely.png
# # │   │   │   ├── image1_sobelmag.png
# # │   │   │   ├── image1_sobelx_vis.png
# # │   │   │   ├── image1_sobely_vis.png
# # │   │   │   ├── image1_sobelmag_vis.png
# # │   │
# # │   ├── nonfire/
# # │       └── （结构完全一样）
# # │
# # ├── val/
# # │   ├── fire/
# # │   └── nonfire/
# # def precompute_fire_ratio_and_mask():           # 主函数：预计算所有数据
# #
# #     print("\n🔥 Precomputing fire masks, ratios, Sobel edges & trunk-cut masks")  # 提示开始
# #
# #     for split, root_dir in dataset_roots.items():   # 遍历数据集（train / val）
# #
# #         for cls_name in classes:                    # 遍历类别（fire / nonfire）
# #
# #             img_dir = os.path.join(root_dir, cls_name)   # 当前类别路径
# #
# #             # ========= 输出目录 =========
# #             ratio_dir = os.path.join(img_dir, "fire_ratio")        # 火焰比例保存目录
# #             edge_dir = os.path.join(img_dir, "fire_edge")          # Sobel边缘保存目录
# #             trunk_mask_dir = os.path.join(img_dir, "fire_trunk_mask")  # 树干裁剪mask目录
# #
# #             os.makedirs(ratio_dir, exist_ok=True)   # 创建目录（存在不报错）
# #             os.makedirs(edge_dir, exist_ok=True)
# #             os.makedirs(trunk_mask_dir, exist_ok=True)
# #
# #             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls_name}"):  # 遍历图片
# #
# #                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 过滤非图像文件
# #                     continue                        # 跳过
# #
# #                 name = os.path.splitext(img_name)[0]   # 去掉扩展名（得到纯文件名）
# #                 img_path = os.path.join(img_dir, img_name)  # 图片完整路径
# #
# #                 # ========= 输出路径 =========
# #                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")  # 火焰比例文件路径
# #
# #                 mask_path = os.path.join(img_dir, name + "_mask.png")        # 原始mask（0/255）
# #                 mask01_path = os.path.join(img_dir, name + "_mask01.png")    # 原始mask（0/1）
# #
# #                 trunk_mask_path = os.path.join(trunk_mask_dir, name + "_mask_trunk.png")       # 裁剪后mask（0/1）
# #                 trunk_mask_vis_path = os.path.join(trunk_mask_dir, name + "_mask_trunk_vis.png")  # 裁剪后mask（0/255）
# #
# #                 sobelx_path = os.path.join(edge_dir, name + "_sobelx.png")     # Sobel X
# #                 sobely_path = os.path.join(edge_dir, name + "_sobely.png")     # Sobel Y
# #                 sobelmag_path = os.path.join(edge_dir, name + "_sobelmag.png") # Sobel幅值
# #
# #                 # ========= 跳过已处理 =========
# #                 if (os.path.exists(ratio_path) and               # 火焰比例已存在
# #                     os.path.exists(mask_path) and               # 原mask已存在
# #                     os.path.exists(mask01_path) and             # 0/1 mask存在
# #                     os.path.exists(sobelmag_path) and           # Sobel存在
# #                     os.path.exists(trunk_mask_path)):           # 裁剪mask存在
# #                     continue                                   # 跳过当前图片
# #
# #                 # =========================
# #                 # 1️⃣ UNet分割
# #                 # =========================
# #                 image = Image.open(img_path).convert("RGB")     # 读取RGB图像
# #
# #                 image_tensor = unet_transform(image).unsqueeze(0).to(device)  # 预处理并加batch维度
# #
# #                 with torch.no_grad():                          # 推理模式（不计算梯度）
# #                     pred = fire_seg_model(image_tensor)[0, 0]  # 得到预测mask
# #                     binary_mask = (pred > threshold_unet).float()  # 二值化（0/1）
# #
# #                 # =========================
# #                 # 2️⃣ 火焰比例
# #                 # =========================
# #                 fire_ratio = (binary_mask.sum() / binary_mask.numel()).item()  # 火焰像素占比
# #                 np.save(ratio_path, fire_ratio)               # 保存为.npy文件
# #
# #                 # =========================
# #                 # 3️⃣ 保存原始mask
# #                 # =========================
# #                 mask_img_255 = (binary_mask.cpu().numpy() * 255).astype(np.uint8)  # 转为0/255
# #                 Image.fromarray(mask_img_255).save(mask_path)  # 保存
# #
# #                 mask_img_01 = binary_mask.cpu().numpy().astype(np.uint8)  # 转为0/1
# #                 Image.fromarray(mask_img_01).save(mask01_path)  # 保存
# #
# #                 # =========================
# #                 # 4️⃣ YOLO树干检测 + 区域过滤
# #                 # =========================
# #                 y1 = get_trunk_top_y(image, scene_model)   # 获取树干最上边界（像素坐标）
# #
# #                 cropped_mask01 = mask_img_01.copy()        # 拷贝0/1 mask
# #                 cropped_mask255 = mask_img_255.copy()      # 拷贝0/255 mask
# #
# #                 if y1 is not None:                         # 如果检测到树干
# #
# #                     h, w = cropped_mask01.shape           # mask尺寸（640×640）
# #                     img_h = image.height                  # 原图高度
# #
# #                     scale = h / img_h                     # 计算缩放比例
# #                     y1 = int(y1 * scale)                  # 映射到mask坐标
# #
# #                     y1 = max(0, min(h, y1))               # 限制范围（防止越界）
# #
# #                     cropped_mask01[y1:, :] = 0            # 将树干以下全部置0（0/1）
# #                     cropped_mask255[y1:, :] = 0           # 同步处理0/255（用于可视化）
# #
# #                 # =========================
# #                 # 5️⃣ 保存裁剪mask
# #                 # =========================
# #                 Image.fromarray(cropped_mask01.astype(np.uint8)).save(trunk_mask_path)  # 保存0/1
# #
# #                 cv2.imwrite(trunk_mask_vis_path, cropped_mask255)  # 保存0/255（更直观）
# #
# #                 # =========================
# #                 # 6️⃣ Sobel边缘（基于裁剪后mask）
# #                 # =========================
# #                 img_mask = cropped_mask255.copy().astype(np.float32)  # 转float，避免Sobel计算截断（输入原本是0/255）
# #
# #                 sobelx = cv2.Sobel(img_mask, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度（会有正负值）
# #                 sobely = cv2.Sobel(img_mask, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
# #
# #                 sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)  # 梯度幅值（连续值）
# #
# #                 # =========================
# #                 # ⭐ 转成严格0/1边缘
# #                 # =========================
# #                 sobelx_bin = (np.abs(sobelx) > 0).astype(np.uint8)  # 有梯度→1，否则0
# #                 sobely_bin = (np.abs(sobely) > 0).astype(np.uint8)  # 同理
# #                 sobelmag_bin = (sobel_mag > 0).astype(np.uint8)  # 幅值>0即边缘
# #
# #                 # =========================
# #                 # 保存0/1版本（用于训练/计算）
# #                 # =========================
# #                 cv2.imwrite(sobelx_path, sobelx_bin)  # 保存0/1（注意：看起来几乎全黑）
# #                 cv2.imwrite(sobely_path, sobely_bin)  # 保存0/1
# #                 cv2.imwrite(sobelmag_path, sobelmag_bin)  # 保存0/1
# #
# #                 # =========================
# #                 # （可选）保存可视化版本（推荐）
# #                 # =========================
# #                 cv2.imwrite(sobelx_path.replace(".png", "_vis.png"), sobelx_bin * 255)  # 转成0/255方便观察
# #                 cv2.imwrite(sobely_path.replace(".png", "_vis.png"), sobely_bin * 255)
# #                 cv2.imwrite(sobelmag_path.replace(".png", "_vis.png"), sobelmag_bin * 255)
# #
# #     print("✅ Fire masks, ratios, Sobel edges & trunk-cut masks cached\n")  # 完成提示
#
# def precompute_fire_ratio_and_mask_fixed():
#     print("\n🔥 Precomputing fire masks, ratios, Sobel edges & trunk-cut masks (fixed version)")
#
#     for split, root_dir in dataset_roots.items():   # train/val
#         for cls_name in classes:                    # 遍历每个类别
#             img_dir = os.path.join(root_dir, cls_name)
#             ratio_dir = os.path.join(img_dir, "fire_ratio")
#             edge_dir = os.path.join(img_dir, "fire_edge")
#             trunk_mask_dir = os.path.join(img_dir, "fire_trunk_mask")
#
#             os.makedirs(ratio_dir, exist_ok=True)
#             os.makedirs(edge_dir, exist_ok=True)
#             os.makedirs(trunk_mask_dir, exist_ok=True)
#
#             for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls_name}"):
#                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
#                     continue
#
#                 name = os.path.splitext(img_name)[0]
#                 img_path = os.path.join(img_dir, img_name)
#
#                 # 输出路径
#                 ratio_path = os.path.join(ratio_dir, img_name + ".npy")
#                 mask_path = os.path.join(img_dir, name + "_mask.png")          # 可视化 0/255
#                 mask01_path = os.path.join(img_dir, name + "_mask01.png")      # 训练用 0/1
#                 trunk_mask_path = os.path.join(trunk_mask_dir, name + "_mask_trunk.png")
#                 trunk_mask_vis_path = os.path.join(trunk_mask_dir, name + "_mask_trunk_vis.png")
#                 sobelx_path = os.path.join(edge_dir, name + "_sobelx.png")
#                 sobely_path = os.path.join(edge_dir, name + "_sobely.png")
#                 sobelmag_path = os.path.join(edge_dir, name + "_sobelmag.png")
#
#                 # 跳过已处理
#                 if all(os.path.exists(p) for p in [ratio_path, mask_path, mask01_path, trunk_mask_path, sobelmag_path]):
#                     continue
#
#                 # =========================
#                 # 1️⃣ UNet 分割
#                 # =========================
#                 image = Image.open(img_path).convert("RGB")
#                 image_tensor = unet_transform(image).unsqueeze(0).to(device)
#
#                 with torch.no_grad():
#                     pred = fire_seg_model(image_tensor)[0, 0]  # 预测 mask
#                     pred = torch.sigmoid(pred)                 # 确保 0~1
#
#                 # =========================
#                 # 2️⃣ 二值化掩膜
#                 # =========================
#                 threshold = 0.5007
#                 mask01 = (pred > threshold).float().cpu().numpy().astype(np.uint8)  # 0/1 mask
#                 mask255 = (mask01 * 255).astype(np.uint8)                             # 可视化 mask
#
#                 # =========================
#                 # 3️⃣ 火焰比例
#                 # =========================
#                 fire_ratio = mask01.sum() / mask01.size
#                 np.save(ratio_path, fire_ratio)
#
#                 # =========================
#                 # 4️⃣ 保存掩膜
#                 # =========================
#                 Image.fromarray(mask01).save(mask01_path)
#                 Image.fromarray(mask255).save(mask_path)
#
#                 # =========================
#                 # 5️⃣ YOLO 树干裁剪
#                 # =========================
#                 y1 = get_trunk_top_y(image, scene_model)
#                 cropped_mask01 = mask01.copy()
#                 cropped_mask255 = mask255.copy()
#
#                 if y1 is not None:
#                     h, w = cropped_mask01.shape
#                     scale = h / image.height
#                     y1 = int(y1 * scale)
#                     y1 = max(0, min(h, y1))
#                     cropped_mask01[y1:, :] = 0
#                     cropped_mask255[y1:, :] = 0
#
#                 # 保存裁剪 mask
#                 Image.fromarray(cropped_mask01).save(trunk_mask_path)
#                 Image.fromarray(cropped_mask255).save(trunk_mask_vis_path)
#
#                 # =========================
#                 # 6️⃣ Sobel 边缘（基于裁剪后 mask）
#                 # =========================
#                 img_mask_float = cropped_mask255.astype(np.float32)
#                 sobelx = cv2.Sobel(img_mask_float, cv2.CV_64F, 1, 0, ksize=3)
#                 sobely = cv2.Sobel(img_mask_float, cv2.CV_64F, 0, 1, ksize=3)
#                 sobel_mag = np.sqrt(sobelx**2 + sobely**2)
#
#                 # 转 0/1
#                 sobelx_bin = (np.abs(sobelx) > 0).astype(np.uint8)
#                 sobely_bin = (np.abs(sobely) > 0).astype(np.uint8)
#                 sobelmag_bin = (sobel_mag > 0).astype(np.uint8)
#
#                 # 保存 0/1 用于训练
#                 cv2.imwrite(sobelx_path, sobelx_bin)
#                 cv2.imwrite(sobely_path, sobely_bin)
#                 cv2.imwrite(sobelmag_path, sobelmag_bin)
#
#                 # 保存可视化 0/255
#                 cv2.imwrite(sobelx_path.replace(".png","_vis.png"), sobelx_bin*255)
#                 cv2.imwrite(sobely_path.replace(".png","_vis.png"), sobely_bin*255)
#                 cv2.imwrite(sobelmag_path.replace(".png","_vis.png"), sobelmag_bin*255)
#
#     print("✅ Fire masks, ratios, Sobel edges & trunk-cut masks cached (fixed)\n")
#
#
# # ===============================
# # 场景三维向量MLP
# # ===============================
# class SceneFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(SceneFeatureExtractor, self).__init__()
#         # MLP结构：3 → 16 → 32 → 64 → 128
#         self.mlp = nn.Sequential(
#             nn.Linear(3, 16),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),  # Dropout 防过拟合
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_p)
#         )
#
#     def forward(self, x):
#         """
#         x: [batch_size, 3]
#         """
#         feat_128 = self.mlp(x)  # [B, 128]
#         A1, B1 = torch.split(feat_128, 64, dim=1)  # 🔥 从中间拆分为两个64维
#         return A1, B1
#
# # ===============================
# # 火焰图像比例一维向量MLP
# # ===============================
# class FireRatioFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FireRatioFeatureExtractor, self).__init__()
#         # MLP结构：1 → 16 → 32 → 64 → 128
#         self.mlp = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_p)
#         )
#
#     def forward(self, x):
#         """
#         x: [batch_size, 1]  （火焰比例）
#         """
#         feat_128 = self.mlp(x)  # [B, 128]
#         A2, B2 = torch.split(feat_128, 64, dim=1)
#         return A2, B2
#
# # ===============================
# # 火焰图像特征CNN网络
# # ===============================
# class FireVisualFeatureCNN_20x20_FC64_Optim(nn.Module):
#     """
#     输入: (batch, 5, 640, 640)
#     输出: (batch, 64)
#     卷积下采样到 20x20x128，再接优化后的递减型全连接输出64维向量
#     """
#     def __init__(self, out_dim=64):
#         super(FireVisualFeatureCNN_20x20_FC64_Optim, self).__init__()
#
#         # 卷积下采样到 20x20
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),    # 640->320
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 320->160
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 160->80
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 80->40
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 40->20
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#
#         # 递减型全连接
#         self.fc = nn.Sequential(
#             nn.Linear(128*20*20, 256),  # 51200 -> 256
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#
#             nn.Linear(256, 128),        # 256 -> 128
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#
#             nn.Linear(128, out_dim)     # 128 -> 64
#         )
#
#     def forward(self, x):
#         x = self.conv_block(x)       # -> (batch, 128, 20, 20)
#         x = x.view(x.size(0), -1)    # -> (batch, 128*20*20)
#         visual_feat = self.fc(x)     # -> (batch, 64)
#         return visual_feat
#
# # ===============================
# # 特征缩放网络及最后的分类网络
# # ===============================
# class F1F2FusionClassifier(nn.Module):
#     def __init__(self, visual_feat_dim=64, num_classes=11):
#         super(F1F2FusionClassifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(visual_feat_dim * 2, 64),  # 128 -> 64
#             nn.ReLU(),
#             nn.Linear(64, 32),  # 64 -> 32
#             nn.ReLU(),
#             nn.Linear(32, num_classes)  # 32 -> 11
#         )
#
#     def forward(self, visual_feat, A1, B1, A2, B2):
#         """
#         visual_feat: [B, 64] CNN输出的特征
#         A1, B1, A2, B2: [B, 64] 可训练缩放和偏移量
#         """
#         # 逐元素变换
#         F1 = A1 * visual_feat + B1  # [B,64]
#         F2 = A2 * visual_feat + B2  # [B,64]
#
#         # 拼接成128维
#         fused_feat = torch.cat([F1, F2], dim=1)  # [B,128]
#
#         # 分类
#         logits = self.classifier(fused_feat)  # [B,11]
#         probs = F.softmax(logits, dim=1)  # [B,11]
#
#         return probs, fused_feat, F1, F2
#
# # ===============================
# # 执行预计算
# # ===============================
# precompute_fire_ratio_and_mask_fixed()  # 调用预计算函数
#
# # channel 0: sobelx_bin
# # channel 1: sobely_bin
# # channel 2: sobelmag_bin
# # channel 3: 原始 0/1 mask_img_01
# # channel 4: 裁剪后的树干 mask_trunk
# class FireVisualDataset(Dataset):  # 定义自定义数据集类，继承 Dataset
#     """
#     火焰数据集读取类
#     每个样本包含5个通道：
#         0: sobelx_bin
#         1: sobely_bin
#         2: sobelmag_bin
#         3: mask_img_01
#         4: mask_trunk
#     标签对应类别：
#         smallscene_nofire -> 0
#         smallscene_smallfire -> 1
#         smallscene_midfire -> 2
#         smallscene_bigfire -> 3
#         midscene_nofire -> 4
#         midscene_smallfire -> 5
#         midscene_midfire -> 6
#         midscene_bigfire -> 7
#         bigscene_nofire -> 8
#         bigscene_midfire -> 9
#         bigscene_bigfire -> 10
#     """
#     def __init__(self, root_dir, img_size=640, transform=None):  # 初始化函数
#         super().__init__()                         # 调用父类 Dataset 的初始化
#         self.root_dir = root_dir                   # 数据集根目录
#         self.img_size = img_size                   # 图片统一大小
#         self.transform = transform                 # 可选 transform，用于数据增强
#
#         # 定义类别到标签的映射字典
#         self.label_map = {
#             "smallscene_nofire": 0,               # 小场景无火
#             "smallscene_smallfire": 1,            # 小场景小火
#             "smallscene_midfire": 2,              # 小场景中火
#             "smallscene_bigfire": 3,              # 小场景大火
#             "midscene_nofire": 4,                 # 中场景无火
#             "midscene_smallfire": 5,              # 中场景小火
#             "midscene_midfire": 6,                # 中场景中火
#             "midscene_bigfire": 7,                # 中场景大火
#             "bigscene_nofire": 8,                 # 大场景无火
#             "bigscene_midfire": 9,                # 大场景中火
#             "bigscene_bigfire": 10                # 大场景大火
#         }
#
#         self.data_list = []                        # 用于存储所有样本信息的列表
#         for cls_name, cls_idx in self.label_map.items():  # 遍历类别及其标签
#             cls_dir = os.path.join(root_dir, cls_name)   # 当前类别文件夹路径
#             if not os.path.exists(cls_dir):              # 如果文件夹不存在
#                 continue                                 # 跳过该类别
#
#             # 查找所有 mask01 文件（标记火焰区域的二值图）
#             mask01_paths = glob.glob(os.path.join(cls_dir, "*mask01.png"))
#             for mask01_path in mask01_paths:             # 遍历每个 mask01 文件
#                 name = os.path.splitext(os.path.basename(mask01_path))[0].replace("_mask01", "")  # 去掉后缀和 _mask01
#
#                 sample = {                               # 构建每个样本的路径字典
#                     "sobelx": os.path.join(cls_dir, "fire_edge", f"{name}_sobelx.png"),           # Sobel X 通道
#                     "sobely": os.path.join(cls_dir, "fire_edge", f"{name}_sobely.png"),           # Sobel Y 通道
#                     "sobelmag": os.path.join(cls_dir, "fire_edge", f"{name}_sobelmag.png"),       # Sobel magnitude 通道
#                     "mask01": mask01_path,                                                         # 原始火焰 mask
#                     "mask_trunk": os.path.join(cls_dir, "fire_trunk_mask", f"{name}_mask_trunk.png"),  # 树干 mask
#
#                     # ===============================
#                     # 🔥 新增：scene_vec 和 fire_ratio 路径
#                     # ===============================
#                     "scene_vec": os.path.join(cls_dir, "scene_vec", f"{name}.npy"),              # 场景三维向量
#                     "fire_ratio": os.path.join(cls_dir, "fire_ratio", f"{name}.jpg.npy"),        # 火焰比例
#
#                     "label": cls_idx                                                              # 对应类别标签
#                 }
#
#                 self.data_list.append(sample)            # 添加到样本列表中
#
#     def __len__(self):                                # 返回数据集大小
#         return len(self.data_list)                    # 样本数量
#
#     def __getitem__(self, idx):                       # 获取第 idx 个样本
#         item = self.data_list[idx]                    # 获取样本字典
#
#         # ===============================
#         # 1️⃣ 读取 5 通道视觉输入
#         # ===============================
#         channels = []                                 # 用于存储 5 个通道图像
#         for key in ["sobelx", "sobely", "sobelmag", "mask01", "mask_trunk"]:  # 遍历每个通道
#             img = Image.open(item[key]).convert("L").resize((self.img_size, self.img_size))  # 读取为灰度图并 resize
#             img_arr = np.array(img, dtype=np.float32)  # 转为 numpy float32 数组
#             channels.append(img_arr)                   # 添加到通道列表
#
#         visual_input = np.stack(channels, axis=0)      # 堆叠为 (5, H, W) 的数组
#         visual_input = torch.tensor(visual_input, dtype=torch.float32)  # 转为 PyTorch 张量
#
#         # ===============================
#         # 2️⃣ 读取 scene_vec（🔥新增）
#         # ===============================
#         scene_vec = np.load(item["scene_vec"])         # 读取 .npy（三维向量）
#         scene_vec = torch.tensor(scene_vec, dtype=torch.float32)  # 转为 tensor [3]
#
#         # ===============================
#         # 3️⃣ 读取 fire_ratio（🔥新增）
#         # ===============================
#         fire_ratio = np.load(item["fire_ratio"])       # 读取火焰比例
#         fire_ratio = torch.tensor([fire_ratio], dtype=torch.float32)  # 转为 [1]
#
#         # ===============================
#         # 4️⃣ 标签
#         # ===============================
#         label_tensor = torch.tensor(item["label"], dtype=torch.long)  # 标签转为 tensor
#
#         # ===============================
#         # 5️⃣ 可选 transform
#         # ===============================
#         if self.transform:                             # 如果有 transform
#             visual_input = self.transform(visual_input)  # 应用 transform
#
#         # ===============================
#         # 🔥 返回三输入 + 标签
#         # ===============================
#         return visual_input, scene_vec, fire_ratio, label_tensor
#
# # class FireVisualDataset(Dataset):  # 定义自定义数据集类，继承 Dataset
# #     """
# #     火焰数据集读取类
# #     每个样本包含5个通道：
# #         0: sobelx_bin
# #         1: sobely_bin
# #         2: sobelmag_bin
# #         3: mask_img_01
# #         4: mask_trunk
# #     标签对应类别：
# #         smallscene_nofire -> 0
# #         smallscene_smallfire -> 1
# #         smallscene_midfire -> 2
# #         smallscene_bigfire -> 3
# #         midscene_nofire -> 4
# #         midscene_smallfire -> 5
# #         midscene_midfire -> 6
# #         midscene_bigfire -> 7
# #         bigscene_nofire -> 8
# #         bigscene_midfire -> 9
# #         bigscene_bigfire -> 10
# #     """
# #     def __init__(self, root_dir, img_size=640, transform=None):  # 初始化函数
# #         super().__init__()                         # 调用父类 Dataset 的初始化
# #         self.root_dir = root_dir                   # 数据集根目录
# #         self.img_size = img_size                   # 图片统一大小
# #         self.transform = transform                 # 可选 transform，用于数据增强
# #
# #         # 定义类别到标签的映射字典
# #         self.label_map = {
# #             "smallscene_nofire": 0,               # 小场景无火
# #             "smallscene_smallfire": 1,            # 小场景小火
# #             "smallscene_midfire": 2,              # 小场景中火
# #             "smallscene_bigfire": 3,              # 小场景大火
# #             "midscene_nofire": 4,                 # 中场景无火
# #             "midscene_smallfire": 5,              # 中场景小火
# #             "midscene_midfire": 6,                # 中场景中火
# #             "midscene_bigfire": 7,                # 中场景大火
# #             "bigscene_nofire": 8,                 # 大场景无火
# #             "bigscene_midfire": 9,                # 大场景中火
# #             "bigscene_bigfire": 10                # 大场景大火
# #         }
# #
# #         self.data_list = []                        # 用于存储所有样本信息的列表
# #         for cls_name, cls_idx in self.label_map.items():  # 遍历类别及其标签
# #             cls_dir = os.path.join(root_dir, cls_name)   # 当前类别文件夹路径
# #             if not os.path.exists(cls_dir):              # 如果文件夹不存在
# #                 continue                                 # 跳过该类别
# #
# #             # 查找所有 mask01 文件（标记火焰区域的二值图）
# #             mask01_paths = glob.glob(os.path.join(cls_dir, "*mask01.png"))
# #             for mask01_path in mask01_paths:             # 遍历每个 mask01 文件
# #                 name = os.path.splitext(os.path.basename(mask01_path))[0].replace("_mask01", "")  # 去掉后缀和 _mask01
# #                 sample = {                               # 构建每个样本的路径字典
# #                     "sobelx": os.path.join(cls_dir, "fire_edge", f"{name}_sobelx.png"),           # Sobel X 通道
# #                     "sobely": os.path.join(cls_dir, "fire_edge", f"{name}_sobely.png"),           # Sobel Y 通道
# #                     "sobelmag": os.path.join(cls_dir, "fire_edge", f"{name}_sobelmag.png"),       # Sobel magnitude 通道
# #                     "mask01": mask01_path,                                                         # 原始火焰 mask
# #                     "mask_trunk": os.path.join(cls_dir, "fire_trunk_mask", f"{name}_mask_trunk.png"),  # 树干 mask
# #                     "label": cls_idx                                                              # 对应类别标签
# #                 }
# #                 self.data_list.append(sample)            # 添加到样本列表中
# #
# #     def __len__(self):                                # 返回数据集大小
# #         return len(self.data_list)                    # 样本数量
# #
# #     def __getitem__(self, idx):                       # 获取第 idx 个样本
# #         item = self.data_list[idx]                    # 获取样本字典
# #         channels = []                                 # 用于存储 5 个通道图像
# #         for key in ["sobelx", "sobely", "sobelmag", "mask01", "mask_trunk"]:  # 遍历每个通道
# #             img = Image.open(item[key]).convert("L").resize((self.img_size, self.img_size))  # 读取为灰度图并 resize
# #             img_arr = np.array(img, dtype=np.float32)  # 转为 numpy float32 数组
# #             channels.append(img_arr)                   # 添加到通道列表
# #
# #         visual_input = np.stack(channels, axis=0)      # 堆叠为 (5, H, W) 的数组
# #         visual_input = torch.tensor(visual_input, dtype=torch.float32)  # 转为 PyTorch 张量
# #
# #         label_tensor = torch.tensor(item["label"], dtype=torch.long)  # 标签转为 tensor
# #
# #         if self.transform:                             # 如果有 transform
# #             visual_input = self.transform(visual_input)  # 应用 transform
# #
# #         return visual_input, label_tensor              # 返回 5 通道图像和标签
#
# class FireFullModel(nn.Module):
#     def __init__(self, num_classes=11):
#         super(FireFullModel, self).__init__()
#
#         # ===============================
#         # 1️⃣ 三个可训练子网络
#         # ===============================
#         self.scene_extractor = SceneFeatureExtractor()          # 3维 → A1,B1
#         self.ratio_extractor = FireRatioFeatureExtractor()      # 1维 → A2,B2
#         self.visual_cnn = FireVisualFeatureCNN_20x20_FC64_Optim()  # 图像 → visual_feat
#
#         # ===============================
#         # 2️⃣ 融合分类器
#         # ===============================
#         self.fusion_classifier = F1F2FusionClassifier(
#             visual_feat_dim=64,
#             num_classes=num_classes
#         )
#
#     def forward(self, visual_input, scene_vec, fire_ratio):
#         """
#         visual_input: [B, 5, 640, 640]
#         scene_vec:    [B, 3]
#         fire_ratio:   [B, 1]
#         """
#
#         # ===============================
#         # 1️⃣ CNN提取视觉特征
#         # ===============================
#         visual_feat = self.visual_cnn(visual_input)  # [B,64]
#
#         # ===============================
#         # 2️⃣ MLP提取缩放参数
#         # ===============================
#         A1, B1 = self.scene_extractor(scene_vec)     # [B,64]
#         A2, B2 = self.ratio_extractor(fire_ratio)    # [B,64]
#
#         # ===============================
#         # 3️⃣ 融合分类
#         # ===============================
#         logits, fused_feat, F1, F2 = self.fusion_classifier(
#             visual_feat, A1, B1, A2, B2
#         )
#
#         return logits, fused_feat, F1, F2
#
# # ===============================
# # 训练集 DataLoader
# # ===============================
# train_dataset = FireVisualDataset(
#     root_dir=dataset_roots["train"],  # 训练集根目录
#     img_size=img_size,                # 图片统一尺寸
#     transform=None                    # 可选 transform(输入的只有0和1，不需要再做归一化)
# )
# train_loader = DataLoader(
#     dataset=train_dataset,             # 训练数据集
#     batch_size=batch_size,             # 每个 batch 样本数
#     shuffle=True,                      # 打乱训练数据
#     num_workers=0                      # 多线程读取
# )
#
# # ===============================
# # 验证集 DataLoader
# # ===============================
# val_dataset = FireVisualDataset(
#     root_dir=dataset_roots["val"],     # 验证集根目录
#     img_size=img_size,                 # 图片统一尺寸
#     transform=None                     # 可选 transform
# )
# val_loader = DataLoader(
#     val_dataset,                       # 验证数据集
#     batch_size=batch_size,             # 每个 batch 样本数
#     shuffle=False,                     # 验证数据不打乱
#     num_workers=0                      # 多线程读取
# )
#
# # ===============================
# # 训练与验证
# # ===============================
# model_last = FireFullModel(num_classes=11).to(device)  # 总模型
# criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
# optimizer = optim.Adam(model_last.parameters(), lr=learning_rate)  # 定义优化器（Adam）
#
# # ===============================
# # 📊 初始化DataFrame
# # ===============================
# training_results = pd.DataFrame(columns=[  # 定义表头
#     "Epoch",
#     "Train Loss",
#     "Val Loss",
#     "Train Acc",
#     "Val Acc"
# ])
#
# excel_path = os.path.join(save_dir, "training_results.xlsx")  # Excel保存路径
#
# # ===============================
# # 训练记录
# # ===============================
# train_losses, val_losses = [], []
# train_accs, val_accs = [], []
#
# # ===============================
# # 开始训练
# # ===============================
# for epoch in range(epochs):  # 遍历每一轮训练（epoch）
#
#     # ===============================
#     # 🔥 Train
#     # ===============================
#     model_last.train()  # 设置模型为训练模式（启用Dropout和BatchNorm更新）
#     train_loss = 0.0  # 初始化训练损失累计值
#     train_correct = 0  # 初始化训练正确样本数
#     train_total = 0    # 初始化训练样本总数
#
#     with tqdm(train_loader, unit="batch", desc=f"Train {epoch}") as tepoch:  # 创建训练进度条
#
#         for step, (visual_input, scene_vec, fire_ratio, label) in enumerate(tepoch):  # 遍历每一个batch
#
#             visual_input = visual_input.to(device)  # 将图像输入移动到GPU或CPU
#             scene_vec = scene_vec.to(device)        # 将场景三维向量移动到设备
#             fire_ratio = fire_ratio.to(device)      # 将火焰比例数据移动到设备
#             label = label.to(device)                # 将标签移动到设备
#
#             logits, _, _, _ = model_last(visual_input, scene_vec, fire_ratio)  # 前向传播，得到logits输出
#
#             batch_loss = criterion(logits, label)  # 计算当前batch的损失（交叉熵）
#
#             optimizer.zero_grad()  # 清空上一轮计算的梯度
#             batch_loss.backward()  # 反向传播，计算梯度
#             optimizer.step()       # 更新模型参数
#
#             train_loss += batch_loss.item()  # 累加当前batch的loss（用于计算epoch平均）
#
#             preds = torch.argmax(logits, dim=1)  # 获取预测类别（取最大概率对应的索引）
#             train_correct += (preds == label).sum().item()  # 累加预测正确的样本数
#             train_total += label.size(0)  # 累加总样本数
#
#             epoch_loss = train_loss / (step + 1)  # 当前epoch的平均loss（到当前batch为止）
#             epoch_acc = train_correct / train_total  # 当前epoch的平均准确率
#
#             tepoch.set_postfix(  # 在进度条右侧显示实时信息
#                 batch_loss=f"{batch_loss.item():.4f}",  # 当前batch的loss
#                 epoch_loss=f"{epoch_loss:.4f}",         # 当前epoch平均loss
#                 acc=f"{epoch_acc:.4f}"                  # 当前准确率
#             )
#
#     train_losses.append(train_loss / len(train_loader))  # 保存当前epoch的平均训练损失
#     train_accs.append(train_correct / train_total)       # 保存当前epoch的训练准确率
#
#     # ===============================
#     # 🔵 Val
#     # ===============================
#     model_last.eval()  # 设置模型为评估模式（关闭Dropout，不更新BN）
#     val_loss = 0.0  # 初始化验证损失
#     val_correct = 0  # 初始化验证正确数
#     val_total = 0    # 初始化验证总数
#
#     with torch.no_grad():  # 关闭梯度计算（节省显存，提高推理速度）
#         with tqdm(val_loader, unit="batch", desc=f"Val {epoch}") as vepoch:  # 创建验证进度条
#
#             for step, (visual_input, scene_vec, fire_ratio, label) in enumerate(vepoch):  # 遍历验证集
#
#                 visual_input = visual_input.to(device)  # 图像数据移动到设备
#                 scene_vec = scene_vec.to(device)        # 场景向量移动到设备
#                 fire_ratio = fire_ratio.to(device)      # 火焰比例移动到设备
#                 label = label.to(device)                # 标签移动到设备
#
#                 logits, _, _, _ = model_last(visual_input, scene_vec, fire_ratio)  # 前向传播
#
#                 batch_loss = criterion(logits, label)  # 计算当前batch的损失
#                 val_loss += batch_loss.item()          # 累加验证损失
#
#                 preds = torch.argmax(logits, dim=1)  # 获取预测类别
#                 val_correct += (preds == label).sum().item()  # 累加预测正确数
#                 val_total += label.size(0)  # 累加总样本数
#
#                 epoch_val_loss = val_loss / (step + 1)  # 当前验证平均loss
#                 epoch_val_acc = val_correct / val_total  # 当前验证准确率
#
#                 vepoch.set_postfix(  # 实时更新验证进度条信息
#                     batch_loss=f"{batch_loss.item():.4f}",  # 当前batch loss
#                     epoch_loss=f"{epoch_val_loss:.4f}",     # 当前平均loss
#                     acc=f"{epoch_val_acc:.4f}"              # 当前准确率
#                 )
#
#     val_losses.append(val_loss / len(val_loader))  # 保存验证集平均loss
#     val_accs.append(val_correct / val_total)       # 保存验证集准确率
#
#     # ===============================
#     # 📊 写入DataFrame（🔥关键部分）
#     # ===============================
#     training_results.loc[len(training_results)] = {  # 在末尾新增一行
#         "Epoch": epoch,
#         "Train Loss": train_losses[-1],
#         "Val Loss": val_losses[-1],
#         "Train Acc": train_accs[-1],
#         "Val Acc": val_accs[-1]
#     }
#
#     # ===============================
#     # 💾 保存Excel（每轮都保存，防止中断丢失）
#     # ===============================
#     training_results.to_excel(excel_path, index=False)
#
#     print(  # 打印当前epoch总结信息
#         f"Epoch {epoch}: "
#         f"Train Loss={train_losses[-1]:.6f}, "  # 打印列表里“最后一个元素”
#         f"Train Acc={train_accs[-1]:.4f}, "
#         f"Val Loss={val_losses[-1]:.6f}, "
#         f"Val Acc={val_accs[-1]:.4f}"
#     )
#
#     # ===============================
#     # 💾 保存模型
#     # ===============================
#     save_path = os.path.join(save_dir, f"FireFullModel_epoch_{epoch}.pt")  # 定义完整保存路径
#     torch.save(model_last.state_dict(), save_path)  # 保存模型参数（state_dict）
#     print(f"第{epoch}轮模型已保存")  # 打印当前是第几轮
#     print(f"📊 Excel updated: {excel_path}")
#     print(f"✅ Saved model: {save_path}")  # 打印具体保存文件路径
#
# # ===============================
# # Loss与Acc曲线
# # ===============================
# excel_path = "pt_yangnet/yangnet_pt/training_results.xlsx"  # Excel文件路径
#
# os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#
# # ===============================
# # 📊 读取Excel数据
# # ===============================
# df = pd.read_excel(excel_path)  # 读取Excel
#
# epochs = df["Epoch"]  # 读取epoch
# train_losses = df["Train Loss"]  # 训练loss
# val_losses = df["Val Loss"]  # 验证loss
# train_accs = df["Train Acc"]  # 训练acc
# val_accs = df["Val Acc"]  # 验证acc
#
# # ===============================
# # 🔴 绘制 Loss 曲线
# # ===============================
# plt.figure(figsize=(10, 5))  # 设置图像大小
#
# plt.plot(epochs, train_losses, label='Training Loss', color='blue')  # 训练loss
# plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--', color='red')  # 验证loss
#
# plt.xlabel('Epoch')  # x轴
# plt.ylabel('Loss')  # y轴
# plt.title('Training & Validation Loss')  # 标题
#
# plt.legend()  # 图例
# plt.grid(True)  # 网格
#
# # 自动范围（避免贴边）
# min_loss = min(min(train_losses), min(val_losses))
# max_loss = max(max(train_losses), max(val_losses))
# plt.ylim(min_loss * 0.9, max_loss * 1.1)
#
# plt.savefig(f'{save_path}/training_val_loss.png', dpi=300)  # 保存
# plt.show()  # 显示
#
# # ===============================
# # 🔵 绘制 Accuracy 曲线
# # ===============================
# plt.figure(figsize=(10, 5))  # 新建图
#
# plt.plot(epochs, train_accs, label='Training Acc', color='green')  # 训练acc
# plt.plot(epochs, val_accs, label='Validation Acc', linestyle='--', color='orange')  # 验证acc
#
# plt.xlabel('Epoch')  # x轴
# plt.ylabel('Accuracy')  # y轴
# plt.title('Training & Validation Accuracy')  # 标题
#
# plt.legend()  # 图例
# plt.grid(True)  # 网格
#
# # 自动范围
# min_acc = min(min(train_accs), min(val_accs))
# max_acc = max(max(train_accs), max(val_accs))
# plt.ylim(min_acc * 0.95, max_acc * 1.05)
#
# plt.savefig(f'{save_path}/training_val_acc.png', dpi=300)  # 保存
# plt.show()  # 显示
#
# # # ===============================
# # # Dataset
# # # ===============================
# # class FireSizeDataset(torch.utils.data.Dataset):  # 定义自定义数据集类
# #     def __init__(self, root_dir):  # 初始化函数
# #         self.samples = []  # 初始化样本列表
# #         self.label_map = {  # 定义标签映射字典
# #             "smallscene_nofire": 0,
# #             "smallscene_smallfire": 1,
# #             "smallscene_midfire": 2,
# #             "smallscene_bigfire": 3,
# #             "midscene_nofire": 4,
# #             "midscene_smallfire": 5,
# #             "midscene_midfire": 6,
# #             "midscene_bigfire": 7,
# #             "bigscene_nofire": 8,
# #             "bigscene_midfire": 9,
# #             "bigscene_bigfire": 10
# #         }
# #
# #         for cls, label in self.label_map.items():  # 遍历每个类别
# #             cls_dir = os.path.join(root_dir, cls)  # 构建类别目录路径
# #             for img_name in os.listdir(cls_dir):  # 遍历类别目录中的文件
# #                 if img_name.lower().endswith((".jpg", ".png", ".jpeg")):  # 检查是否为图像文件
# #                     self.samples.append(  # 添加到样本列表
# #                         (os.path.join(cls_dir, img_name), label)  # (图像路径, 标签)
# #                     )
# #
# #     def __len__(self):  # 返回数据集大小
# #         return len(self.samples)
# #
# #     def __getitem__(self, idx):  # 获取单个样本
# #         img_path, label = self.samples[idx]  # 获取图像路径和标签
# #
# #         ratio_path = os.path.join(  # 构建火焰比例文件路径
# #             os.path.dirname(img_path),
# #             "fire_ratio",
# #             os.path.basename(img_path) + ".npy"
# #         )
# #         fire_ratio = np.load(ratio_path).item()  # 加载火焰比例
# #
# #         yolo_res = scene_model(img_path, verbose=False)[0]  # 使用YOLO进行场景检测
# #         class_ids = (  # 获取检测到的类别ID
# #             yolo_res.boxes.cls.cpu().numpy()  # 从YOLO结果中提取类别
# #             if yolo_res.boxes is not None else []  # 如果没有检测到物体则为空列表
# #         )
# #
# #         scene_feat = scene_size_classification(class_ids)  # 获取场景特征向量
# #
# #         feature = torch.cat([  # 拼接特征向量
# #             scene_feat,  # 场景特征（3维）
# #             torch.tensor([fire_ratio])  # 火焰比例（1维）
# #         ])
# #
# #         return feature, torch.tensor(label, dtype=torch.long)  # 返回特征和标签
# #
# # # ===============================
# # # 执行预计算
# # # ===============================
# # precompute_fire_ratio_and_mask()  # 调用预计算函数
# #
# # # ===============================
# # # DataLoader
# # # ===============================
# # train_loader = torch.utils.data.DataLoader(  # 创建训练数据加载器
# #     FireSizeDataset(dataset_roots["train"]),  # 训练数据集
# #     batch_size=batch_size,  # 批量大小
# #     shuffle=True  # 打乱数据
# # )
# #
# # val_dataset = FireSizeDataset(dataset_roots["val"])  # 创建验证数据集
# # val_loader = torch.utils.data.DataLoader(  # 创建验证数据加载器
# #     val_dataset,  # 验证数据集
# #     batch_size=batch_size,  # 批量大小
# #     shuffle=False  # 不打乱数据
# # )
# #
# # # ===============================
# # # 训练与验证
# # # ===============================
# # criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
# # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器（Adam）
# #
# # train_losses, val_losses = [], []  # 初始化训练和验证损失列表
# # val_results = []  # 初始化验证结果列表
# #
# # for epoch in range(epochs):  # 遍历每个训练轮次
# #     # -------- Train --------
# #     model.train()  # 设置为训练模式
# #     train_loss = 0.0  # 初始化训练损失
# #
# #     for feats, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):  # 遍历训练批次
# #         feats, labels = feats.to(device), labels.to(device)  # 移动数据到设备
# #
# #         optimizer.zero_grad()  # 清零梯度
# #         outputs = model(feats)  # 前向传播
# #         loss = criterion(outputs, labels)  # 计算损失
# #         loss.backward()  # 反向传播
# #         optimizer.step()  # 更新参数
# #
# #         train_loss += loss.item()  # 累加损失
# #
# #     train_losses.append(train_loss / len(train_loader))  # 计算平均训练损失
# #
# #     # -------- Val --------
# #     model.eval()  # 设置为评估模式
# #     val_loss = 0.0  # 初始化验证损失
# #
# #     with torch.no_grad():  # 禁用梯度计算
# #         for bidx, (feats, labels) in enumerate(tqdm(val_loader, desc="Val")):  # 遍历验证批次
# #             feats, labels = feats.to(device), labels.to(device)  # 移动数据到设备
# #
# #             outputs = model(feats)  # 前向传播
# #             loss = criterion(outputs, labels)  # 计算损失
# #             val_loss += loss.item()  # 累加损失
# #
# #             preds = torch.argmax(outputs, dim=1)  # 获取预测类别
# #
# #             start = bidx * batch_size  # 计算批次起始索引
# #             samples = val_dataset.samples[start:start + len(labels)]  # 获取当前批次对应的样本
# #
# #             for i in range(len(labels)):  # 遍历当前批次的每个样本
# #                 img_path, true_label = samples[i]  # 获取图像路径和真实标签
# #                 pred_label = preds[i].item()  # 获取预测标签
# #
# #                 val_results.append({  # 添加验证结果
# #                     "Image_Path": img_path,  # 图像路径
# #                     "True_Label": idx_to_class[true_label],  # 真实标签（类别名）
# #                     "Pred_Label": idx_to_class[pred_label],  # 预测标签（类别名）
# #                     "Correct": int(true_label == pred_label)  # 是否正确预测（0或1）
# #                 })
# #
# #     val_losses.append(val_loss / len(val_loader))  # 计算平均验证损失
# #
# #     print(  # 打印训练信息
# #         f"Epoch {epoch+1}: "
# #         f"Train Loss={train_losses[-1]:.6f}, "  # 当前轮次的训练损失
# #         f"Val Loss={val_losses[-1]:.6f}"  # 当前轮次的验证损失
# #     )
# #
# # # ===============================
# # # 保存模型 & Excel
# # # ===============================
# # os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
# #
# # torch.save(model.state_dict(), f"{save_dir}/fire_size_fcnn.pt")  # 保存模型权重
# #
# # pd.DataFrame({  # 创建损失数据DataFrame
# #     "Epoch": range(1, epochs + 1),  # 轮次数
# #     "Train_Loss": train_losses,  # 训练损失
# #     "Val_Loss": val_losses  # 验证损失
# # }).to_excel(f"{save_dir}/train_val_loss.xlsx", index=False)  # 保存为Excel文件
# #
# # pd.DataFrame(val_results).to_excel(  # 创建验证结果DataFrame并保存为Excel
# #     f"{save_dir}/val_prediction_results.xlsx",
# #     index=False
# # )
# #
# # # ===============================
# # # Loss 曲线
# # # ===============================
# # plt.figure()  # 创建新图形
# # plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")  # 绘制训练损失曲线
# # plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")  # 绘制验证损失曲线
# # plt.xlabel("Epoch")  # 设置x轴标签
# # plt.ylabel("Loss")  # 设置y轴标签
# # plt.title("Fire Size FCNN Training & Validation Loss")  # 设置图形标题
# # plt.legend()  # 显示图例
# # plt.grid(True)  # 显示网格
# # plt.savefig(f"{save_dir}/train_val_loss_curve.png", dpi=300)  # 保存图形
# # plt.close()  # 关闭图形




































































#复杂的火势判断模型
import os  # 导入操作系统模块，用于文件和目录操作
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import cv2  # 导入OpenCV库，用于图像处理（读取、写入、变换等）
import time  #用于计算训练时间
import glob  # 导入 glob，用于批量查找文件
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
import torch.nn.functional as F  # 导入 PyTorch 的函数式接口（activation、loss、卷积等），并命名为 F，方便直接调用函数
import pandas as pd  # 导入Pandas库，用于数据处理
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图

from PIL import Image  # 导入PIL库，用于图像处理
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from torch.utils.data import Dataset  # 导入 PyTorch Dataset 基类，用于自定义数据集
from ultralytics import YOLO  # 导入YOLO模型
from torchvision import transforms  # 导入torchvision的图像变换模块
from torch.utils.data import DataLoader   # 用于按批次(batch)加载数据
from smallest_unet_model import smallest_UNet  # 导入自定义的UNet模型

# ===============================
# 超参数
# ===============================
epochs = 50  # 训练的总轮数
img_size = 640  # 图像尺寸
batch_size = 10  # 批量大小
dropout_p = 0.3  #随机失活
learning_rate = 1e-4  # 学习率
threshold_unet = 0.5007  # UNet分割的阈值
save_dir = "pt_yangnet/yangnet_pt_CNN+ratio"  # 保存yangnet模型的路径
pt_unet = "pt_yangnet/unet_pt/UNet_student_47.pt"  # UNet模型路径
pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"  # YOLO场景尺度模型路径
excel_path = os.path.join(save_dir, "training_results.xlsx")  # Excel保存路径
excel_path_loss_acc = "pt_yangnet/yangnet_pt_CNN+ratio/training_results.xlsx"  # Excel文件路径（用于话loss和acc图）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU可用性并设置设备
print("Using device:", device)  # 打印使用的设备

# ===============================
# UNet（离线分割）
# ===============================
fire_seg_model = smallest_UNet().to(device)  # 实例化UNet模型并移动到设备
fire_seg_model.load_state_dict(torch.load(pt_unet, map_location=device))  # 加载预训练权重
fire_seg_model.eval()  # 设置为评估模式

unet_transform = transforms.Compose([  # 定义图像预处理转换
    transforms.Resize((img_size, img_size)),  # 调整图像大小
    transforms.ToTensor()  # 转换为张量
])

# ===============================
# YOLO 场景尺度模型
# ===============================
scene_model = YOLO(pt_scene_size)  # 加载YOLO模型

# ===============================
# 场景尺度编码
# ===============================
def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):  # 定义场景尺度分类函数
    has_sky, has_trunk = False, False  # 初始化天空和树干标志
    for cid in cls_ids:  # 遍历检测到的类别ID
        if int(cid) == sky_id:  # 如果检测到天空
            has_sky = True  # 设置天空标志为True
        elif int(cid) == trunk_id:  # 如果检测到树干
            has_trunk = True  # 设置树干标志为True

    if has_trunk:  # 如果检测到树干（不管是否检测到天空都为近景）
        return torch.tensor([1, 0, 0], dtype=torch.float32)  # 返回[1,0,0]表示近景
    if has_sky:  # 如果检测到天空
        return torch.tensor([0, 0, 1], dtype=torch.float32)  # 返回[0,0,1]表示远景
    return torch.tensor([0, 1, 0], dtype=torch.float32)  # 否则返回[0,1,0]表示中景

# ===============================
# 数据路径
# ===============================
dataset_roots = {  # 定义数据集路径字典
    "train": "yangnet/forest_fire_size_classification_dataset_detail/train",  # 训练集路径
    "val": "yangnet/forest_fire_size_classification_dataset_detail/val"  # 验证集路径
}

classes = ["smallscene_nofire", "smallscene_smallfire", "smallscene_midfire", "smallscene_bigfire", "midscene_nofire",
           "midscene_smallfire", "midscene_midfire", "midscene_bigfire", "bigscene_nofire", "bigscene_midfire", "bigscene_bigfire"]  # 类别列表
idx_to_class = {0: "smallscene_nofire", 1: "smallscene_smallfire", 2: "smallscene_midfire", 3: "smallscene_bigfire",
                4: "midscene_nofire", 5: "midscene_smallfire", 6: "midscene_midfire", 7: "midscene_bigfire",
                8: "bigscene_nofire", 9: "bigscene_midfire", 10: "bigscene_bigfire"}  # 索引到类别的映射

# ===============================
# 预计算火焰比例 & 掩膜
# ===============================
def get_trunk_top_y(image, yolo_model):     # 获取所有trunk中最小的y1（只看class=1）
    """
    输入：PIL图像
    输出：树干最上边界y1（忽略sky=0，只使用trunk=1）
    """
    results = yolo_model(image)             # YOLO模型推理

    if results is None or len(results) == 0:    # 如果没有检测结果
        return None                         # 返回None

    if results[0].boxes is None:            # 如果没有检测框
        return None                         # 返回None

    boxes = results[0].boxes                # 获取检测框对象

    xyxy = boxes.xyxy.cpu().numpy()         # 获取坐标 (N,4)
    cls = boxes.cls.cpu().numpy()           # 获取类别 (N,)
    conf = boxes.conf.cpu().numpy()         # 获取置信度 (N,)

    trunk_y1_list = []                      # 用于存储所有trunk的y1

    for i in range(len(cls)):               # 遍历所有检测框

        if int(cls[i]) != 1:                # 只保留 trunk（class=1）
            continue                       # 跳过 sky

        if conf[i] < 0.5:                  # 过滤低置信度检测（可调）
            continue

        x1, y1, x2, y2 = xyxy[i]           # 获取坐标
        trunk_y1_list.append(y1)           # 收集y1

    if len(trunk_y1_list) == 0:            # 如果没有检测到trunk
        return None                        # 返回None

    return int(min(trunk_y1_list))       # 传统方法：取最小

def precompute_fire_ratio_and_mask_fixed():
    print("\n🔥 Precomputing fire masks, ratios, Sobel edges & trunk-cut masks (fixed version)")

    for split, root_dir in dataset_roots.items():   # train/val
        for cls_name in classes:                    # 遍历每个类别
            img_dir = os.path.join(root_dir, cls_name)
            ratio_dir = os.path.join(img_dir, "fire_ratio")
            edge_dir = os.path.join(img_dir, "fire_edge")
            trunk_mask_dir = os.path.join(img_dir, "fire_trunk_mask")

            os.makedirs(ratio_dir, exist_ok=True)
            os.makedirs(edge_dir, exist_ok=True)
            os.makedirs(trunk_mask_dir, exist_ok=True)

            for img_name in tqdm(os.listdir(img_dir), desc=f"{split}/{cls_name}"):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                name = os.path.splitext(img_name)[0]
                img_path = os.path.join(img_dir, img_name)

                # 输出路径
                ratio_path = os.path.join(ratio_dir, img_name + ".npy")
                mask_path = os.path.join(img_dir, name + "_mask.png")          # 可视化 0/255
                mask01_path = os.path.join(img_dir, name + "_mask01.png")      # 训练用 0/1
                trunk_mask_path = os.path.join(trunk_mask_dir, name + "_mask_trunk.png")
                trunk_mask_vis_path = os.path.join(trunk_mask_dir, name + "_mask_trunk_vis.png")
                sobelx_path = os.path.join(edge_dir, name + "_sobelx.png")
                sobely_path = os.path.join(edge_dir, name + "_sobely.png")
                sobelmag_path = os.path.join(edge_dir, name + "_sobelmag.png")

                # 跳过已处理
                if all(os.path.exists(p) for p in [ratio_path, mask_path, mask01_path, trunk_mask_path, sobelmag_path]):
                    continue

                # =========================
                # 1️⃣ UNet 分割
                # =========================
                image = Image.open(img_path).convert("RGB")
                image_tensor = unet_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = fire_seg_model(image_tensor)[0, 0]  # 预测 mask
                    pred = torch.sigmoid(pred)                 # 确保 0~1

                # =========================
                # 2️⃣ 二值化掩膜
                # =========================
                threshold = 0.5007
                mask01 = (pred > threshold).float().cpu().numpy().astype(np.uint8)  # 0/1 mask
                mask255 = (mask01 * 255).astype(np.uint8)                             # 可视化 mask

                # =========================
                # 3️⃣ 火焰比例
                # =========================
                fire_ratio = mask01.sum() / mask01.size
                np.save(ratio_path, fire_ratio)

                # =========================
                # 4️⃣ 保存掩膜
                # =========================
                Image.fromarray(mask01).save(mask01_path)
                Image.fromarray(mask255).save(mask_path)

                # =========================
                # 5️⃣ YOLO 树干裁剪
                # =========================
                y1 = get_trunk_top_y(image, scene_model)
                cropped_mask01 = mask01.copy()
                cropped_mask255 = mask255.copy()

                if y1 is not None:
                    h, w = cropped_mask01.shape
                    scale = h / image.height
                    y1 = int(y1 * scale)
                    y1 = max(0, min(h, y1))
                    cropped_mask01[y1:, :] = 0
                    cropped_mask255[y1:, :] = 0

                # 保存裁剪 mask
                Image.fromarray(cropped_mask01).save(trunk_mask_path)
                Image.fromarray(cropped_mask255).save(trunk_mask_vis_path)

                # =========================
                # 6️⃣ Sobel 边缘（基于裁剪后 mask）
                # =========================
                img_mask_float = cropped_mask255.astype(np.float32)
                sobelx = cv2.Sobel(img_mask_float, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img_mask_float, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = np.sqrt(sobelx**2 + sobely**2)

                # 转 0/1
                sobelx_bin = (np.abs(sobelx) > 0).astype(np.uint8)
                sobely_bin = (np.abs(sobely) > 0).astype(np.uint8)
                sobelmag_bin = (sobel_mag > 0).astype(np.uint8)

                # 保存 0/1 用于训练
                cv2.imwrite(sobelx_path, sobelx_bin)
                cv2.imwrite(sobely_path, sobely_bin)
                cv2.imwrite(sobelmag_path, sobelmag_bin)

                # 保存可视化 0/255
                cv2.imwrite(sobelx_path.replace(".png","_vis.png"), sobelx_bin*255)
                cv2.imwrite(sobely_path.replace(".png","_vis.png"), sobely_bin*255)
                cv2.imwrite(sobelmag_path.replace(".png","_vis.png"), sobelmag_bin*255)

    print("✅ Fire masks, ratios, Sobel edges & trunk-cut masks cached (fixed)\n")

# # ===============================
# # 执行预计算
# # ===============================
# precompute_fire_ratio_and_mask_fixed()  # 调用预计算函数

# ===============================
# 场景三维向量MLP
# ===============================
class SceneFeatureExtractor(nn.Module):
    def __init__(self):
        super(SceneFeatureExtractor, self).__init__()
        # MLP结构：3 → 16 → 32 → 64 → 128
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # Dropout 防过拟合
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Tanh(),  #加入这个防止出现训练不动的情况，限制一下输出的范围
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        """
        x: [batch_size, 3]
        """
        feat_128 = self.mlp(x)  # [B, 128]
        A1, B1 = torch.split(feat_128, 64, dim=1)  # 🔥 从中间拆分为两个64维
        return A1, B1

# ===============================
# 火焰图像比例一维向量MLP
# ===============================
class FireRatioFeatureExtractor(nn.Module):
    def __init__(self):
        super(FireRatioFeatureExtractor, self).__init__()
        # MLP结构：1 → 16 → 32 → 64 → 128
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        """
        x: [batch_size, 1]  （火焰比例）
        """
        feat_128 = self.mlp(x)  # [B, 128]
        A2, B2 = torch.split(feat_128, 64, dim=1)
        return A2, B2

# ===============================
# 火焰图像特征CNN网络
# ===============================
class FireVisualFeatureCNN_20x20_FC64_Optim(nn.Module):
    """
    输入: (batch, 5, 640, 640)
    输出: (batch, 64)
    卷积下采样到 20x20x128，再接优化后的递减型全连接输出64维向量
    """
    def __init__(self, out_dim=64):
        super(FireVisualFeatureCNN_20x20_FC64_Optim, self).__init__()

        # 卷积下采样到 20x20
        self.conv_block = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),    # 640->320
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 320->160
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 160->80
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 80->40
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 40->20
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 递减型全连接
        self.fc = nn.Sequential(
            nn.Linear(128*20*20, 256),  # 51200 -> 256
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(256, 128),        # 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(128, out_dim)     # 128 -> 64
        )

    def forward(self, x):
        x = self.conv_block(x)       # -> (batch, 128, 20, 20)
        x = x.view(x.size(0), -1)    # -> (batch, 128*20*20)
        visual_feat = self.fc(x)     # -> (batch, 64)
        return visual_feat

# ===============================
# 特征缩放网络及最后的分类网络
# ===============================
class F1F2FusionClassifier(nn.Module):
    def __init__(self, visual_feat_dim=64, num_classes=11):
        super(F1F2FusionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(visual_feat_dim * 2, 64),  # 128 -> 64
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, num_classes)  # 32 -> 11
        )

    def forward(self, visual_feat, A1, B1, A2, B2):
        """
        visual_feat: [B, 64] CNN输出的特征
        A1, B1, A2, B2: [B, 64] 可训练缩放和偏移量
        """
        # 逐元素变换
        F1 = A1 * visual_feat + B1  # [B,64]
        F2 = A2 * visual_feat + B2  # [B,64]

        # 拼接成128维
        fused_feat = torch.cat([F1, F2], dim=1)  # [B,128]

        # 分类
        # logits = self.classifier(fused_feat)  # [B,11]
        # probs = F.softmax(logits, dim=1)  # [B,11]
        # return probs, fused_feat, F1, F2
        # 在进行可视化的时候用下面这两行，训练的时候注释掉用上面三行
        logits = self.classifier(fused_feat)  # [B,11]
        return logits, fused_feat, F1, F2  # ✅ 直接返回logits

class FireVisualDataset(Dataset):
    """
    火焰数据集读取类（根据实际路径结构定制）
    每个样本包含：
        - 视觉输入：5个通道（sobelx, sobely, sobelmag, mask01, mask_trunk），形状 (5, H, W)
        - 场景向量：从 scene_vec 目录读取的 .npy 文件，形状 (3,)
        - 火焰比例：从 fire_ratio 目录读取的 .npy 文件，形状 (1,)
    标签对应类别（与 label_map 一致）
    """

    def __init__(self, root_dir, img_size=640, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform

        self.label_map = {
            "smallscene_nofire": 0,
            "smallscene_smallfire": 1,
            "smallscene_midfire": 2,
            "smallscene_bigfire": 3,
            "midscene_nofire": 4,
            "midscene_smallfire": 5,
            "midscene_midfire": 6,
            "midscene_bigfire": 7,
            "bigscene_nofire": 8,
            "bigscene_midfire": 9,
            "bigscene_bigfire": 10
        }

        self.data_list = []

        # 遍历每个类别
        for cls_name, cls_idx in self.label_map.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue

            mask_paths = glob.glob(os.path.join(cls_dir, "*_mask01.png"))
            for mask_path in mask_paths:
                base_name = self._extract_base_name(mask_path)

                # 构建所有相关文件路径
                sample = {
                    "sobelx": os.path.join(cls_dir, "fire_edge", f"{base_name}_sobelx.png"),
                    "sobely": os.path.join(cls_dir, "fire_edge", f"{base_name}_sobely.png"),
                    "sobelmag": os.path.join(cls_dir, "fire_edge", f"{base_name}_sobelmag.png"),
                    "mask01": mask_path,
                    "mask_trunk": os.path.join(cls_dir, "fire_trunk_mask", f"{base_name}_mask_trunk.png"),
                    "scene_vec": os.path.join(cls_dir, "scene_vec", f"{base_name}.npy"),
                    "fire_ratio": os.path.join(cls_dir, "fire_ratio", f"{base_name}.jpg.npy"),
                    "label": cls_idx
                }

                # 检查文件是否存在
                missing_files = [k for k, v in sample.items() if k != "label" and not os.path.exists(v)]
                if missing_files:
                    print(f"⚠️ WARNING: Missing files for sample {base_name}, skipped: {missing_files}")
                    continue

                self.data_list.append(sample)

    def _extract_base_name(self, mask_path):
        """
        从掩码文件路径中提取基础文件名（去掉 mask 后缀）
        例如:
            smallscene_nofire_0006_mask01.png -> smallscene_nofire_0006
        """
        filename = os.path.splitext(os.path.basename(mask_path))[0]
        base_name = re.sub(r'_mask\d*$', '', filename)  # 去掉 _mask, _mask01, _mask1 等后缀
        return base_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # ========== 读取视觉输入 ==========
        channels = []
        for key in ["sobelx", "sobely", "sobelmag", "mask01", "mask_trunk"]:
            img = Image.open(item[key]).convert("L").resize((self.img_size, self.img_size))
            img_arr = np.array(img, dtype=np.float32)
            channels.append(img_arr)

        visual_input = np.stack(channels, axis=0)
        visual_input = torch.tensor(visual_input, dtype=torch.float32)

        # ========== 读取场景向量 ==========
        scene_vec = np.load(item["scene_vec"])
        scene_vec = torch.tensor(scene_vec, dtype=torch.float32)

        # ========== 读取火焰比例 ==========
        fire_ratio_np = np.load(item["fire_ratio"])
        fire_ratio = torch.tensor([fire_ratio_np.item()], dtype=torch.float32)

        # ========== 标签 ==========
        label = torch.tensor(item["label"], dtype=torch.long)

        if self.transform:
            visual_input = self.transform(visual_input)

        return visual_input, scene_vec, fire_ratio, label

# #未使用消融实验
# class FireFullModel(nn.Module):
#     def __init__(self, num_classes=11):
#         super(FireFullModel, self).__init__()
#
#         # ===============================
#         # 1️⃣ 三个可训练子网络
#         # ===============================
#         self.scene_extractor = SceneFeatureExtractor()          # 3维 → A1,B1
#         self.ratio_extractor = FireRatioFeatureExtractor()      # 1维 → A2,B2
#         self.visual_cnn = FireVisualFeatureCNN_20x20_FC64_Optim()  # 图像 → visual_feat
#
#         # ===============================
#         # 2️⃣ 融合分类器
#         # ===============================
#         self.fusion_classifier = F1F2FusionClassifier(
#             visual_feat_dim=64,
#             num_classes=num_classes
#         )
#
#     def forward(self, visual_input, scene_vec, fire_ratio):
#         """
#         visual_input: [B, 5, 640, 640]
#         scene_vec:    [B, 3]
#         fire_ratio:   [B, 1]
#         """
#
#         # ===============================
#         # 1️⃣ CNN提取视觉特征
#         # ===============================
#         visual_feat = self.visual_cnn(visual_input)  # [B,64]
#
#         # ===============================
#         # 2️⃣ MLP提取缩放参数
#         # ===============================
#         A1, B1 = self.scene_extractor(scene_vec)     # [B,64]
#         A2, B2 = self.ratio_extractor(fire_ratio)    # [B,64]
#
#         # ===============================
#         # 3️⃣ 融合分类
#         # ===============================
#         logits, fused_feat, F1, F2 = self.fusion_classifier(
#             visual_feat, A1, B1, A2, B2
#         )
#
#         return logits, fused_feat, F1, F2

#使用消融实验
class FireFullModel(nn.Module):
    def __init__(self, num_classes=11, use_scene=True, use_ratio=True):
        super(FireFullModel, self).__init__()

        # 🔥 消融开关
        self.use_scene = use_scene
        self.use_ratio = use_ratio

        # ===============================
        # 1️⃣ 三个子网络（不删除！！）
        # ===============================
        self.scene_extractor = SceneFeatureExtractor()
        self.ratio_extractor = FireRatioFeatureExtractor()
        self.visual_cnn = FireVisualFeatureCNN_20x20_FC64_Optim()

        # ===============================
        # 2️⃣ 融合分类器
        # ===============================
        self.fusion_classifier = F1F2FusionClassifier(
            visual_feat_dim=64,
            num_classes=num_classes
        )

    def forward(self, visual_input, scene_vec, fire_ratio):

        # ===============================
        # 1️⃣ CNN视觉特征
        # ===============================
        visual_feat = self.visual_cnn(visual_input)  # [B,64]

        # ===============================
        # 2️⃣ Scene 分支（可关闭）
        # ===============================
        if self.use_scene:
            A1, B1 = self.scene_extractor(scene_vec)
        else:
            A1 = torch.ones_like(visual_feat)
            B1 = torch.zeros_like(visual_feat)

        # ===============================
        # 3️⃣ Ratio 分支（可关闭）
        # ===============================
        if self.use_ratio:
            A2, B2 = self.ratio_extractor(fire_ratio)
        else:
            A2 = torch.ones_like(visual_feat)
            B2 = torch.zeros_like(visual_feat)

        # ===============================
        # 4️⃣ 融合
        # ===============================
        logits, fused_feat, F1, F2 = self.fusion_classifier(
            visual_feat, A1, B1, A2, B2
        )

        return logits, fused_feat, F1, F2




# ===============================
# 训练集 DataLoader
# ===============================
train_dataset = FireVisualDataset(
    root_dir=dataset_roots["train"],  # 训练集根目录
    img_size=img_size,                # 图片统一尺寸
    transform=None                    # 可选 transform(输入的只有0和1，不需要再做归一化)
)
train_loader = DataLoader(
    dataset=train_dataset,             # 训练数据集
    batch_size=batch_size,             # 每个 batch 样本数
    shuffle=True,                      # 打乱训练数据
    num_workers=0                      # 多线程读取
)

# ===============================
# 验证集 DataLoader
# ===============================
val_dataset = FireVisualDataset(
    root_dir=dataset_roots["val"],     # 验证集根目录
    img_size=img_size,                 # 图片统一尺寸
    transform=None                     # 可选 transform
)
val_loader = DataLoader(
    val_dataset,                       # 验证数据集
    batch_size=batch_size,             # 每个 batch 样本数
    shuffle=False,                     # 验证数据不打乱
    num_workers=0                      # 多线程读取
)

# ===============================
# 训练与验证
# ===============================
#================================
#消融实验
#================================
# #只有CNN网络
# model_last = FireFullModel(
#     num_classes=11,
#     use_scene=False,
#     use_ratio=False
# ).to(device)

##CNN+scene
# model_last = FireFullModel(
#     num_classes=11,
#     use_scene=True,
#     use_ratio=False
# ).to(device)
#
#CNN+ratio
model_last = FireFullModel(
    num_classes=11,
    use_scene=False,
    use_ratio=True
).to(device)
#
# #CNN+scene+ratio（默认）
# model_last = FireFullModel(
#     num_classes=11,
#     use_scene=True,
#     use_ratio=True
# ).to(device)

criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
optimizer = optim.Adam(model_last.parameters(), lr=learning_rate)  # 定义优化器（Adam）

# ===============================
# 📊 初始化DataFrame
# ===============================
training_results = pd.DataFrame(columns=[  # 定义表头
    "Epoch",
    "Train Loss",
    "Val Loss",
    "Train Acc",
    "Val Acc",
    "all time"
])

# ===============================
# 训练记录
# ===============================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ===============================
# 开始训练
# ===============================
# 记录训练开始时间
all_start_time = time.time()
if __name__ == "__main__":
    for epoch in range(epochs):  # 遍历每一轮训练（epoch）

        print(f"第 {epoch} 轮训练开始")

        # ===============================
        # 🔥 Train
        # ===============================
        model_last.train()  # 设置模型为训练模式（启用Dropout和BatchNorm更新）
        train_loss = 0.0  # 初始化训练损失累计值
        train_correct = 0  # 初始化训练正确样本数
        train_total = 0    # 初始化训练样本总数
        start_train_time = time.time()

        with tqdm(train_loader, unit="batch", desc=f"Train {epoch}") as tepoch:  # 创建训练进度条

            for step, (visual_input, scene_vec, fire_ratio, label) in enumerate(tepoch):  # 遍历每一个batch

                visual_input = visual_input.to(device)  # 将图像输入移动到GPU或CPU
                scene_vec = scene_vec.to(device)        # 将场景三维向量移动到设备
                fire_ratio = fire_ratio.to(device)      # 将火焰比例数据移动到设备
                label = label.to(device)                # 将标签移动到设备

                logits, _, _, _ = model_last(visual_input, scene_vec, fire_ratio)  # 前向传播，得到logits输出

                batch_loss = criterion(logits, label)  # 计算当前batch的损失（交叉熵）

                optimizer.zero_grad()  # 清空上一轮计算的梯度
                batch_loss.backward()  # 反向传播，计算梯度
                optimizer.step()       # 更新模型参数

                train_loss += batch_loss.item()  # 累加当前batch的loss（用于计算epoch平均）

                preds = torch.argmax(logits, dim=1)  # 获取预测类别（取最大概率对应的索引）
                train_correct += (preds == label).sum().item()  # 累加预测正确的样本数
                train_total += label.size(0)  # 累加总样本数

                epoch_loss = train_loss / (step + 1)  # 当前epoch的平均loss（到当前batch为止）
                epoch_acc = train_correct / train_total  # 当前epoch的平均准确率

                tepoch.set_postfix(  # 在进度条右侧显示实时信息
                    batch_loss=f"{batch_loss.item():.4f}",  # 当前batch的loss
                    epoch_loss=f"{epoch_loss:.4f}",         # 当前epoch平均loss
                    acc=f"{epoch_acc:.4f}"                  # 当前准确率
                )

        train_losses.append(train_loss / len(train_loader))  # 保存当前epoch的平均训练损失
        train_accs.append(train_correct / train_total)       # 保存当前epoch的训练准确率

        end_train_time = time.time()  # 记录训练结束时间
        train_duration = end_train_time - start_train_time  # 计算训练轮次耗时
        minutes, seconds = divmod(train_duration, 60)
        print(f"第{epoch}轮训练结束，耗时 {int(minutes)} 分 {int(seconds)} 秒")

        # ===============================
        # 🔵 Val
        # ===============================
        model_last.eval()  # 设置模型为评估模式（关闭Dropout，不更新BN）
        val_loss = 0.0  # 初始化验证损失
        val_correct = 0  # 初始化验证正确数
        val_total = 0    # 初始化验证总数
        start_val_time = time.time()

        with torch.no_grad():  # 关闭梯度计算（节省显存，提高推理速度）
            with tqdm(val_loader, unit="batch", desc=f"Val {epoch}") as vepoch:  # 创建验证进度条

                for step, (visual_input, scene_vec, fire_ratio, label) in enumerate(vepoch):  # 遍历验证集

                    visual_input = visual_input.to(device)  # 图像数据移动到设备
                    scene_vec = scene_vec.to(device)        # 场景向量移动到设备
                    fire_ratio = fire_ratio.to(device)      # 火焰比例移动到设备
                    label = label.to(device)                # 标签移动到设备

                    logits, _, _, _ = model_last(visual_input, scene_vec, fire_ratio)  # 前向传播

                    batch_loss = criterion(logits, label)  # 计算当前batch的损失
                    val_loss += batch_loss.item()          # 累加验证损失

                    preds = torch.argmax(logits, dim=1)  # 获取预测类别
                    val_correct += (preds == label).sum().item()  # 累加预测正确数
                    val_total += label.size(0)  # 累加总样本数

                    epoch_val_loss = val_loss / (step + 1)  # 当前验证平均loss
                    epoch_val_acc = val_correct / val_total  # 当前验证准确率

                    vepoch.set_postfix(  # 实时更新验证进度条信息
                        batch_loss=f"{batch_loss.item():.4f}",  # 当前batch loss
                        epoch_loss=f"{epoch_val_loss:.4f}",     # 当前平均loss
                        acc=f"{epoch_val_acc:.4f}"              # 当前准确率
                    )

        val_losses.append(val_loss / len(val_loader))  # 保存验证集平均loss
        val_accs.append(val_correct / val_total)       # 保存验证集准确率

        end_val_time = time.time()  # 记录训练结束时间
        val_duration = end_val_time - start_val_time  # 计算训练轮次耗时
        minutes, seconds = divmod(val_duration, 60)
        print(f"第{epoch}轮验证结束，耗时 {int(minutes)} 分 {int(seconds)} 秒")

        # ===============================
        # 📊 写入DataFrame（🔥关键部分）
        # ===============================
        training_results.loc[len(training_results)] = {  # 在末尾新增一行
            "Epoch": epoch,
            "Train Loss": train_losses[-1],
            "Val Loss": val_losses[-1],
            "Train Acc": train_accs[-1],
            "Val Acc": val_accs[-1]
        }

        # ===============================
        # 💾 保存Excel（每轮都保存，防止中断丢失）
        # ===============================
        training_results.to_excel(excel_path, index=False)

        print(  # 打印当前epoch总结信息
            f"Epoch {epoch}: "
            f"Train Loss={train_losses[-1]:.6f}, "  # 打印列表里“最后一个元素”
            f"Train Acc={train_accs[-1]:.4f}, "
            f"Val Loss={val_losses[-1]:.6f}, "
            f"Val Acc={val_accs[-1]:.4f}"
        )

        # ===============================
        # 💾 保存模型
        # ===============================
        save_path = os.path.join(save_dir, f"FireFullModel_epoch_{epoch}.pt")  # 定义完整保存路径
        torch.save(model_last.state_dict(), save_path)  # 保存模型参数（state_dict）
        print(f"第{epoch}轮模型已保存")  # 打印当前是第几轮
        print(f"📊 Excel updated: {excel_path}")
        print(f"✅ Saved model: {save_path}")  # 打印具体保存文件路径

all_over_time = time.time()
all_over = all_over_time - all_start_time
all_hours, all_remainder = divmod(all_over, 3600)  # 转换为小时和剩余的秒数
all_minutes, all_seconds = divmod(all_remainder, 60)  # 转换为分钟和秒
print(f"训练总时长，耗时 {int(all_hours)} 小时 {int(all_minutes)} 分 {int(all_seconds)} 秒")

# 将训练总时间保存到DataFrame中
training_results.loc[len(training_results)] = {
    "Epoch": "Total",
    "Train Loss": None,
    "Val Loss": None,
    "Train Acc": None,
    "Val Acc": None,
    "all time": f'{int(all_hours)} 小时 {int(all_minutes)} 分 {int(all_seconds)} 秒'
}
training_results.to_excel(excel_path, index=False)

# ===============================
# Loss与Acc曲线
# ===============================
os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

# # ===============================
# # 📊 读取Excel数据
# # ===============================
# # df = pd.read_excel(excel_path)  # 读取Excel
# df = pd.read_excel(excel_path_loss_acc, engine='openpyxl')  # 使用openpyxl读取xlsx
# # 🔥 只保留数值行（过滤掉Total）
# df = df[pd.to_numeric(df["Epoch"], errors='coerce').notnull()]
#
# epochs = df["Epoch"]  # 读取epoch
# train_losses = df["Train Loss"]  # 训练loss
# val_losses = df["Val Loss"]  # 验证loss
# train_accs = df["Train Acc"]  # 训练acc
# val_accs = df["Val Acc"]  # 验证acc
#
# # ===============================
# # 🔴 绘制 Loss 曲线
# # ===============================
# plt.figure(figsize=(10, 5))  # 设置图像大小
#
# plt.plot(epochs, train_losses, label='Training Loss', color='blue')  # 训练loss
# plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--', color='red')  # 验证loss
#
# plt.xlabel('Epoch')  # x轴
# plt.ylabel('Loss')  # y轴
# plt.title('Training & Validation Loss')  # 标题
#
# plt.legend()  # 图例
# plt.grid(True)  # 网格
#
# # 自动范围（避免贴边）
# min_loss = min(min(train_losses), min(val_losses))
# max_loss = max(max(train_losses), max(val_losses))
# plt.ylim(min_loss * 0.9, max_loss * 1.1)
#
# plt.savefig(f'{save_dir}/training_val_loss.png', dpi=300)  # 保存
# plt.show()  # 显示
#
# # ===============================
# # 🔵 绘制 Accuracy 曲线
# # ===============================
# plt.figure(figsize=(10, 5))  # 新建图
#
# plt.plot(epochs, train_accs, label='Training Acc', color='green')  # 训练acc
# plt.plot(epochs, val_accs, label='Validation Acc', linestyle='--', color='orange')  # 验证acc
#
# plt.xlabel('Epoch')  # x轴
# plt.ylabel('Accuracy')  # y轴
# plt.title('Training & Validation Accuracy')  # 标题
#
# plt.legend()  # 图例
# plt.grid(True)  # 网格
#
# # 自动范围
# min_acc = min(min(train_accs), min(val_accs))
# max_acc = max(max(train_accs), max(val_accs))
# plt.ylim(min_acc * 0.95, max_acc * 1.05)
#
# plt.savefig(f'{save_dir}/training_val_acc.png', dpi=300)  # 保存
# plt.show()  # 显示
