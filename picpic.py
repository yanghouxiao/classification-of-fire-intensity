import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import cv2

#复杂的火势判断模型
import os  # 导入操作系统模块，用于文件和目录操作
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
learning_rate = 1e-3  # 学习率
img_size = 640  # 图像尺寸
batch_size = 10  # 批量大小
dropout_p = 0.3  #随机失活
threshold_unet = 0.5007  # UNet分割的阈值

save_dir = "pt_yangnet/yangnet_pt"  # 保存yangnet模型的路径
pt_unet = "pt_yangnet/unet_pt/UNet_student_47.pt"  # UNet模型路径
pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"  # YOLO场景尺度模型路径

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

# 假设 fire_seg_model、unet_transform、get_trunk_top_y、scene_model 已经定义
device = "cuda" if torch.cuda.is_available() else "cpu"

def precompute_fire_ratio_and_mask_smallscene_nofire():
    """
    只处理 smallscene_nofire 文件夹里的图片
    """
    print("\n🔥 Precomputing fire masks, ratios, Sobel edges & trunk-cut masks for smallscene_nofire")

    # 固定文件夹路径
    root_dir = "yangnet/forest_fire_size_classification_dataset_detail/train/smallscene_nofire"

    ratio_dir = os.path.join(root_dir, "fire_ratio")
    edge_dir = os.path.join(root_dir, "fire_edge")
    trunk_mask_dir = os.path.join(root_dir, "fire_trunk_mask")

    os.makedirs(ratio_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(trunk_mask_dir, exist_ok=True)

    # 遍历 smallscene_nofire 文件夹
    for img_name in tqdm(os.listdir(root_dir), desc="smallscene_nofire"):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        name = os.path.splitext(img_name)[0]
        img_path = os.path.join(root_dir, img_name)

        # 输出路径
        ratio_path = os.path.join(ratio_dir, img_name + ".npy")
        mask_path = os.path.join(root_dir, name + "_mask.png")          # 可视化 0/255
        mask01_path = os.path.join(root_dir, name + "_mask01.png")      # 训练用 0/1
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
            pred = torch.sigmoid(pred)                 # 0~1

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
        # 5️⃣ 树干裁剪
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
        # 6️⃣ Sobel 边缘
        # =========================
        img_mask_float = cropped_mask255.astype(np.float32)
        sobelx = cv2.Sobel(img_mask_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_mask_float, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)

        # 转 0/1
        sobelx_bin = (np.abs(sobelx) > 0).astype(np.uint8)
        sobely_bin = (np.abs(sobely) > 0).astype(np.uint8)
        sobelmag_bin = (sobel_mag > 0).astype(np.uint8)

        # 保存 0/1
        cv2.imwrite(sobelx_path, sobelx_bin)
        cv2.imwrite(sobely_path, sobely_bin)
        cv2.imwrite(sobelmag_path, sobelmag_bin)

        # 保存可视化 0/255
        cv2.imwrite(sobelx_path.replace(".png", "_vis.png"), sobelx_bin*255)
        cv2.imwrite(sobely_path.replace(".png", "_vis.png"), sobely_bin*255)
        cv2.imwrite(sobelmag_path.replace(".png", "_vis.png"), sobelmag_bin*255)

    print("✅ Fire masks, ratios, Sobel edges & trunk-cut masks cached for smallscene_nofire\n")

# ===============================
# 执行预计算
# ===============================
precompute_fire_ratio_and_mask_smallscene_nofire()