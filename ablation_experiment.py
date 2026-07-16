#复杂的火势判断模型
import os  # 导入操作系统模块，用于文件和目录操作
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re  # 用于处理文件名
import cv2  # 导入OpenCV库，用于图像处理（读取、写入、变换等）
import time  #用于计算训练时间
import glob  # 导入 glob，用于批量查找文件
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入PyTorch优化器模块
import torch.nn.functional as F  # 导入 PyTorch 的函数式接口（activation、loss、卷积等），并命名为 F，方便直接调用函数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图

from PIL import Image  # 导入PIL库，用于图像处理
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from ultralytics import YOLO  # 导入YOLO模型
from torchvision import transforms  # 导入torchvision的图像变换模块
from torch.utils.data import Dataset  # 导入 PyTorch Dataset 基类，用于自定义数据集
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


def precompute_fire_ratio_and_mask_fixed():
    print("\n🔥 Precomputing fire masks, ratios, Sobel edges (NO trunk), and trunk masks")

    for split, root_dir in dataset_roots.items():   # train/val
        for cls_name in classes:
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

                # =========================
                # 路径
                # =========================
                ratio_path = os.path.join(ratio_dir, img_name + ".npy")

                mask_path = os.path.join(img_dir, name + "_mask.png")
                mask01_path = os.path.join(img_dir, name + "_mask01.png")

                trunk_mask_path = os.path.join(trunk_mask_dir, name + "_mask_trunk.png")
                trunk_mask_vis_path = os.path.join(trunk_mask_dir, name + "_mask_trunk_vis.png")

                sobelx_path = os.path.join(edge_dir, name + "_sobelx.png")
                sobely_path = os.path.join(edge_dir, name + "_sobely.png")
                sobelmag_path = os.path.join(edge_dir, name + "_sobelmag.png")

                # =========================
                # 跳过（这里改成只检查sobel即可）
                # =========================
                if all(os.path.exists(p) for p in [ratio_path, mask01_path, sobelmag_path]):
                    continue

                # =========================
                # 1️⃣ UNet 分割
                # =========================
                image = Image.open(img_path).convert("RGB")
                image_tensor = unet_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = fire_seg_model(image_tensor)[0, 0]
                    pred = torch.sigmoid(pred)

                # =========================
                # 2️⃣ 二值化
                # =========================
                threshold = 0.5007
                mask01 = (pred > threshold).float().cpu().numpy().astype(np.uint8)
                mask255 = (mask01 * 255).astype(np.uint8)

                # =========================
                # 3️⃣ 火焰比例
                # =========================
                fire_ratio = mask01.sum() / mask01.size
                np.save(ratio_path, fire_ratio)

                # =========================
                # 4️⃣ 保存mask
                # =========================
                Image.fromarray(mask01).save(mask01_path)
                Image.fromarray(mask255).save(mask_path)

                # =========================
                # 5️⃣ trunk mask（仍然保留）
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

                Image.fromarray(cropped_mask01).save(trunk_mask_path)
                Image.fromarray(cropped_mask255).save(trunk_mask_vis_path)

                # =========================
                # 🔥 6️⃣ Sobel（🔥改这里：用原始mask01）
                # =========================
                img_mask_float = mask255.astype(np.float32)  # ❗不再用 cropped_mask

                sobelx = cv2.Sobel(img_mask_float, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img_mask_float, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = np.sqrt(sobelx**2 + sobely**2)

                # 二值化
                sobelx_bin = (np.abs(sobelx) > 0).astype(np.uint8)
                sobely_bin = (np.abs(sobely) > 0).astype(np.uint8)
                sobelmag_bin = (sobel_mag > 0).astype(np.uint8)

                # 保存训练用（0/1）
                cv2.imwrite(sobelx_path, sobelx_bin)
                cv2.imwrite(sobely_path, sobely_bin)
                cv2.imwrite(sobelmag_path, sobelmag_bin)

                # 保存可视化（0/255）
                cv2.imwrite(sobelx_path.replace(".png", "_vis.png"), sobelx_bin * 255)
                cv2.imwrite(sobely_path.replace(".png", "_vis.png"), sobely_bin * 255)
                cv2.imwrite(sobelmag_path.replace(".png", "_vis.png"), sobelmag_bin * 255)

    print("✅ Done: Sobel based on ORIGINAL mask (no trunk influence)")


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
    def __init__(self, in_channels=5, out_dim=64):
        super(FireVisualFeatureCNN_20x20_FC64_Optim, self).__init__()

        # 卷积下采样到 20x20
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),    # 640->320
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

    def __init__(self, root_dir, img_size=640, transform=None, input_keys=None):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform

        if input_keys is None:
            self.input_keys = ["sobelx", "sobely", "sobelmag", "mask01", "mask_trunk"]
        else:
            self.input_keys = input_keys

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
        for key in self.input_keys:  # 🔥 动态输入
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
    def __init__(self, num_classes=11, use_scene=True, use_ratio=True, in_channels=5):
        super(FireFullModel, self).__init__()

        # 🔥 消融开关
        self.use_scene = use_scene
        self.use_ratio = use_ratio

        # ===============================
        # 1️⃣ 三个子网络（不删除！！）
        # ===============================
        self.scene_extractor = SceneFeatureExtractor()
        self.ratio_extractor = FireRatioFeatureExtractor()
        self.visual_cnn = FireVisualFeatureCNN_20x20_FC64_Optim(in_channels=in_channels)

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
# CNN消融实验开关
# ===============================
#只有U-Net得到的mask图
input_keys = ["mask01"]
in_channels = 1

# #mask图与被trunk裁切过的mask_trunk
# input_keys = ["mask01", "mask_trunk"]
# in_channels = 2

# #mask图与mask图的sobel图
# input_keys = ["mask01", "sobelx", "sobely", "sobelmag"]
# in_channels = 4

# #mask图与mask图完整的
# input_keys = ["sobelx", "sobely", "sobelmag", "mask01", "mask_trunk"]
# in_channels = 5

# #mask图与mask_trunk图的sobel图
# input_keys = ["mask01", "sobelx", "sobely", "sobelmag"]
# in_channels = 4

# ===============================
# 训练集 DataLoader
# ===============================
train_dataset = FireVisualDataset(
    root_dir=dataset_roots["train"],  # 训练集根目录
    img_size=img_size,                # 图片统一尺寸
    input_keys=input_keys,
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
    input_keys=input_keys,
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
# #CNN+ratio
# model_last = FireFullModel(
#     num_classes=11,
#     use_scene=False,
#     use_ratio=True
# ).to(device)

#CNN+scene+ratio（默认）
model_last = FireFullModel(
    num_classes=11,
    use_scene=True,
    use_ratio=True,
    in_channels=in_channels
).to(device)

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

# all_over_time = time.time()
# all_over = all_over_time - all_start_time
# all_hours, all_remainder = divmod(all_over, 3600)  # 转换为小时和剩余的秒数
# all_minutes, all_seconds = divmod(all_remainder, 60)  # 转换为分钟和秒
# print(f"训练总时长，耗时 {int(all_hours)} 小时 {int(all_minutes)} 分 {int(all_seconds)} 秒")
#
# # 将训练总时间保存到DataFrame中
# training_results.loc[len(training_results)] = {
#     "Epoch": "Total",
#     "Train Loss": None,
#     "Val Loss": None,
#     "Train Acc": None,
#     "Val Acc": None,
#     "all time": f'{int(all_hours)} 小时 {int(all_minutes)} 分 {int(all_seconds)} 秒'
# }
# training_results.to_excel(excel_path, index=False)
#
# # ===============================
# # Loss与Acc曲线
# # ===============================
# os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
#
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