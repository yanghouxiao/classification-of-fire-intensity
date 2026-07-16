# import os  # 路径操作
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突
#
# import torch  # PyTorch
# import torch.nn as nn  # 神经网络模块
# import numpy as np  # 数值计算
# import cv2  # 图像处理
# import matplotlib.pyplot as plt  # 可视化
# from PIL import Image  # 图像读取
# import re  # 正则
#
# # =========================
# # 🔥 导入你的模型
# # =========================
# from ablation_experiment import FireFullModel
#
#
# # =========================
# # ✅ 1️⃣ Grad-CAM类
# # =========================
# class FireGradCAM:
#     def __init__(self, model, target_layer):  # 初始化
#         self.model = model  # 模型
#         self.target_layer = target_layer  # 目标卷积层
#
#         self.feature_maps = None  # 存特征图
#         self.gradients = None  # 存梯度
#
#         self._register_hooks()  # 注册hook
#
#     def _register_hooks(self):  # hook函数
#
#         def forward_hook(module, input, output):  # 前向hook
#             self.feature_maps = output  # 保存特征图
#
#         def backward_hook(module, grad_input, grad_output):  # 反向hook
#             self.gradients = grad_output[0]  # 保存梯度
#
#         self.target_layer.register_forward_hook(forward_hook)  # 注册前向
#         self.target_layer.register_full_backward_hook(backward_hook)  # 注册反向
#
#     def generate_cam(self, visual_input, scene_vec, fire_ratio, class_idx=None):  # 生成CAM
#
#         logits, _, _, _ = self.model(visual_input, scene_vec, fire_ratio)  # 前向（用logits）
#
#         if class_idx is None:  # 如果没指定类别
#             class_idx = torch.argmax(logits, dim=1).item()  # 取预测类别
#
#         self.model.zero_grad()  # 清梯度
#
#         class_score = logits[0, class_idx]  # 类别得分
#         class_score.backward()  # 反向传播
#
#         gradients = self.gradients.detach().cpu().numpy()[0]  # 梯度 [C,H,W]
#         feature_maps = self.feature_maps.detach().cpu().numpy()[0]  # 特征图
#
#         # 🔥 打印检查
#         print("🔥 grad mean:", np.mean(gradients))
#         print("🔥 feat mean:", np.mean(feature_maps))
#
#         weights = np.mean(gradients, axis=(1, 2))  # GAP权重
#
#         cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)  # 初始化CAM
#
#         for i, w in enumerate(weights):  # 加权求和
#             cam += w * feature_maps[i]
#
#         cam = np.maximum(cam, 0)  # ReLU
#
#         cam = cv2.resize(cam, (640, 640))  # 上采样
#
#         # 🔥 归一化 + 增强
#         cam = cam - np.min(cam)
#         cam = cam / (np.max(cam) + 1e-8)
#         cam = np.power(cam, 0.5)  # 🔥增强亮区域
#
#         return cam, class_idx
#
#
# # =========================
# # ✅ 2️⃣ CAM数值分析
# # =========================
# def analyze_cam(cam):
#     print("\n===== CAM数值分析 =====")
#     print("min:", np.min(cam))
#     print("max:", np.max(cam))
#     print("mean:", np.mean(cam))
#     print("std:", np.std(cam))
#     print("unique数量:", len(np.unique(cam)))
#
#     if np.max(cam) < 1e-6:
#         print("❌ CAM全黑（梯度没传回来）")
#
#     if np.std(cam) < 1e-5:
#         print("⚠️ CAM对比度极低")
#
#
# # =========================
# # ✅ 3️⃣ 找原图
# # =========================
# def get_original_image_path(mask_path):
#     base_dir = os.path.dirname(mask_path)  # 类目录
#     filename = os.path.basename(mask_path)  # 文件名
#
#     base_name = re.sub(r'_mask\d*\.png$', '', filename)  # 去mask
#
#     for ext in ['.jpg', '.png', '.jpeg']:  # 尝试不同格式
#         candidate = os.path.join(base_dir, base_name + ext)
#         if os.path.exists(candidate):
#             return candidate
#
#     return None
#
#
# # =========================
# # ✅ 4️⃣ 可视化（🔥终极版）
# # =========================
# def show_all(mask_path, cam, pred_class, mask_img, trunk_img):
#
#     original_path = get_original_image_path(mask_path)  # 找原图
#
#     if original_path is None:
#         print("❌ 原图不存在")
#         return
#
#     # 原图
#     image = Image.open(original_path).convert('RGB').resize((640, 640))
#     img_np = np.array(image).astype(np.float32) / 255.0
#
#     # CAM热力图
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = heatmap.astype(np.float32) / 255
#
#     # 普通叠加
#     overlay = img_np * 0.5 + heatmap * 0.5
#     overlay = np.clip(overlay, 0, 1)
#
#     # 🔥 CAM + mask
#     mask_3 = np.stack([mask_img]*3, axis=-1)
#     cam_mask_overlay = heatmap * mask_3
#
#     # 🔥 CAM + trunk
#     trunk_3 = np.stack([trunk_img]*3, axis=-1)
#     cam_trunk_overlay = heatmap * trunk_3
#
#     # =========================
#     # 🔥 显示
#     # =========================
#     plt.figure(figsize=(18, 10))
#
#     plt.subplot(2, 4, 1)
#     plt.imshow(img_np)
#     plt.title("Original")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 2)
#     plt.imshow(heatmap)
#     plt.title("Grad-CAM")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 3)
#     plt.imshow(overlay)
#     plt.title(f"Overlay (Class {pred_class})")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 4)
#     plt.imshow(mask_img, cmap='gray')
#     plt.title("Mask")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 5)
#     plt.imshow(trunk_img, cmap='gray')
#     plt.title("Mask Trunk")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 6)
#     plt.imshow(cam_mask_overlay)
#     plt.title("CAM ∩ Mask")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 7)
#     plt.imshow(cam_trunk_overlay)
#     plt.title("CAM ∩ Trunk")
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # =========================
# # ✅ 5️⃣ 主程序
# # =========================
# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model = FireFullModel(num_classes=11).to(device)
#
#     model_path = "pt_yangnet/yangnet_pt_all_raw/FireFullModel_epoch_49.pt"
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     model.eval()
#
#     # 🔥 最后一层Conv
#     target_layer = model.visual_cnn.conv_block[12]
#
#     grad_cam = FireGradCAM(model, target_layer)
#
#     # =========================
#     # ⚠️ 路径
#     # =========================
#     mask_path = "yangnet/forest_fire_size_classification_dataset_detail/train/smallscene_bigfire/smallscene_bigfire_0000_mask01.png"
#
#     base_name = re.sub(r'_mask\d*\.png$', '', os.path.basename(mask_path))
#     cls_dir = os.path.dirname(mask_path)
#
#     # =========================
#     # 🔥 读取0/1图
#     # =========================
#     def load_bin(path):
#         img = Image.open(path).convert("L").resize((640, 640))
#         img = np.array(img, dtype=np.float32)
#         img = (img > 0).astype(np.float32)
#         return img
#
#     sobelx = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobelx.png"))
#     sobely = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobely.png"))
#     sobelmag = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobelmag.png"))
#     mask_img = load_bin(mask_path)
#     trunk_img = load_bin(os.path.join(cls_dir, "fire_trunk_mask", base_name + "_mask_trunk.png"))
#
#     visual_input = np.stack([sobelx, sobely, sobelmag, mask_img, trunk_img], axis=0)
#
#     visual_input = torch.tensor(visual_input).unsqueeze(0).to(device)
#
#     scene_vec = torch.tensor(
#         np.load(os.path.join(cls_dir, "scene_vec", base_name + ".npy")),
#         dtype=torch.float32
#     ).unsqueeze(0).to(device)
#
#     fire_ratio = torch.tensor(
#         [[np.load(os.path.join(cls_dir, "fire_ratio", base_name + ".jpg.npy")).item()]],
#         dtype=torch.float32
#     ).to(device)
#
#     # =========================
#     # 🔥 CAM
#     # =========================
#     cam, pred_class = grad_cam.generate_cam(visual_input, scene_vec, fire_ratio)
#
#     # =========================
#     # 🔥 分析
#     # =========================
#     analyze_cam(cam)
#
#     # =========================
#     # 🔥 可视化
#     # =========================
#     show_all(mask_path, cam, pred_class, mask_img, trunk_img)


# #只有mask01通道时
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import re
#
# from ablation_experiment import FireFullModel
#
#
# # =========================
# # 🔥 Grad-CAM类
# # =========================
# class FireGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#
#         self.feature_maps = None
#         self.gradients = None
#
#         self._register_hooks()
#
#     def _register_hooks(self):
#
#         def forward_hook(module, input, output):
#             self.feature_maps = output
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0]
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)
#
#     def generate_cam(self, visual_input, scene_vec, fire_ratio, class_idx=None):
#
#         logits, _, _, _ = self.model(visual_input, scene_vec, fire_ratio)
#
#         if class_idx is None:
#             class_idx = torch.argmax(logits, dim=1).item()
#
#         self.model.zero_grad()
#
#         class_score = logits[0, class_idx]
#         class_score.backward()
#
#         gradients = self.gradients.detach().cpu().numpy()[0]
#         feature_maps = self.feature_maps.detach().cpu().numpy()[0]
#
#         weights = np.mean(gradients, axis=(1, 2))
#
#         cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
#
#         for i, w in enumerate(weights):
#             cam += w * feature_maps[i]
#
#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, (640, 640))
#
#         cam = cam - np.min(cam)
#         cam = cam / (np.max(cam) + 1e-8)
#         cam = np.power(cam, 0.5)
#
#         return cam, class_idx
#
#
# # =========================
# # ✅ 找原图
# # =========================
# def get_original_image_path(mask_path):
#     base_dir = os.path.dirname(mask_path)
#     filename = os.path.basename(mask_path)
#
#     base_name = re.sub(r'_mask\d*\.png$', '', filename)
#
#     for ext in ['.jpg', '.png', '.jpeg']:
#         candidate = os.path.join(base_dir, base_name + ext)
#         if os.path.exists(candidate):
#             return candidate
#
#     return None
#
#
# # =========================
# # ✅ 可视化
# # =========================
# def show_all(mask_path, cam, pred_class, mask_img):
#
#     original_path = get_original_image_path(mask_path)
#
#     if original_path is None:
#         print("❌ 原图不存在")
#         return
#
#     image = Image.open(original_path).convert('RGB').resize((640, 640))
#     img_np = np.array(image).astype(np.float32) / 255.0
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = heatmap.astype(np.float32) / 255
#
#     overlay = img_np * 0.5 + heatmap * 0.5
#     overlay = np.clip(overlay, 0, 1)
#
#     mask_3 = np.stack([mask_img]*3, axis=-1)
#     cam_mask_overlay = heatmap * mask_3
#
#     plt.figure(figsize=(15, 6))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(img_np)
#     plt.title("Original")
#     plt.axis('off')
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(heatmap)
#     plt.title("Grad-CAM")
#     plt.axis('off')
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(overlay)
#     plt.title(f"Overlay (Class {pred_class})")
#     plt.axis('off')
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(cam_mask_overlay)
#     plt.title("CAM ∩ Mask")
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # =========================
# # ✅ 主程序
# # =========================
# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model = FireFullModel(num_classes=11, in_channels=1).to(device)
#
#     model_path = "pt_yangnet/yangnet_pt_mask01/FireFullModel_epoch_49.pt"
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     model.eval()
#
#     # 🔥 最后一层Conv
#     target_layer = model.visual_cnn.conv_block[12]
#
#     grad_cam = FireGradCAM(model, target_layer)
#
#     # =========================
#     # 🔥 路径
#     # =========================
#     mask_path = "yangnet/forest_fire_size_classification_dataset_detail_notrunkcut/train/smallscene_bigfire/smallscene_bigfire_0000_mask01.png"
#
#     base_name = re.sub(r'_mask\d*\.png$', '', os.path.basename(mask_path))
#     cls_dir = os.path.dirname(mask_path)
#
#     # =========================
#     # 🔥 读取 mask ONLY
#     # =========================
#     def load_bin(path):
#         img = Image.open(path).convert("L").resize((640, 640))
#         img = np.array(img, dtype=np.float32)
#         img = (img > 0).astype(np.float32)
#         return img
#
#     mask_img = load_bin(mask_path)
#
#     # ⭐⭐⭐ 核心修改：只用1通道
#     visual_input = np.expand_dims(mask_img, axis=0)
#
#     visual_input = torch.tensor(visual_input).unsqueeze(0).to(device)
#
#     scene_vec = torch.tensor(
#         np.load(os.path.join(cls_dir, "scene_vec", base_name + ".npy")),
#         dtype=torch.float32
#     ).unsqueeze(0).to(device)
#
#     fire_ratio = torch.tensor(
#         [[np.load(os.path.join(cls_dir, "fire_ratio", base_name + ".jpg.npy")).item()]],
#         dtype=torch.float32
#     ).to(device)
#
#     cam, pred_class = grad_cam.generate_cam(visual_input, scene_vec, fire_ratio)
#
#     show_all(mask_path, cam, pred_class, mask_img)


# #mask01+masktrunk
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import re
#
# from ablation_experiment import FireFullModel
#
#
# # =========================
# # 🔥 Grad-CAM类
# # =========================
# class FireGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#
#         self.feature_maps = None
#         self.gradients = None
#
#         self._register_hooks()
#
#     def _register_hooks(self):
#
#         def forward_hook(module, input, output):
#             self.feature_maps = output
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0]
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)
#
#     def generate_cam(self, visual_input, scene_vec, fire_ratio, class_idx=None):
#
#         logits, _, _, _ = self.model(visual_input, scene_vec, fire_ratio)
#
#         if class_idx is None:
#             class_idx = torch.argmax(logits, dim=1).item()
#
#         self.model.zero_grad()
#
#         class_score = logits[0, class_idx]
#         class_score.backward()
#
#         gradients = self.gradients.detach().cpu().numpy()[0]
#         feature_maps = self.feature_maps.detach().cpu().numpy()[0]
#
#         weights = np.mean(gradients, axis=(1, 2))
#
#         cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
#
#         for i, w in enumerate(weights):
#             cam += w * feature_maps[i]
#
#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, (640, 640))
#
#         cam = cam - np.min(cam)
#         cam = cam / (np.max(cam) + 1e-8)
#         cam = np.power(cam, 0.5)
#
#         return cam, class_idx
#
#
# # =========================
# # ✅ 找原图
# # =========================
# def get_original_image_path(mask_path):
#     base_dir = os.path.dirname(mask_path)
#     filename = os.path.basename(mask_path)
#
#     base_name = re.sub(r'_mask\d*\.png$', '', filename)
#
#     for ext in ['.jpg', '.png', '.jpeg']:
#         candidate = os.path.join(base_dir, base_name + ext)
#         if os.path.exists(candidate):
#             return candidate
#
#     return None
#
#
# # =========================
# # ✅ 可视化（升级版）
# # =========================
# def show_all(mask_path, cam, pred_class, mask_img, trunk_img):
#
#     original_path = get_original_image_path(mask_path)
#
#     if original_path is None:
#         print("❌ 原图不存在")
#         return
#
#     image = Image.open(original_path).convert('RGB').resize((640, 640))
#     img_np = np.array(image).astype(np.float32) / 255.0
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = heatmap.astype(np.float32) / 255
#
#     overlay = img_np * 0.5 + heatmap * 0.5
#     overlay = np.clip(overlay, 0, 1)
#
#     # mask overlay
#     mask_3 = np.stack([mask_img]*3, axis=-1)
#     cam_mask_overlay = heatmap * mask_3
#
#     # trunk overlay
#     trunk_3 = np.stack([trunk_img]*3, axis=-1)
#     cam_trunk_overlay = heatmap * trunk_3
#
#     plt.figure(figsize=(18, 8))
#
#     plt.subplot(2, 4, 1)
#     plt.imshow(img_np)
#     plt.title("Original")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 2)
#     plt.imshow(heatmap)
#     plt.title("Grad-CAM")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 3)
#     plt.imshow(overlay)
#     plt.title(f"Overlay (Class {pred_class})")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 4)
#     plt.imshow(mask_img, cmap='gray')
#     plt.title("Mask01")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 5)
#     plt.imshow(trunk_img, cmap='gray')
#     plt.title("Mask Trunk")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 6)
#     plt.imshow(cam_mask_overlay)
#     plt.title("CAM ∩ Mask")
#     plt.axis('off')
#
#     plt.subplot(2, 4, 7)
#     plt.imshow(cam_trunk_overlay)
#     plt.title("CAM ∩ Trunk")
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # =========================
# # ✅ 主程序
# # =========================
# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # ⭐⭐⭐ 关键：2通道模型
#     model = FireFullModel(num_classes=11, in_channels=2).to(device)
#
#     model_path = "pt_yangnet/yangnet_pt_mask01+sobel_raw/FireFullModel_epoch_49.pt"
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     model.eval()
#
#     target_layer = model.visual_cnn.conv_block[12]
#
#     grad_cam = FireGradCAM(model, target_layer)
#
#     # =========================
#     # 路径
#     # =========================
#     mask_path = "yangnet/forest_fire_size_classification_dataset_detail/train/smallscene_bigfire/smallscene_bigfire_0000_mask01.png"
#
#     base_name = re.sub(r'_mask\d*\.png$', '', os.path.basename(mask_path))
#     cls_dir = os.path.dirname(mask_path)
#
#     # =========================
#     # 读取数据
#     # =========================
#     def load_bin(path):
#         img = Image.open(path).convert("L").resize((640, 640))
#         img = np.array(img, dtype=np.float32)
#         img = (img > 0).astype(np.float32)
#         return img
#
#     mask_img = load_bin(mask_path)
#     trunk_img = load_bin(os.path.join(cls_dir, "fire_trunk_mask", base_name + "_mask_trunk.png"))
#
#     # ⭐⭐⭐ 关键：2通道输入
#     visual_input = np.stack([mask_img, trunk_img], axis=0)
#
#     visual_input = torch.tensor(visual_input).unsqueeze(0).to(device)
#
#     scene_vec = torch.tensor(
#         np.load(os.path.join(cls_dir, "scene_vec", base_name + ".npy")),
#         dtype=torch.float32
#     ).unsqueeze(0).to(device)
#
#     fire_ratio = torch.tensor(
#         [[np.load(os.path.join(cls_dir, "fire_ratio", base_name + ".jpg.npy")).item()]],
#         dtype=torch.float32
#     ).to(device)
#
#     cam, pred_class = grad_cam.generate_cam(visual_input, scene_vec, fire_ratio)
#
#     show_all(mask_path, cam, pred_class, mask_img, trunk_img)


#mask01+sobel
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import re

from ablation_experiment import FireFullModel


# =========================
# 🔥 Grad-CAM类
# =========================
class FireGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.feature_maps = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, visual_input, scene_vec, fire_ratio, class_idx=None):

        logits, _, _, _ = self.model(visual_input, scene_vec, fire_ratio)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()

        class_score = logits[0, class_idx]
        class_score.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        feature_maps = self.feature_maps.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (640, 640))

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam = np.power(cam, 0.5)

        return cam, class_idx


# =========================
# ✅ 找原图
# =========================
def get_original_image_path(mask_path):
    base_dir = os.path.dirname(mask_path)
    filename = os.path.basename(mask_path)

    base_name = re.sub(r'_mask\d*\.png$', '', filename)

    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = os.path.join(base_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


# =========================
# ✅ 可视化
# =========================
def show_all(mask_path, cam, pred_class, mask_img):

    original_path = get_original_image_path(mask_path)

    if original_path is None:
        print("❌ 原图不存在")
        return

    image = Image.open(original_path).convert('RGB').resize((640, 640))
    img_np = np.array(image).astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    overlay = img_np * 0.5 + heatmap * 0.5
    overlay = np.clip(overlay, 0, 1)

    mask_3 = np.stack([mask_img]*3, axis=-1)
    cam_mask_overlay = heatmap * mask_3

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay (Class {pred_class})")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cam_mask_overlay)
    plt.title("CAM ∩ Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# ✅ 主程序
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⭐⭐⭐ 4通道模型
    model = FireFullModel(num_classes=11, in_channels=4).to(device)

    model_path = "pt_yangnet/yangnet_pt_mask01+sobel_trunk/FireFullModel_epoch_49.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    target_layer = model.visual_cnn.conv_block[12]

    grad_cam = FireGradCAM(model, target_layer)

    # =========================
    # 🔥 路径
    # =========================
    mask_path = "yangnet/forest_fire_size_classification_dataset_detail_notrunkcut/train/smallscene_bigfire/smallscene_bigfire_0000_mask01.png"

    base_name = re.sub(r'_mask\d*\.png$', '', os.path.basename(mask_path))
    cls_dir = os.path.dirname(mask_path)

    # =========================
    # 🔥 读取函数
    # =========================
    def load_bin(path):
        img = Image.open(path).convert("L").resize((640, 640))
        img = np.array(img, dtype=np.float32)
        img = (img > 0).astype(np.float32)
        return img

    # ⭐⭐⭐ 读取未trunk裁剪的sobel
    sobelx = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobelx.png"))
    sobely = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobely.png"))
    sobelmag = load_bin(os.path.join(cls_dir, "fire_edge", base_name + "_sobelmag.png"))
    mask_img = load_bin(mask_path)

    # ⭐⭐⭐ 4通道输入
    visual_input = np.stack([mask_img, sobelx, sobely, sobelmag], axis=0)

    visual_input = torch.tensor(visual_input).unsqueeze(0).to(device)

    scene_vec = torch.tensor(
        np.load(os.path.join(cls_dir, "scene_vec", base_name + ".npy")),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    fire_ratio = torch.tensor(
        [[np.load(os.path.join(cls_dir, "fire_ratio", base_name + ".jpg.npy")).item()]],
        dtype=torch.float32
    ).to(device)

    cam, pred_class = grad_cam.generate_cam(visual_input, scene_vec, fire_ratio)

    show_all(mask_path, cam, pred_class, mask_img)



