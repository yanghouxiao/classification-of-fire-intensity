import time                      # 用于计算推理时间
import torch                     # PyTorch深度学习框架
import numpy as np               # 数值计算
import cv2                       # 图像处理（Sobel边缘）
from PIL import Image            # 读取图片
from ultralytics import YOLO     # YOLO目标检测模型
from torchvision import transforms  # 图像预处理
import matplotlib.pyplot as plt

# ===============================
# 参数设置
# ===============================
img_path = "yangnet/forest_fire_size_classification_dataset_detail/train/smallscene_bigfire/smallscene_bigfire_0000.jpg"  # 测试图片路径
img_size = 640                  # 输入模型的统一尺寸
threshold_unet = 0.5007         # UNet分割阈值（二值化）

# 自动选择GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 加载模型
# ===============================
from smallest_unet_model import smallest_UNet   # UNet分割模型
from ablation_experiment import FireFullModel   # 火势分类模型

# -------- UNet --------
unet_model = smallest_UNet().to(device)  # 加载模型到设备
unet_model.load_state_dict(torch.load("pt_yangnet/unet_pt/UNet_student_47.pt", map_location=device))  # 加载权重
unet_model.eval()  # 评估模式（关闭dropout）

# -------- YOLO --------
scene_model = YOLO("pt_yangnet/scene_size_pt/best.pt")  # 加载YOLO模型（用于场景+trunk检测）

# -------- 主模型 --------
model = FireFullModel(
    num_classes=11,   # 分类类别数
    use_scene=True,   # 是否使用scene分支
    use_ratio=True,   # 是否使用ratio分支
    in_channels=1     # 输入通道数（mask + mask_trunk + sobelx + sobely + sobelmag）
).to(device)

model.load_state_dict(torch.load("pt_yangnet/yangnet_pt_mask01/FireFullModel_epoch_49.pt", map_location=device))
model.eval()  # 推理模式

# ===============================
# 图像预处理
# ===============================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # resize到640×640
    transforms.ToTensor()                     # 转为Tensor [0,1]
])

# ===============================
# 场景分类函数
# ===============================
def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):
    """
    根据YOLO检测结果判断场景尺度
    """
    has_sky, has_trunk = False, False

    for cid in cls_ids:
        if int(cid) == sky_id:
            has_sky = True
        elif int(cid) == trunk_id:
            has_trunk = True

    # 优先级：trunk > sky > none
    if has_trunk:
        return torch.tensor([1, 0, 0], dtype=torch.float32)  # 近景
    if has_sky:
        return torch.tensor([0, 0, 1], dtype=torch.float32)  # 远景
    return torch.tensor([0, 1, 0], dtype=torch.float32)      # 中景

# ===============================
# trunk裁剪函数
# ===============================
def get_trunk_top_y_from_results(results):
    """
    从YOLO结果中获取最靠上的trunk边界y坐标
    """
    if results is None or len(results) == 0:
        return None
    if results[0].boxes is None:
        return None

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标
    cls = boxes.cls.cpu().numpy()    # 类别
    conf = boxes.conf.cpu().numpy()  # 置信度

    trunk_y1_list = []

    for i in range(len(cls)):
        if int(cls[i]) != 1:   # 只保留trunk类别
            continue
        if conf[i] < 0.5:      # 过滤低置信度
            continue

        x1, y1, x2, y2 = xyxy[i]
        trunk_y1_list.append(y1)

    if len(trunk_y1_list) == 0:
        return None

    return int(min(trunk_y1_list))  # 取最靠上的trunk

# ===============================
# 读取图像
# ===============================
image = Image.open(img_path).convert("RGB")  # 读取RGB图像

# ===============================
# 🔥 GPU预热（避免第一次推理过慢）
# ===============================
for _ in range(5):
    dummy_input = torch.randn(1, 1, 640, 640).to(device)
    dummy_scene = torch.randn(1, 3).to(device)
    dummy_ratio = torch.randn(1, 1).to(device)
    _ = model(dummy_input, dummy_scene, dummy_ratio)

if device.type == "cuda":
    torch.cuda.synchronize()  # 等GPU执行完

# ===============================
# 🚀 开始计时
# ===============================
start = time.time()

# ===============================
# 1️⃣ UNet分割
# ===============================
img_tensor = transform(image).unsqueeze(0).to(device)  # [1,3,640,640]

with torch.no_grad():
    pred = unet_model(img_tensor)[0, 0]  # 取mask通道
    pred = torch.sigmoid(pred)           # 转概率

mask01 = (pred > threshold_unet).float().cpu().numpy().astype(np.uint8)  # 二值mask
mask255 = mask01 * 255  # 用于Sobel

# ===============================
# 2️⃣ 火焰比例
# ===============================
fire_ratio_val = mask01.sum() / mask01.size  # 火焰像素比例
fire_ratio = torch.tensor([[fire_ratio_val]], dtype=torch.float32).to(device)

# ===============================
# 3️⃣ YOLO（只跑一次）
# ===============================
results = scene_model(image)

cls_ids = []
if results and results[0].boxes is not None:
    cls_ids = results[0].boxes.cls.cpu().numpy()

scene_vec = scene_size_classification(cls_ids).unsqueeze(0).to(device)

# ===============================
# 4️⃣ trunk mask
# ===============================
y1 = get_trunk_top_y_from_results(results)

mask01_trunk = mask01.copy()

if y1 is not None:
    h, w = mask01.shape
    scale = h / image.height  # 坐标映射
    y1 = int(y1 * scale)
    y1 = max(0, min(h, y1))

    mask01_trunk[y1:, :] = 0  # 下方全部裁掉

mask01_trunk = mask01_trunk.astype(np.float32)

# ===============================
# 5️⃣ Sobel边缘
# ===============================
img_mask_float = mask01_trunk.astype(np.float32)

sobelx = cv2.Sobel(img_mask_float, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_mask_float, cv2.CV_64F, 0, 1, ksize=3)
sobelmag = np.sqrt(sobelx**2 + sobely**2)

# plt.figure(figsize=(15, 5))
#
# # sobelx
# plt.subplot(1, 3, 1)
# plt.title("Sobel X")
# plt.imshow(sobelx, cmap='gray')
# plt.axis('off')
#
# # sobely
# plt.subplot(1, 3, 2)
# plt.title("Sobel Y")
# plt.imshow(sobely, cmap='gray')
# plt.axis('off')
#
# # sobel magnitude
# plt.subplot(1, 3, 3)
# plt.title("Sobel Magnitude")
# plt.imshow(sobelmag, cmap='gray')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

# 二值化
sobelx = (np.abs(sobelx) > 0).astype(np.float32)
sobely = (np.abs(sobely) > 0).astype(np.float32)
sobelmag = (sobelmag > 0).astype(np.float32)

# ===============================
# 6️⃣ 构造CNN输入
# ===============================
mask01 = mask01.astype(np.float32)

visual_input = np.stack([
    mask01
    # mask01_trunk,
    # sobelx,
    # sobely,
    # sobelmag
], axis=0)  # (5,640,640)

visual_input = torch.tensor(visual_input).unsqueeze(0).to(device)

# ===============================
# 7️⃣ 推理
# ===============================
with torch.no_grad():
    logits, _, _, _ = model(visual_input, scene_vec, fire_ratio)
    pred_class = torch.argmax(logits, dim=1)

# ===============================
# 结束计时
# ===============================
if device.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

# ===============================
# 输出结果
# ===============================
inference_time_ms = (end - start) * 1000  # 转毫秒
fps = 1000 / inference_time_ms            # 帧率

print("预测类别:", pred_class.item())
print("推理时间: {:.2f} ms".format(inference_time_ms))
print("FPS: {:.2f}".format(fps))

