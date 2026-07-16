import os
import glob
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ===============================
# YOLO模型加载
# ===============================
pt_scene_size = "pt_yangnet/scene_size_pt/best.pt"
scene_model = YOLO(pt_scene_size)

scene_model.model.eval()  # 推理模式

# ===============================
# 三维向量函数
# ===============================
def scene_size_classification(cls_ids, sky_id=0, trunk_id=1):
    has_sky, has_trunk = False, False

    for cid in cls_ids:
        if int(cid) == sky_id:
            has_sky = True
        elif int(cid) == trunk_id:
            has_trunk = True

    if has_trunk:
        return np.array([1, 0, 0], dtype=np.float32)  # 近景
    if has_sky:
        return np.array([0, 0, 1], dtype=np.float32)  # 远景
    return np.array([0, 1, 0], dtype=np.float32)      # 中景

# ===============================
# 数据集路径
# ===============================
dataset_root = "yangnet/forest_fire_size_classification_dataset_detail_notrunkcut/train"
classes = os.listdir(dataset_root)

# ===============================
# 开始生成
# ===============================
for cls_name in classes:
    cls_dir = os.path.join(dataset_root, cls_name)

    if not os.path.isdir(cls_dir):
        continue

    print(f"\nProcessing {cls_name}")

    # 创建保存目录
    save_dir = os.path.join(cls_dir, "scene_vec")
    os.makedirs(save_dir, exist_ok=True)

    # 找到所有 mask01
    mask_paths = glob.glob(os.path.join(cls_dir, "*mask01.png"))

    for mask_path in tqdm(mask_paths):

        # -------------------------------
        # 1️⃣ 提取名字
        # -------------------------------
        name = os.path.basename(mask_path).replace("_mask01.png", "")

        # -------------------------------
        # 2️⃣ 构造原图路径（🔥关键修改）
        # -------------------------------
        img_path = os.path.join(cls_dir, f"{name}.jpg")

        # 如果原图不存在，跳过（防止报错）
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            continue

        # -------------------------------
        # 3️⃣ YOLO 推理
        # -------------------------------
        results = scene_model(img_path, verbose=False)

        if results[0].boxes is not None:
            cls_ids = results[0].boxes.cls.cpu().numpy()
        else:
            cls_ids = []

        # -------------------------------
        # 4️⃣ 转三维向量
        # -------------------------------
        scene_vec = scene_size_classification(cls_ids)

        # -------------------------------
        # 5️⃣ 保存
        # -------------------------------
        save_path = os.path.join(save_dir, f"{name}.npy")
        np.save(save_path, scene_vec)