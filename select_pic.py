import os
import random
from PIL import Image

# 原始文件夹
src_dir = r"F:\UE\fire\froestfire_dataset\midscene_bigfire"

# 输出文件夹
dst_dir = r"F:\yangnet_dataset\forest_fire_size_classification_dataset_detail\train\midscene_bigfire"

# 从路径中提取“文件夹名字”
folder_name = os.path.basename(dst_dir)

# 选取数量
num_samples = 625

# 图片格式
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# 获取所有图片
all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(img_exts)]

# 检查数量
if len(all_images) < num_samples:
    raise ValueError(f"图片数量不足！当前只有 {len(all_images)} 张")

# 不重复随机抽样
selected_images = random.sample(all_images, num_samples)

# 创建输出目录
os.makedirs(dst_dir, exist_ok=True)

# 处理图片
for i, img_name in enumerate(selected_images):
    src_path = os.path.join(src_dir, img_name)

    # 新文件名 = 文件夹名 + 编号
    new_name = f"{folder_name}_{i:04d}.jpg"
    dst_path = os.path.join(dst_dir, new_name)

    try:
        img = Image.open(src_path).convert("RGB")

        # resize 到 640×640
        img = img.resize((640, 640), Image.BILINEAR)

        # 保存
        img.save(dst_path, quality=95)

    except Exception as e:
        print(f"处理失败: {img_name}, 错误: {e}")

print(f"完成！已生成 {num_samples} 张图片到：{dst_dir}")