import os
import shutil

# ====== 参数设置 ======
input_path = r"F:\UE\fire_text\froest\Saved\MovieRenders"
output_path = r"F:\UE\fire\froestfire_dataset\midscene_midfire"

step = 2  # 每n张取一张（自己改这个n）

# =====================

# 创建输出文件夹（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 获取所有图片文件（按名称排序）
images = sorted([
    f for f in os.listdir(input_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
])

print(f"总图片数量: {len(images)}")

# 按步长选取
selected_images = images[::step]

print(f"选取图片数量: {len(selected_images)}")

# 复制文件
for img_name in selected_images:
    src = os.path.join(input_path, img_name)
    dst = os.path.join(output_path, img_name)

    shutil.copy2(src, dst)

print("处理完成！")