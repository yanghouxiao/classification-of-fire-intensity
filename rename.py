import os

# 图片文件夹路径
folder_path = r"F:\UE\fire_text\froest\Saved\MovieRenders"

# 获取文件夹内所有文件
files = os.listdir(folder_path)

# 只保留图片文件
image_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
images = [f for f in files if f.lower().endswith(image_ext)]

# 排序（保证顺序一致）
images.sort()

# 重新命名
for i, filename in enumerate(images, start=1):
    old_path = os.path.join(folder_path, filename)

    # 新文件名（4位编号）
    new_name = f"fire_{i:04d}.jpg"

    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)

print("图片重命名完成！")