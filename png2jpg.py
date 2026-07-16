from PIL import Image
import os


def convert_png_to_jpg_in_directory(directory_path, quality=95):
    """
    将指定目录中的所有PNG图像转换为JPG格式

    参数:
        directory_path (str): 包含PNG图像的目录路径
        quality (int): JPG质量，1-100，默认95
    """
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在")
        return

    # 统计变量
    converted_count = 0
    error_count = 0

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.png'):
            # 构建完整文件路径
            png_path = os.path.join(directory_path, filename)

            # 生成JPG文件名（保持原文件名，只改扩展名）
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(directory_path, jpg_filename)

            try:
                # 打开PNG图像并转换
                with Image.open(png_path) as img:
                    # 转换为RGB模式（移除Alpha通道）
                    rgb_img = img.convert('RGB')

                    # 保存为JPG
                    rgb_img.save(jpg_path, 'JPEG', quality=quality)

                print(f"✓ 转换成功: {filename} -> {jpg_filename}")
                converted_count += 1

            except Exception as e:
                print(f"✗ 转换失败 {filename}: {str(e)}")
                error_count += 1

    # 输出总结
    print(f"\n转换完成!")
    print(f"成功: {converted_count} 个文件")
    print(f"失败: {error_count} 个文件")


# 使用示例
if __name__ == "__main__":
    # 指定您的目录路径
    image_directory = r"C:\Users\yanghouxiao\Desktop\F20"

    # 执行转换
    convert_png_to_jpg_in_directory(image_directory)

    # 等待用户按键后退出（防止窗口立即关闭）
    input("\n按Enter键退出...")