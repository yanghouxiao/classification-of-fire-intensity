import json
import numpy as np
from PIL import Image, ImageDraw
import os
import glob
import traceback


def find_image_file(json_path, data):
    """
    查找与JSON文件对应的图像文件
    """
    # 方法1: 使用JSON中记录的imagePath
    json_dir = os.path.dirname(json_path)
    image_filename = data['imagePath']
    image_path = os.path.join(json_dir, image_filename)

    if os.path.exists(image_path):
        return image_path

    # 方法2: 尝试在JSON同目录下查找同名但不同扩展名的文件
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']

    for ext in possible_extensions:
        possible_path = os.path.join(json_dir, json_basename + ext)
        if os.path.exists(possible_path):
            return possible_path

    # 方法3: 尝试在JSON同目录下查找任何包含相同基础名的图像文件
    for file in os.listdir(json_dir):
        if json_basename in file and any(
                file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
            return os.path.join(json_dir, file)

    return None


def force_resize_image(image, target_size=(3840, 2160)):
    """
    强制将图像调整为指定尺寸，不保持宽高比
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def resize_and_convert_json_to_png(json_path, output_mask_dir, output_image_dir, target_size=(3840, 2160)):
    """
    将JSON文件转换为Mask，并将原图和Mask都强制调整为3840×2160
    """
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 查找原图文件
        image_path = find_image_file(json_path, data)

        if image_path is None:
            print(f"❌ 找不到原图文件: {json_path}")
            return False

        # 读取原图
        try:
            original_image = Image.open(image_path)
            print(f"📄 处理: {os.path.basename(json_path)}")
            print(f"  原图尺寸: {original_image.size} -> 目标尺寸: {target_size}")
        except Exception as e:
            print(f"❌ 无法读取原图 {image_path}: {str(e)}")
            return False

        # 强制调整原图大小为3840×2160（不保持宽高比）
        resized_image = force_resize_image(original_image, target_size)

        # 保存调整后的原图
        image_output_filename = os.path.basename(image_path)
        # 确保文件扩展名正确
        name, ext = os.path.splitext(image_output_filename)
        if not ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_output_filename = name + '.png'

        image_output_path = os.path.join(output_image_dir, image_output_filename)
        resized_image.save(image_output_path, quality=95)

        # 创建Mask（使用JSON中的原始尺寸）
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        # 创建全零数组（背景为0）
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # 创建PIL图像用于绘制
        pil_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil_mask)

        # 遍历所有标注形状
        fire_found = False
        for shape in data['shapes']:
            label = shape['label'].lower()

            # 只处理fire标签
            if 'fire' in label:
                fire_found = True
                points = shape['points']
                # 将点转换为整数元组
                polygon_points = [(int(x), int(y)) for x, y in points]
                # 绘制填充的多边形，值为1
                draw.polygon(polygon_points, fill=1)

        if not fire_found:
            print(f"⚠️  警告: {os.path.basename(json_path)} 中没有找到fire标签")

        # 转换回numpy数组
        mask = np.array(pil_mask)

        # 强制调整Mask大小为3840×2160（使用最近邻插值保持像素值不变）
        resized_mask = pil_mask.resize(target_size, Image.Resampling.NEAREST)

        # 保存调整后的Mask
        mask_filename = os.path.basename(json_path).replace('.json', '.png')
        mask_output_path = os.path.join(output_mask_dir, mask_filename)
        resized_mask.save(mask_output_path)

        # 验证输出尺寸
        output_img = Image.open(image_output_path)
        output_mask = Image.open(mask_output_path)

        print(f"✅ 处理成功: {os.path.basename(json_path)}")
        print(f"  原图: {original_image.size} -> {output_img.size}")
        print(f"  Mask: {pil_mask.size} -> {output_mask.size}")
        return True

    except Exception as e:
        print(f"❌ 处理失败 {json_path}: {str(e)}")
        print(f"   详细错误: {traceback.format_exc()}")
        return False


def clear_output_directories():
    """
    清空输出目录，确保没有旧文件
    """
    output_mask_dir = r"C:\Users\yanghouxiao\Desktop\mask"
    output_image_dir = r"C:\Users\yanghouxiao\Desktop\image"

    # 清空mask目录
    if os.path.exists(output_mask_dir):
        for file in glob.glob(os.path.join(output_mask_dir, "*")):
            try:
                os.remove(file)
                print(f"🗑️  删除旧文件: {file}")
            except Exception as e:
                print(f"❌ 删除失败 {file}: {e}")

    # 清空image目录
    if os.path.exists(output_image_dir):
        for file in glob.glob(os.path.join(output_image_dir, "*")):
            try:
                os.remove(file)
                print(f"🗑️  删除旧文件: {file}")
            except Exception as e:
                print(f"❌ 删除失败 {file}: {e}")


def batch_process_all():
    """
    批量处理所有JSON文件，调整原图和Mask大小
    """
    # 输入目录 - JSON文件位置
    input_dir = r"C:\Users\yanghouxiao\Desktop\unet_picture"
    # 输出目录 - Mask保存位置
    output_mask_dir = r"C:\Users\yanghouxiao\Desktop\mask"
    # 输出目录 - 原图保存位置
    output_image_dir = r"C:\Users\yanghouxiao\Desktop\image"

    # 目标尺寸 3840×2160
    target_size = (3840, 2160)

    # 清空输出目录
    print("🗑️  清空输出目录中的旧文件...")
    clear_output_directories()

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
        print(f"📁 创建Mask输出目录: {output_mask_dir}")

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        print(f"📁 创建原图输出目录: {output_image_dir}")

    # 查找所有JSON文件
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"❌ 在目录 {input_dir} 中没有找到JSON文件")
        return

    print(f"📂 找到 {len(json_files)} 个JSON文件，开始处理...")
    print(f"🎯 目标尺寸: {target_size[0]}×{target_size[1]}")
    print("-" * 60)

    success_count = 0
    failed_files = []

    # 逐个处理文件
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] ", end="")
        if resize_and_convert_json_to_png(json_path, output_mask_dir, output_image_dir, target_size):
            success_count += 1
        else:
            failed_files.append(json_path)

    print(f"\n" + "=" * 60)
    print(f"📊 处理完成！成功: {success_count}/{len(json_files)} 个文件")
    print(f"📁 原图保存到: {output_image_dir}")
    print(f"📁 Mask保存到: {output_mask_dir}")

    if failed_files:
        print(f"\n❌ 以下 {len(failed_files)} 个文件处理失败:")
        for failed_file in failed_files:
            print(f"   - {os.path.basename(failed_file)}")


def verify_all_output_sizes():
    """
    验证所有输出的原图和Mask尺寸是否为3840×2160
    """
    output_mask_dir = r"C:\Users\yanghouxiao\Desktop\mask"
    output_image_dir = r"C:\Users\yanghouxiao\Desktop\image"

    target_size = (3840, 2160)

    print("\n🔍 验证所有输出文件尺寸...")

    # 检查Mask文件
    mask_files = glob.glob(os.path.join(output_mask_dir, "*.png"))
    mask_errors = []

    if mask_files:
        print(f"\n📊 检查 {len(mask_files)} 个Mask文件:")
        for mask_path in mask_files:
            try:
                img = Image.open(mask_path)
                if img.size == target_size:
                    print(f"  ✅ {os.path.basename(mask_path)}: {img.size[0]}×{img.size[1]}")
                else:
                    print(f"  ❌ {os.path.basename(mask_path)}: {img.size[0]}×{img.size[1]} (错误!)")
                    mask_errors.append(mask_path)
            except Exception as e:
                print(f"  ❌ {os.path.basename(mask_path)}: 验证失败 - {str(e)}")
                mask_errors.append(mask_path)
    else:
        print("❌ 没有找到Mask文件")

    # 检查原图文件
    image_files = glob.glob(os.path.join(output_image_dir, "*.*"))
    # 过滤常见的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]
    image_errors = []

    if image_files:
        print(f"\n📊 检查 {len(image_files)} 个原图文件:")
        for image_path in image_files:
            try:
                img = Image.open(image_path)
                if img.size == target_size:
                    print(f"  ✅ {os.path.basename(image_path)}: {img.size[0]}×{img.size[1]}")
                else:
                    print(f"  ❌ {os.path.basename(image_path)}: {img.size[0]}×{img.size[1]} (错误!)")
                    image_errors.append(image_path)
            except Exception as e:
                print(f"  ❌ {os.path.basename(image_path)}: 验证失败 - {str(e)}")
                image_errors.append(image_path)
    else:
        print("❌ 没有找到原图文件")

    # 总结报告
    print(f"\n" + "=" * 60)
    print("📋 尺寸验证总结:")
    print(f"🎯 目标尺寸: {target_size[0]}×{target_size[1]}")
    print(f"📁 Mask文件: {len(mask_files) - len(mask_errors)}/{len(mask_files)} 正确")
    print(f"📁 原图文件: {len(image_files) - len(image_errors)}/{len(image_files)} 正确")

    if mask_errors or image_errors:
        print(f"\n❌ 发现 {len(mask_errors) + len(image_errors)} 个尺寸错误的文件:")
        for error in mask_errors + image_errors:
            print(f"  - {os.path.basename(error)}")
        return False
    else:
        print("✅ 所有文件尺寸正确！")
        return True


def check_unprocessed_files():
    """
    检查哪些JSON文件没有被处理
    """
    input_dir = r"C:\Users\yanghouxiao\Desktop\unet_picture"
    output_mask_dir = r"C:\Users\yanghouxiao\Desktop\mask"

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    json_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in json_files)

    # 获取所有Mask文件
    mask_files = glob.glob(os.path.join(output_mask_dir, "*.png"))
    mask_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in mask_files)

    # 找出未处理的JSON文件
    unprocessed = json_basenames - mask_basenames

    if unprocessed:
        print(f"\n⚠️  发现 {len(unprocessed)} 个未处理的JSON文件:")
        for file in unprocessed:
            print(f"  - {file}.json")
    else:
        print("\n✅ 所有JSON文件都已处理!")

    return unprocessed


if __name__ == "__main__":
    print("🚀 开始批量处理JSON文件...")
    print("📂 输入目录: C:\\Users\\yanghouxiao\\Desktop\\unet_picture")
    print("📁 Mask输出: C:\\Users\\yanghouxiao\\Desktop\\mask")
    print("📁 原图输出: C:\\Users\\yanghouxiao\\Desktop\\image")
    print("🎯 目标尺寸: 3840×2160 (强制调整，不保持宽高比)")
    print("=" * 60)

    # 执行批量处理
    batch_process_all()

    print("\n" + "=" * 60)
    # 检查未处理的文件
    unprocessed = check_unprocessed_files()

    print("\n" + "=" * 60)
    # 验证所有输出文件
    all_correct = verify_all_output_sizes()

    if all_correct and not unprocessed:
        print("\n🎉 所有文件处理完成且尺寸正确！")
    else:
        print("\n⚠️  处理完成，但存在一些问题，请检查上述信息。")