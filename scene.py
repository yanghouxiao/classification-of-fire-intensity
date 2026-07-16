import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings('ignore')

import cv2
import time
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

confidence_threshold = 0.5
model_path1 = 'runs/train/exp4/weights/best.pt'

# 设置matplotlib支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'sans-serif'
except:
    print("警告: 中文字体设置失败，可能会显示乱码")


def classify_scene(detected_classes):
    """
    根据检测到的类别进行场景分类
    规则：
    - 如果只检测到0(sky) -> big_scene
    - 如果只检测到1(trunk)或同时检测到0和1 -> small_scene
    - 其他情况 -> mid_scene
    """
    detected_classes = set(detected_classes)  # 去重

    if detected_classes == {0}:  # 只检测到sky
        return "big_scene"
    elif detected_classes == {1} or detected_classes == {0, 1}:  # 只检测到trunk或两者都检测到
        return "small_scene"
    else:  # 其他情况（包括未检测到任何目标）
        return "mid_scene"


def predict_single_image(model, image_path, conf_threshold=confidence_threshold):
    """
    预测单张图片并返回检测结果和场景分类
    """
    # 进行预测，设置置信度阈值为0.5
    results = model.predict(
        source=image_path,
        conf=conf_threshold,  # 只保留置信度大于等于0.5的检测结果
        save=False,
        verbose=False
    )

    # 处理结果
    for result in results:
        # 获取检测信息
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 获取检测到的类别
            detected_classes = boxes.cls.cpu().numpy().astype(int).tolist()
            confidences = boxes.conf.cpu().numpy().tolist()

            # 过滤置信度低于0.5的检测结果
            filtered_classes = []
            filtered_confidences = []

            for cls, conf in zip(detected_classes, confidences):
                if conf >= confidence_threshold:  # 只保留置信度大于等于0.5的结果
                    filtered_classes.append(cls)
                    filtered_confidences.append(conf)

            # 场景分类
            scene_type = classify_scene(filtered_classes)

            # 可视化结果
            annotated_image = result.plot()

            # 确保图像是RGB格式
            if annotated_image.shape[2] == 3:
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            return annotated_image, filtered_classes, scene_type, filtered_confidences
        else:
            # 未检测到任何目标
            scene_type = classify_scene([])

            # 读取原图
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image, [], scene_type, []


def batch_test_folders(model, folder_paths, output_dir="batch_test_results"):
    """
    批量测试多个文件夹中的图片
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 存储结果
    results = []
    example_images = defaultdict(list)  # 存储每个类别的示例图片

    # 遍历每个文件夹
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue

        # 从文件夹路径获取真实类别
        true_class = os.path.basename(folder_path.rstrip('/\\'))
        print(f"\n正在处理类别: {true_class}")

        # 获取文件夹中的所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))

        if not image_files:
            print(f"  在 {folder_path} 中未找到图片文件")
            continue

        print(f"  找到 {len(image_files)} 张图片")

        # 处理每张图片
        for i, image_path in enumerate(image_files):
            print(f"  处理第 {i + 1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")

            try:
                # 进行预测
                annotated_image, detected_classes, predicted_class, confidences = predict_single_image(model,
                                                                                                       image_path)

                # 判断是否正确
                is_correct = (predicted_class == true_class)

                # 存储结果
                result = {
                    'image_path': image_path,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'detected_classes': detected_classes,
                    'confidences': confidences,
                    'is_correct': is_correct
                }
                results.append(result)

                # 存储示例图片（每个类别最多存储4张，包含正确和错误分类）
                if len(example_images[true_class]) < 4:
                    example_images[true_class].append({
                        'image': annotated_image,
                        'true_class': true_class,
                        'predicted_class': predicted_class,
                        'is_correct': is_correct,
                        'detected_classes': detected_classes
                    })

            except Exception as e:
                print(f"  处理图片时出错: {e}")
                continue

    return results, example_images


def calculate_accuracy(results):
    """
    计算每个类别的准确率和总体准确率
    """
    # 按真实类别分组
    class_results = defaultdict(list)
    for result in results:
        class_results[result['true_class']].append(result)

    # 计算每个类别的准确率
    accuracy_report = {}
    for class_name, class_result_list in class_results.items():
        correct_count = sum(1 for r in class_result_list if r['is_correct'])
        total_count = len(class_result_list)
        accuracy = correct_count / total_count if total_count > 0 else 0
        accuracy_report[class_name] = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count
        }

    # 计算总体准确率
    total_correct = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    accuracy_report['overall'] = {
        'accuracy': overall_accuracy,
        'correct_count': total_correct,
        'total_count': total_count
    }

    return accuracy_report


def display_accuracy_report(accuracy_report):
    """
    显示准确率报告
    """
    print("\n" + "=" * 60)
    print("场景分类准确率报告")
    print("=" * 60)

    for class_name, metrics in accuracy_report.items():
        if class_name == 'overall':
            print(f"\n总体准确率: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_count']})")
        else:
            print(f"{class_name}: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_count']})")


def save_accuracy_report(accuracy_report, output_dir):
    """
    保存准确率报告到CSV文件
    """
    report_data = []
    for class_name, metrics in accuracy_report.items():
        report_data.append({
            '类别': class_name,
            '准确率': f"{metrics['accuracy']:.2%}",
            '正确数量': metrics['correct_count'],
            '总数量': metrics['total_count']
        })

    df = pd.DataFrame(report_data)
    csv_path = os.path.join(output_dir, "accuracy_report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n准确率报告已保存到: {csv_path}")


def display_example_images(example_images, output_dir):
    """
    显示示例图片
    """
    for class_name, examples in example_images.items():
        if not examples:
            continue

        print(f"\n显示 {class_name} 类别的示例图片:")

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{class_name} 类别示例图片', fontsize=16)

        for i, example in enumerate(examples):
            if i >= 4:  # 最多显示4张
                break

            row = i // 2
            col = i % 2

            # 显示图片
            axes[row, col].imshow(example['image'])
            axes[row, col].axis('off')

            # 设置标题
            status = "正确" if example['is_correct'] else "错误"
            title = f"预测: {example['predicted_class']} ({status})"
            axes[row, col].set_title(title, fontsize=12)

            # 添加检测类别信息
            detected_text = f"检测类别: {example['detected_classes']}"
            axes[row, col].text(0.5, -0.1, detected_text, transform=axes[row, col].transAxes,
                                ha='center', fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(output_dir, f"{class_name}_examples.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  示例图片已保存到: {save_path}")

        # 显示图片
        plt.show()


def save_detailed_results(results, output_dir):
    """
    保存详细结果到CSV文件
    """
    detailed_data = []
    for result in results:
        detailed_data.append({
            '图片路径': result['image_path'],
            '真实类别': result['true_class'],
            '预测类别': result['predicted_class'],
            '检测类别': str(result['detected_classes']),
            '置信度': str(result['confidences']),
            '是否正确': '是' if result['is_correct'] else '否'
        })

    df = pd.DataFrame(detailed_data)
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: {csv_path}")


def main():
    # 加载训练好的模型
    model_path = model_path1

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请检查模型路径是否正确")
        return

    print("正在加载模型...")
    model = YOLO(model_path)
    print("模型加载完成!")
    print("模型运行设备:", model.device)

    # 定义要测试的文件夹路径
    folder_paths = [
        # r'D:\Python\Text\forest_fire_size_classification_network\ultralytics-main\scene_size\big_scene',
        # r'D:\Python\Text\forest_fire_size_classification_network\ultralytics-main\scene_size\mid_scene',
        # r'D:\Python\Text\forest_fire_size_classification_network\ultralytics-main\scene_size\small_scene'

        r"D:\UE\fire_text\froest\Saved\small_scene",
        r"D:\UE\fire_text\froest\Saved\big_scene"

    ]

    # 输出目录
    # output_dir = "batch_test_results"
    output_dir = "UE_test"

    # 批量测试
    print("开始批量测试...")
    start_time_total = time.time()  # 总计时开始

    results = []
    example_images = defaultdict(list)
    total_images = 0  # 总图片数量

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue

        true_class = os.path.basename(folder_path.rstrip('/\\'))
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if any(f.lower().endswith(ext) for ext in image_extensions)]
        if not image_files:
            print(f"  在 {folder_path} 中未找到图片文件")
            continue

        print(f"\n正在处理类别: {true_class}, 共 {len(image_files)} 张图片")
        total_images += len(image_files)

        for i, image_path in enumerate(image_files):
            print(f"  处理第 {i + 1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
            try:
                annotated_image, detected_classes, predicted_class, confidences = predict_single_image(model, image_path)
                is_correct = (predicted_class == true_class)

                # 存储结果
                result = {
                    'image_path': image_path,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'detected_classes': detected_classes,
                    'confidences': confidences,
                    'is_correct': is_correct
                }
                results.append(result)

                # 存储示例图片
                if len(example_images[true_class]) < 4:
                    example_images[true_class].append({
                        'image': annotated_image,
                        'true_class': true_class,
                        'predicted_class': predicted_class,
                        'is_correct': is_correct,
                        'detected_classes': detected_classes
                    })

            except Exception as e:
                print(f"  处理图片时出错: {e}")
                continue

    end_time_total = time.time()
    total_elapsed = end_time_total - start_time_total
    avg_time_per_image = total_elapsed / total_images if total_images > 0 else 0

    print(f"\n批量测试总耗时: {total_elapsed:.3f} 秒")
    print(f"平均每张图片耗时: {avg_time_per_image:.3f} 秒")  # 平均耗时

    if not results:
        print("没有找到任何可测试的图片")
        return

    # 计算准确率
    accuracy_report = calculate_accuracy(results)

    # 显示准确率报告
    display_accuracy_report(accuracy_report)

    # 保存准确率报告
    save_accuracy_report(accuracy_report, output_dir)

    # 保存详细结果
    save_detailed_results(results, output_dir)

    # 显示示例图片
    display_example_images(example_images, output_dir)

    print(f"\n所有结果已保存到: {output_dir} 文件夹")




if __name__ == "__main__":
    main()

