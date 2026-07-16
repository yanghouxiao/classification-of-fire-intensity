import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 修复OpenMP冲突
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11s.yaml')  # 该参数填入模型结构文件（m,n,s,l,x选择不同的话表示不同的模型权重大小）
    model.train(data='scene_size_classification_yolov11_data.yaml',  # 该参数可以填入训练数据集配置文件的路径
                imgsz=640,  # 该参数代表输入图像的尺寸，指定为 640x640 像素
                epochs=100,  # 该参数代表训练的轮数
                batch=10,  # 该参数代表批处理大小，电脑显存越大，就设置越大，根据自己电脑性能设置（一次处理的图像个数）
                workers=0,  # 该参数代表数据加载的工作线程数，出现显存爆了的话可以设置为0，默认是8
                device='0',  # 该参数代表用哪个显卡训练，留空表示自动选择可用的GPU或CPU
                optimizer='SGD',  # 该参数代表优化器类型
                close_mosaic=10,  # 该参数代表在多少个 epoch 后关闭 mosaic 数据增强
                resume=False,  # 该参数代表是否从上一次中断的训练状态继续训练。设置为False表示从头开始新的训练。如果设置为True，则会加载上一次训练的模型权重和优化器状态，继续训练。这在训练被中断或在已有模型的基础上进行进一步训练时非常有用。
                project='runs/train',  # 该参数代表项目文件夹，用于保存训练结果
                name='exp',  # 该参数代表命名保存的结果文件夹
                single_cls=False,  # 该参数代表是否将所有类别视为一个类别，设置为False表示保留原有类别
                cache=False,  # 该参数代表是否缓存数据，设置为False表示不缓存
                )