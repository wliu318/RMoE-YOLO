import warnings
import os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    projname = 'simd'
    weightspath = './runs/weights/yolov8n.pt'
    yolomodelpath='./ultralytics/cfg/models/v8/yolov8n_simd_airplane.yaml'
    datapath = './ultralytics/cfg/datasets/simd_airplane.yaml'
    model = YOLO(yolomodelpath, task='detect')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    model.load(weightspath)      #加载预训练的权重文件'yolov8l.pt'，加速训练并提升模型性能
    model.train(data=datapath,  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
                cache=False,  # 是否缓存数据集以加快后续训练速度，False表示不缓存
                imgsz=1024,  # 指定训练时使用的图像尺寸，640表示将输入图像调整为640x640像素
                epochs=400,  # 设置训练的总轮数为200轮
                batch=32,  # 设置每个训练批次的大小为16，即每次更新模型时使用16张图片
                close_mosaic=10,  # 设置在训练结束前多少轮关闭 Mosaic 数据增强，10 表示在训练的最后 10 轮中关闭 Mosaic
                workers=2,  # 设置用于数据加载的线程数为8，更多线程可以加快数据加载速度
                patience=100,  # 在训练时，如果经过50轮性能没有提升，则停止训练（早停机制）
                device='0',  # 指定使用的设备，'0'表示使用第一块GPU进行训练
                optimizer='SGD',  # 设置优化器为SGD（随机梯度下降），用于模型参数更新
                degrees= 360.0, # (float) image rotation (+/- deg),在指定的度数范围内随机旋转图像，提高模型识别不同方向对象的能力。选择旋转角度后，在选择的旋转角度范围内进行数据随机旋转。
                scale=0.5,  # (float) image scale (+/- gain),随机缩放，范围为 ±50%
                flipud=0.5,  # (float) image flip up-down (probability),50% 概率进行垂直翻转
                fliplr=0.5,  # (float) image flip left-right (probability), 50% 概率进行水平翻转
                auto_augment='randaugment',  # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix),面向分类任务，自动应用预定义的增强策略（randaugment、autoaugment和augmix），通过使视觉特征多样化来优化分类任务。
                                              # 默认数值为randaugment，范围是（randaugment、autoaugment和augmix）。
                resume=False,
                # ckpt_path='last.pt',
                name=projname + '_' + 'train',
                # freeze=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22],  #冻结前10层，仅训练neck和head
                )
    model.val(data=datapath,  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
              split='test',
              imgsz=1024,
              batch=1,
              conf=0.001,
              iou=0.5,
              name=projname+'_'+'test',
              )