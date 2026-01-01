import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/v8/yolov8n_rsod.yaml', task='detect')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    model.load('./runs/detect/train21/weights/best.pt')      #加载预训练的权重文件'yolov8s.pt'，加速训练并提升模型性能
    model.val(data='./ultralytics/cfg/datasets/rsod.yaml',  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
              split='test',
              imgsz=1024,
              batch=1,
              conf=0.001,
              iou=0.5,
              # name='nwpu_test'
              )
