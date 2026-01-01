import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_cfg = 'simd'
    weights_cfg = '/simd_15c_32_4'
    testdata_cfg='simd'

    weights_path = './runs'+weights_cfg+'/weights/best.pt'
    yolomodel_path = './ultralytics/cfg/models/v8/yolov8n_' + model_cfg + '.yaml'
    testdata_path = './ultralytics/cfg/datasets/' + testdata_cfg + '.yaml'
    model = YOLO(yolomodel_path, task='detect')
    model.load(weights_path)
    model.val(data=testdata_path,
              split='test',
              imgsz=1024,
              batch=1,
              save=True,
              conf=0.001,
              iou=0.5,
              line_width = 1,
              name=model_cfg+'_'+testdata_cfg + '_' + 'test',
              # save_txt = True, save_conf = True, save_json = False, show_labels = True, show_conf = True
              )


