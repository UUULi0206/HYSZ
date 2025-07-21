import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os

# 解决 OpenMP 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# YOLO命令行示例（Ultralytics版本）

if __name__ == '__main__':
    model = YOLO(R'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\weights\best.pt')
    model.val(data=r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\data.yaml',
              split='val',
              imgsz=640,
              batch=32,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              val_interval=5,
              project='runs/val/YAN',
              name='YAN-VAL-yolo11',
              )