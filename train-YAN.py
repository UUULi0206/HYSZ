import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os

# 解决 OpenMP 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if __name__ == '__main__':
    model = YOLO(R'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\ultralytics\cfg\models\11\yolo11.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=5,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='YAN',
                name='YAN-yolo11',
                )