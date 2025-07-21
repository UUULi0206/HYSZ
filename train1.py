import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\ultralytics\cfg\models\11\yolo11.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'yolo11.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=4,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='BCCD',
                name='BCCD-yolo12n-PGI',
                )