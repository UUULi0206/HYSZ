import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os

# 解决 OpenMP 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if __name__ == '__main__':
    model = YOLO(r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\weights\best.pt')

    # 验证模型
    results = model.val(
        data=r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\data.yaml',
        split='val',
        imgsz=640,
        batch=32,
        use_simotm="RGBT",
        channels=4,
        project='runs/val/YAN',
        name='YAN-VAL-yolo11',
        save_txt=True  # 保存结果到txt文件
    )

    # 如果需要保存验证数据集路径到txt文件
    validation_data_path = r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\data.yaml'
    output_file = "validation_dataset.txt"
    with open(output_file, "w") as f:
        f.write("Validation Dataset Path:\n")
        f.write(f"{validation_data_path}\n")

    print(f"Validation dataset path saved to {output_file}")