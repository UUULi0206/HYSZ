import warnings
import shutil
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings('ignore')


def predict_and_classify():
    # 初始化模型
    model = YOLO(r"D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\weights\best.pt")

    # 原始数据路径
    source_dir = r"D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\dataset\images\visible\val"

    # 统一推理并保存结果到临时目录
    temp_exp = Path("runs/detect/exp_temp")  # 临时推理目录
    results = model.predict(
        source=source_dir,
        imgsz=640,
        project=str(temp_exp.parent),
        name=temp_exp.name,
        show=False,
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True, # 允许覆盖已有结果
        channels = 3,  # 关键修改！强制使用3通道输入
        use_simotm = None  # 关闭特殊通道处理
    )

    # 创建最终分类目录
    output_base = Path("classified_results")
    class_dirs = {
        "part1": {
            "images": output_base / "part1/images",
            "labels": output_base / "part1/labels"
        },
        "part2": {
            "images": output_base / "part2/images",
            "labels": output_base / "part2/labels"
        }
    }
    for cls in class_dirs.values():
        cls["images"].mkdir(parents=True, exist_ok=True)
        cls["labels"].mkdir(parents=True, exist_ok=True)

    # 分类处理结果
    processed = {"part1": 0, "part2": 0}
    for result in results:
        # 获取原始文件名
        src_img_path = Path(result.path)
        src_txt_path = temp_exp / "labels" / f"{src_img_path.stem}.txt"

        # 确定分类
        filename = src_img_path.name.lower()
        if "part1" in filename:
            cls = "part1"
        elif "part2" in filename:
            cls = "part2"
        else:
            print(f"跳过未分类文件: {src_img_path.name}")
            continue

        # 复制图像和标签
        dest_img = class_dirs[cls]["images"] / src_img_path.name
        dest_txt = class_dirs[cls]["labels"] / src_txt_path.name

        shutil.copy(src_img_path, dest_img)
        if src_txt_path.exists():
            shutil.copy(src_txt_path, dest_txt)
            processed[cls] += 1

    # 打印统计信息
    print(f"\n处理完成！分类结果：")
    print(f"part1: {processed['part1']} 个样本")
    print(f"part2: {processed['part2']} 个样本")


if __name__ == '__main__':
    predict_and_classify()