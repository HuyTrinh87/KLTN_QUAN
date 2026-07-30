
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/kaggle/working/Mamba-YOLO/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml")
    model.train(
        data="/kaggle/working/dataset.yaml",
        imgsz=640, device=[0, 1], batch=8, workers=8, amp=True,
        epochs=150, patience=25,
        optimizer="SGD", lr0=0.01,          # SGD = chuan repo Mamba (thay cho MuSGD doc quyen YOLO26)
        cos_lr=True, close_mosaic=10, seed=42, deterministic=True,
        project="/kaggle/working/output_mamba", name="mambayolo_base", save_period=10,
    )
