
from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("/kaggle/working/output_mamba/mambayolo_base/weights/last.pt")
    model.train(resume=True)
