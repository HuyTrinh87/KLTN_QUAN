
import glob, os
from ultralytics import YOLO
cands = glob.glob('/kaggle/working/output_mamba/**/weights/best.pt', recursive=True)
if not cands:
    print('Khong tim thay best.pt'); raise SystemExit
best = max(cands, key=os.path.getmtime)
print('Best weights:', best)
mt = YOLO(best).val(data='/kaggle/working/dataset.yaml', split='test', imgsz=640)
print(f'>>> Test mAP50-95: {mt.box.map:.4f} | mAP50: {mt.box.map50:.4f} | mAP75: {mt.box.map75:.4f}')
