# 📡 Telecom Antenna Detection & Azimuth Estimation System
> **Khóa Luận Tốt Nghiệp** — MSSV: `22200130` — Họ và tên: **Huỳnh Nguyễn Quân**  
> **Chuyên ngành:** Điện tử Viễn thông — Trắc nghiệm & Xử lý Thị giác Máy tính trong Hạ tầng Viễn thông

---

## 📌 Giới Thiệu Dự Án (Project Overview)

Dự án nghiên cứu và phát triển giải pháp thị giác máy tính (Computer Vision) phục vụ công tác quản lý hạ tầng viễn thông, hỗ trợ ước lượng tự động **Góc Phương Vị (Azimuth)** và góc nghiêng (**Tilt**) của các thiết bị ăng-ten (`anten-4G`, `anten-5G`, `rrh`, `rru`, `viba`) từ ảnh chụp hiện trường.

### ✨ Các Tính Năng Nổi Bật
1. **Object Detection (Nhận diện thiết bị):** Sử dụng các mô hình học sâu hiện đại như **YOLOv26** (Nano, Small, Mid, Large, XLarge) và **Mamba-YOLO** để phát hiện và phân loại chính xác các loại ăng-ten và khối RRH/RRU.
2. **3D Keypoint Pose Estimation:** Dự đoán 4 điểm góc keypoint 2D trên mặt bức xạ ăng-ten bằng mô hình **YOLO Pose**.
3. **Thuật toán PnP & LM Refinement:**
   - Ước lượng hướng pháp tuyến mặt bức xạ ăng-ten dựa trên giải thuật **IPPE** (`cv2.solvePnPGeneric`).
   - Tinh chỉnh non-linear Levenberg-Marquardt (`cv2.solvePnPRefineLM`) giảm thiểu sai số chiếu lại (reprojection RMS).
   - Cơ chế bảo vệ nghiêng **Tilt-Sanity Guard** (lọc góc tilt bất hợp lý trong khoảng $-12^\circ \le \text{tilt} \le 88^\circ$).
4. **Ghép nối Tối ưu Hungarian Matching:** Ghép cặp 1-1 tối ưu toàn cục giữa Bounding Box và Keypoint bằng giải thuật Hungarian (`scipy.optimize.linear_sum_assignment`) kết hợp kiểm tra tâm spatial center.
5. **Tính toán Phương vị Thực:** $\text{true\\_az} = (\text{heading} - \text{az\\_rel} + \text{AZIMUTH\\_OFFSET}) \pmod{360}$

---

## 📁 Cấu Trúc Thư Mục (Project Structure)

Dự án được cấu trúc theo chuẩn sản xuất **ML Project in Production**:

```text
.
├── config/                                 # Cấu hình tham số mô hình & huấn luyện
│   ├── args_yolo26l.yaml
│   └── args_yolo26m.yaml
├── data/                                   # Quản lý dữ liệu dự án
│   ├── 01-raw/                             # Ảnh gốc chụp hiện trường (90.jpg, 180.jpg, 270.jpg,...)
│   └── 04-predictions/                     # Kết quả ước lượng phương vị & báo cáo (CSV, PNG)
├── docs/                                   # Tài liệu báo cáo luận văn (KLTN_22200130_QUAN_FINAL.docx)
├── entrypoint/                             # Điểm thực thi ứng dụng
│   └── inference.py                        # Script thực thi suy luận trực tiếp từ dòng lệnh
├── notebooks/                              # Jupyter Notebooks nghiên cứu & thử nghiệm
│   ├── preprocessing/                      # Tiền xử lý dữ liệu & Augmentation
│   │   ├── preprocess_detection_dataset.ipynb
│   │   └── preprocess_pose_dataset.ipynb
│   ├── training/                           # Huấn luyện mô hình Deep Learning
│   │   ├── detection/                      # Notebooks train YOLOv26 & Mamba-YOLO
│   │   │   ├── train_mambayolo_base.ipynb
│   │   │   ├── train_mambayolo_initial.ipynb
│   │   │   ├── train_yolo26n_nano.ipynb
│   │   │   ├── train_yolo26s_small.ipynb
│   │   │   ├── train_yolo26m_medium.ipynb
│   │   │   ├── train_yolo26l_large.ipynb
│   │   │   └── train_yolo26x_xlarge.ipynb
│   │   └── pose/                           # Notebooks train YOLO Pose Keypoints
│   │       ├── train_yolo26m_pose.ipynb
│   │       └── train_yolo26l_pose.ipynb
│   └── inference/                          # Notebook thử nghiệm suy luận phương vị
│       └── azimuth_inference.ipynb
├── src/                                    # Mã nguồn chính (Modules & Pipelines)
│   └── pipelines/
│       ├── feature_eng_pipeline.py         # Pipeline chia tập dữ liệu & tạo dataset.yaml
│       └── inference_pipeline.py           # Core logic: PnP (IPPE + LM), Hungarian, Azimuth Engine
├── .gitignore                              # Cấu hình bỏ qua file rác, cache & weights (.pt, .zip)
├── .gitlab-ci.yml                          # Cấu hình CI/CD cho GitLab
├── docker-compose.yml                      # Cấu hình Docker Compose cho GPU container
├── Dockerfile                              # Đóng gói môi trường Docker
├── env.yaml                                # Cấu hình môi trường Conda (Production)
├── env-dev.yaml                            # Cấu hình môi trường Conda (Development)
├── Makefile                                # Phím tắt lệnh quản lý dự án (install, clean, inference)
├── README.md                               # Tài liệu hướng dẫn sử dụng dự án
└── requirements.txt                        # Danh sách thư viện phụ thuộc
```

---

## 🛠 Yêu Cầu Môi Trường (Requirements)

* **Python:** 3.9+
* **Thư viện chính:**
  - `torch` & `torchvision` (hỗ trợ CUDA nếu dùng GPU)
  - `ultralytics` (YOLOv8 / YOLOv26)
  - `opencv-python` (`cv2`)
  - `scipy`
  - `numpy`
  - `Pillow`

### Cài Đặt Thư viện
```bash
pip install ultralytics opencv-python scipy numpy pillow torch torchvision
```

---

## 🚀 Hướng Dẫn Sử Dụng (Quick Start)

### 1. Phân chia dữ liệu Pose (Feature Engineering Pipeline)
Để phân chia dữ liệu hình ảnh thành 80% train / 10% val / 10% test và xuất file `dataset.yaml`:
```bash
python src/pipelines/feature_eng_pipeline.py
```

### 2. Thực thi Ước lượng Góc Phương Vị (Inference Pipeline)
Chạy suy luận trực tiếp trên thư mục ảnh bằng entrypoint:
```bash
python entrypoint/inference.py
```

Hoặc gọi trực tiếp từ Python code:
```python
from src.pipelines.inference_pipeline import run_folder

FOLDER = 'data/01-raw'
POSE_WEIGHTS = 'path/to/pose_model_best.pt'
DET_WEIGHTS = 'path/to/detection_model_best.pt'
HEADINGS = {'90.jpg': 90.0, '180.jpg': 180.0, '270.jpg': 270.0}

results = run_folder(
    folder=FOLDER,
    pose_weights=POSE_WEIGHTS,
    det_weights=DET_WEIGHTS,
    headings=HEADINGS,
    azimuth_offset=0.0
)
```

### 3. Huấn luyện mô hình với Notebooks
Mở các notebook trong thư mục `notebooks/training/` thông qua Jupyter Lab hoặc Google Colab / Kaggle để huấn luyện lại các phiên bản mô hình YOLO Detection & Pose.

---

## 📊 Kết Quả Đánh Giá (Evaluation Criteria)

* **Tiêu chí nghiệm thu:** Ăng-ten đạt chuẩn nếu độ lệch phương vị tính toán không vượt quá $15^\circ$ so với giá trị đo thực tế (Ground Truth).
* **Độ ổn định:** Mô hình được đánh giá thông qua chỉ số RMS Reprojection Error và sai số quét nhãn bằng Monte Carlo Uncertainty (`mc_uncertainty`).

---

## 📄 Bản Quyền & Tác Giả (Author)

* **Tác giả:** Huỳnh Nguyễn Quân (MSSV: 22200130)
* **Trường:** Đại học Khoa học Tự nhiên - ĐHQG TP.HCM
* **Năm hoàn thành:** 2026
