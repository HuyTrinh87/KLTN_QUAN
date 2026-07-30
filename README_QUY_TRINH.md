# Quy trình thực hiện — từ chia dữ liệu → gán nhãn pose → xác định azimuth

Tài liệu này ghi lại **mạch làm việc thật** của đề tài, theo đúng thứ tự đã làm, kèm số liệu và
cả những chỗ phải quay lại sửa. Phần tổng quan hệ thống và kết quả cuối nằm ở [README.md](README.md);
tệp này tập trung vào **cách làm ra kết quả đó**.

Bài toán: từ ảnh khảo sát chụp bằng điện thoại ở chân trạm BTS, trả về **góc phương vị (azimuth)
của từng ăng-ten**, thay cho việc trèo cột đo bằng la bàn.

```
Ảnh khảo sát
   │
   ├─► [A] Chuẩn bị dữ liệu nhận diện   COCO → YOLO, chia tập, tăng cường
   ├─► [B] Huấn luyện & chọn mô hình    6 mô hình nhận diện → YOLO26m
   ├─► [C] Gán nhãn pose                4 điểm góc mặt bức xạ (TL→TR→BR→BL)
   ├─► [D] Huấn luyện & chọn mô hình    YOLO26m-pose / YOLO26l-pose
   └─► [E] Hình học                     solvePnP → pháp tuyến → azimuth
```

---

## Giai đoạn A — Chuẩn bị dữ liệu cho bước nhận diện

**Notebook:** `E:\TIEN_XU_LI_ANH_ANTENNA\Pipeline_Dataset.ipynb`
**Kết quả:** `E:\KAGGLE_UPLOAD\YOLO_Final_Dataset` (6 lớp: `anten-4G`, `anten-5G`, `none`, `rrh`, `rru`, `viba`)

### A.1 Đổi COCO → YOLO

Ảnh được gán nhãn trên Roboflow, xuất ra COCO (một tệp `_annotations.coco.json` cho cả tập).
YOLO cần mỗi ảnh một tệp `.txt`, nên phải chuyển đổi:

| | COCO | YOLO |
|---|---|---|
| Khung bao | `[x_trái, y_trên, rộng, cao]` | `[x_tâm, y_tâm, rộng, cao]` |
| Đơn vị | điểm ảnh tuyệt đối | chuẩn hoá 0–1 theo cỡ ảnh |
| Tổ chức | một JSON cho cả tập | một `.txt` mỗi ảnh |

Bản export của Roboflow luôn chèn thêm một category cha (`Nha-tram`, id 0) không phải lớp thật —
ghép lớp **theo tên** với danh sách `CLASSES` để nó tự bị loại, đồng thời bảo đảm chỉ số lớp khớp
đúng `dataset.yaml`.

### A.2 Chia tập TRƯỚC — 70/20/10 trên ảnh gốc

> ⚠️ **Thứ tự này là bắt buộc.** Chia trước rồi mới tăng cường. Nếu làm ngược lại, ảnh biến thể của
> một ảnh val/test sẽ lọt vào train và thổi phồng kết quả đánh giá.

- Chia trên **2.035 ảnh gốc**, tỉ lệ 70/20/10.
- Lớp `viba` ít mẫu (155 đối tượng) nên được **chia riêng theo đúng tỉ lệ** để không dồn hết về một tập.
- Val/test **chỉ chứa ảnh gốc**.

### A.3 Tăng cường CHỈ trên tập train

Mỗi ảnh train sinh thêm 2 biến thể bằng `albumentations` (Resize 640 + Grayscale; hoặc `SomeOf`:
lật ngang/dọc, xoay 90°, đổi Hue/Saturation, sáng/tương phản, nhiễu Gauss…). Ở bước nhận diện,
**lật và xoay là an toàn** vì khung bao không có thứ tự — khác hẳn bước pose ở Giai đoạn C.

**Số cuối cùng:** train **4.269** ảnh (1.423 gốc + 2.846 biến thể) · val **406** · test **206**.

### A.4 Đo riêng ảnh hưởng của ảnh gần trùng

Cùng một trạm chụp liên tiếp ở góc rất gần nhau, hai ảnh gần như trùng có thể rơi vào hai tập khác
nhau. Lọc bỏ những ảnh này khiến mAP@0.5 của mô hình được chọn **tụt từ 0,847 xuống 0,780** — con
số 0,780 mới là ước lượng sát thực tế cho một trạm hoàn toàn mới.

---

## Giai đoạn B — Huấn luyện & chọn mô hình nhận diện

**Notebook train:** `E:\KAGGLE_UPLOAD\train_yolo26{n,s,m,l,x}_*.ipynb`, `train_mambayolo_base.ipynb`
**Kết quả:** [5_MODEL_MOI/](5_MODEL_MOI/) · đo tốc độ: `5_MODEL_MOI\BEST_6_MODEL\DO_FPS\fps_6_models_T4.csv`

Sáu mô hình, cùng bộ dữ liệu, cùng `imgsz=640`, cùng cơ chế dừng sớm: **YOLO26n/s/m/l/x** và
**Mamba-YOLO-B** làm đối chứng kiến trúc.

### Quy trình chọn — đã phải sửa giữa chừng

Bản đầu chọn mô hình bằng chỉ số trên **tập test** → sai nguyên tắc, tập test đã tham gia vào quyết
định nên không còn độc lập. Bản cuối: **chọn trên tập valid, rồi đưa mô hình đã chốt ra tập test
đúng một lần** để báo cáo.

**Kết quả trên tập test (mAP50 / mAP50-95):**

| n | s | m | l | x |
|---|---|---|---|---|
| 0,772 / 0,508 | 0,804 / 0,546 | **0,847 / 0,585** | 0,820 / 0,569 | 0,835 / 0,586 |

**Thời gian suy luận (ms/ảnh, Tesla T4, đo cả 6 mô hình trong MỘT lần chạy):**
n 3,34 · s 7,87 · **m 19,42** · l 23,84 · x 49,15 · Mamba-YOLO-B 53,00.

> Cột thời gian này cũng phải làm lại: bản đầu ghép số từ nhiều nguồn đo khác nhau nên không so sánh
> được với nhau.

**Chốt YOLO26m** — không phải vì mAP tổng cao nhất, mà vì tách chỉ số theo lớp cho thấy phần hơn của
các mô hình lớn nằm trọn ở các lớp phụ, còn trên **hai lớp ăng-ten** (hai lớp duy nhất đi vào bước
tính góc) thì YOLO26m dẫn đầu, cộng chi phí thấp hơn hẳn.

### Ghi chú kỹ thuật rút ra ở giai đoạn này

- **Ultralytics ≥ 8.4 chọn `best.pt` bằng mAP@0.5:0.95**, không còn công thức cũ
  `0,1·mAP50 + 0,9·mAP50-95`. Đã kiểm: hai công thức cho **cùng một epoch** ở cả 6 mô hình → số liệu
  không đổi, chỉ câu mô tả cơ chế là sai.
- **Suy diễn phải để `imgsz=640`**, đúng kích thước lúc huấn luyện. Đo trên tập test: 640 cho
  0,844/0,585 · 17,1 ms; 1280 cho 0,824/0,534 · 160,4 ms. Ở 1280, lớp `anten-5G` thiệt nặng nhất —
  trên ảnh `90.jpg` trạm VTU0510, độ tin cậy tụt 0,935 → 0,524, rớt dưới ngưỡng và **biến mất khỏi
  pipeline azimuth**.

---

## Giai đoạn C — Gán nhãn pose (4 điểm góc)

**Notebook:** [split_pose_clean_211.ipynb](split_pose_clean_211.ipynb)
**Nguồn nhãn:** `KLTN 2026.v2-kltn_data_pose_chua_tang_cuong.coco-segmentation` (211 ảnh, 568 annotation)
**Kết quả:** [KLTN_POSE_FINAL_2026/](KLTN_POSE_FINAL_2026/) — 4 lớp `anten-4G`, `anten-5G`, `rrh`, `rru`

### C.1 Cách gán nhãn

Vẽ tay trên Roboflow ở chế độ **instance segmentation**, mỗi ăng-ten là một đa giác **đúng 4 đỉnh**,
click theo thứ tự **TL → TR → BR → BL** quanh mặt bức xạ. Không dùng bbox — bbox không mang thông
tin hướng.

Chuyển `segmentation` → 4 keypoint: **đọc thẳng thứ tự đã click**, chỉ chuẩn hoá chiều duyệt
(dấu diện tích shoelace) và đảo nếu ngược.

### C.2 Chia tập TRƯỚC, tăng cường SAU — và chỉ tăng cường quang trắc

- Chia **80/10/10 theo ảnh gốc**, `seed = 42` → 169 / 21 / 21.
- Tăng cường **chỉ trên train**, ×4 (3 biến thể mỗi ảnh) → **676 / 21 / 21**.
- Phép tăng cường **chỉ đổi màu/nhiễu**: sáng, tương phản, bão hoà, nhiễu Gauss, làm mờ, làm nét,
  nén JPEG, hoán vị kênh, pha xám.

> ⚠️ **Tuyệt đối không lật, không xoay ở bước pose.** Lật ảnh sẽ hoán vị thứ tự bốn điểm góc — đúng
> cái lỗi mà bước hình học phía sau **không có cách nào phát hiện** (xem Giai đoạn E). Vì keypoint
> không đổi vị trí, tệp nhãn được chép y nguyên cho ảnh biến thể.

- `dataset.yaml`: `kpt_shape [4, 3]`, `flip_idx [1, 0, 3, 2]`.
- Có cell **kiểm tra rò rỉ bắt buộc = 0**: giao của tên gốc giữa train/val/test phải rỗng.

### C.3 Hai sự cố phải quay lại sửa

**1. Heuristic sắp lại 4 góc làm hỏng nhãn.** Hàm `order_4_corners` giải một vấn đề không tồn tại
(nhãn gốc vốn đã sạch: tứ giác tự cắt 0/568, điểm trùng 0/568, cùng chiều duyệt 567/568) và cả hai
đời đều gây hại:

| Đời | Cách làm | Hậu quả |
|---|---|---|
| 1 | tổng/hiệu toạ độ (`argmin(x+y)` = TL…) | ăng-ten chụp nghiêng → 2 chỉ số rơi vào cùng 1 điểm → tứ giác **sập** thành 2 điểm, hỏng **22,4%** nhãn |
| 2 | sắp theo góc quanh tâm + bắt đầu từ cạnh trên cùng | hết sập, nhưng **xoay lệch 1 bậc 8,8%** nhãn (rrh 64%, 5G 22%, 4G chỉ 1%) |

Bài học: mọi heuristic 2D đều gãy dưới phép xoay + phối cảnh. Cũng đã thử luật "cạnh P0→P1 của 4G
phải là cạnh ngắn" → **phá nhãn đang đúng**, vì phối cảnh mạnh làm cạnh 874 mm chiếu ngắn hơn cạnh
380 mm (13% số ca như vậy là ĐÚNG). **Kiểm chất lượng keypoint phải nhìn ảnh**, không có metric hình
học nào thay được — công cụ duyệt: [viz_keypoints.py](viz_keypoints.py) `--audit`.

Cũng vì vậy: **ảnh chụp xiên KHÔNG được loại bỏ** — chúng cho azimuth chính xác hơn ảnh chính diện.

**2. Nhãn không khớp trọng số.** Bộ nhãn đang lưu trong thư mục dữ liệu **không phải** bộ đã dùng để
huấn luyện. Đánh giá trên nhãn hiện hành cho 0,753 trong khi nhật ký huấn luyện ghi 0,818. Truy ngược
mới tìm ra bản đúng (`KLTN_POSE_FINAL_2026_labels_backup_angular`), chênh lệch ở 120/676 tệp train.
→ **Muốn đo lại mô hình pose thì phải trỏ dataset vào bản nhãn angular**, nếu không sẽ phạt oan
mô hình ~0,07 mAP (nặng nhất ở lớp 5G: 0,683 → 0,412).

---

## Giai đoạn D — Huấn luyện & chọn mô hình pose

**Notebook:** [NOTE_BOOK_POSE/](NOTE_BOOK_POSE/) (`TRAIN_YOLO26m_POSE.ipynb`, `TRAIN_YOLO26l_POSE.ipynb`)
**Trọng số:** [MO_HINH_POSE_KHOA_LUAN/](MO_HINH_POSE_KHOA_LUAN/)

### Cấu hình đã chốt

`epochs 200` · `patience 50` · `batch 16` · `imgsz 640` · 2×Tesla T4 · `cos_lr False` · `close_mosaic 10`
Loss gains: box 7.5 · cls 0.5 · dfl 1.5 · **pose 12.0** · kobj 1.0.

- `optimizer=auto` → iterations = 200 × ⌈676/64⌉ = 2.200 < 10.000 → **AdamW**,
  lr₀ = 0.002×5/(4+nc) = **0,00125**, momentum 0,9 (giá trị `lr0`/`momentum` trong `args.yaml` bị bỏ qua).
- **`seed = 0`** là seed lần chạy train; **`seed = 42`** chỉ dùng ở bước chia dữ liệu — hai chỗ khác
  nhau, phải nói rõ kẻo bị hỏi.
- **`dropout=0.25` KHÔNG có tác dụng với task pose** (Ultralytics: "classify only"). Vai trò chống
  quá khớp thuộc về `weight_decay` + dừng sớm + tăng cường.
- Dừng sớm thật: 26m dừng epoch 87 (best 37), 26l dừng epoch 76 (best 26) → mốc `close_mosaic`
  ở epoch 191 không bao giờ kích hoạt.

### Chọn 26m hay 26l — quyết định khó nhất

Pose AP@0.5:0.95 (OKS) trên **tập valid**:

| Lớp | n(val) | 26m | 26l |
|---|---|---|---|
| anten-4G | 48 | **0,881** | 0,873 |
| anten-5G | 7 | 0,689 | **0,750** |
| rrh | 2 | **0,945** | 0,845 |
| rru | 12 | 0,759 | **0,863** |
| TB 4 lớp | | 0,819 | **0,833** |
| **TB có trọng số theo số đối tượng** | | **0,857** | **0,857** |

Lập luận ba bước: (1) trung bình không trọng số nghiêng về 26l, nhưng nó cho lớp 5G chỉ 7 đối tượng
sức nặng ngang lớp 4G có 48; (2) **trung bình có trọng số hoà đến 3 chữ số**; (3) chênh lệch nằm trọn
ở lớp 5G (0,061) — nhỏ hơn một nửa độ phân giải của chính lớp đó (1/7 = 0,143) nên không phân định
được. Hoà → **chốt bằng chi phí**: 23,5M tham số / 26,2 ms so với 27,9M / 30,7 ms, cộng việc 26l đạt
đỉnh sớm hơn (epoch 26 vs 37) = dấu hiệu quá khớp trên tập train chỉ 169 ảnh.

Trên tập test (chỉ để **báo cáo** sau khi đã chốt): 26m 0,720 / 26l 0,792 ở 4 lớp — nhưng khoảng cách
đó đến gần trọn từ lớp `rrh` chỉ có **1 đối tượng**; riêng 2 lớp ăng-ten thì 26m **0,694** > 26l 0,589.

> **Lưu ý quan trọng: mAP pose KHÔNG đo được chất lượng azimuth.** OKS chuẩn hoá theo diện tích và
> đối xứng mọi hướng; còn PnP suy azimuth từ tỉ lệ hai cạnh đứng + độ hội tụ cạnh ngang. Keypoint
> trượt **dọc** cạnh → OKS gần như không đổi nhưng azimuth lệch; cả 4 điểm dịch đều → OKS tụt mạnh
> nhưng azimuth không đổi.

---

## Giai đoạn E — Xác định azimuth

**Thư viện:** [azimuth_optimized.py](azimuth_optimized.py)
**Chạy cho một thư mục ảnh bất kỳ:** [Azimuth_Folder_BatKy.ipynb](Azimuth_Folder_BatKy.ipynb)
**Bản trình bày từng bước:** [VTU0510_Azimuth_TungBuoc.ipynb](VTU0510_Azimuth_TungBuoc.ipynb)

### E.1 Chuỗi tính

```
ảnh ──► YOLO26m-pose  ──► 4 keypoint (TL,TR,BR,BL)   ─┐
    └─► YOLO26m detect ──► nhãn lớp (ghép theo IoU)  ─┤
                                                      ├─► solvePnP (IPPE, vật phẳng) + refine LM
        tiêu cự px từ EXIF (> FOV > f35=27mm) ────────┘        │
                                                               ▼
                          cột 3 của ma trận quay R = pháp tuyến mặt bức xạ
                                        │
                        az_rel = atan2(Nx, −Nz)      (góc lệch ăng-ten so với camera)
                                        │
                        Az = (heading − az_rel + offset) mod 360
```

Tham số chạy: `IMGSZ=640` · `POSE_CONF=0.20` · `DET_CONF=0.55` · `IOU_MATCH=0.30`.
Nhãn lớp quyết định kích thước tham chiếu trong `DIMS_BY_NAME` (tỉ lệ H:W — 4G **2,30**, 5G **1,10**,
rru 1,00), nên gán nhầm lớp làm lệch góc tới ~40°.

### E.2 `heading` — thứ duy nhất không suy ra được từ ảnh

Ảnh cho biết ăng-ten lệch bao nhiêu độ so với camera, **nhưng không cho biết camera chĩa về hướng
nào**. Đây là dữ liệu bắt buộc lấy từ ngoài, theo thứ tự ưu tiên trong notebook:

1. `GPSImgDirection` trong **EXIF** (Galaxy S24 Ultra có; ảnh gửi qua Zalo bị xoá sạch EXIF);
2. **tên tệp** chứa số hướng (`90.jpg`, `hd270.jpg`, `IMG_180.jpg`);
3. `HEADING_MANUAL` — **đọc tay từ hình chìm** của ứng dụng GPS Map Camera in ở góc ảnh;
4. hoặc **hồ sơ thiết kế trạm**: chỉ cần 1 azimuth thật là đủ, vì khoảng cách góc giữa các sector
   tính được mà không cần heading (heading triệt tiêu khi lấy hiệu) — chỉ cần một mốc tuyệt đối để
   xoay cả cụm.

Về hằng số `offset`: trạm duy nhất có số đo thật (LAN0890) cần khoảng **196–201°** mới khớp, trong
khi quy ước đang dùng theo chỉ đạo của người hướng dẫn đặt **`AZIMUTH_OFFSET = 0`**. Mâu thuẫn này
chưa giải quyết dứt điểm, cần thêm trạm có azimuth đo thật. Giả thuyết "180° nằm ở thứ tự keypoint"
**đã bị bác bằng thực nghiệm**: gán nhãn lại theo thứ tự click rồi train lại vẫn cần đúng offset đó.

### E.3 Cách đánh giá đúng — gộp theo sector, không chấm từng cái lẻ

Một ảnh nhìn thấy nhiều sector. Tính azimuth cho **từng thiết bị**, rồi gộp các lần đo cùng sector
bằng trung bình vòng tròn robust. Trên LAN0890: từng cái lẻ chỉ ~6/12 đạt ±15° (nhiễu keypoint trên
ảnh 480×720), **nhưng gộp 3 sector → 3/3 đạt**: 8,5 / 94,5 / 196,3 so với thật 10 / 100 / 200.

### E.4 Kiểm chứng khi không có azimuth thật — trạm VTU0510

3 ảnh chụp ở ba hướng, thu được 8 ăng-ten, sai số tái chiếu 0,54–2,85 điểm ảnh. Mỗi giá đỡ có một
ăng-ten 4G và một ăng-ten 5G **bắt buộc hướng trùng nhau**, mà hai ăng-ten được xử lý hoàn toàn độc
lập (khung bao riêng, keypoint riêng, kích thước tham chiếu khác nhau) → đây là **phép kiểm chứng
chéo không cần dữ liệu ngoài**. Bốn cặp lệch **2,8° · 4,7° · 5,1° · 5,9°**.

### E.5 Ba kết luận bằng số đã định hình cách hiểu hệ thống

- **Kích thước tuyệt đối không ảnh hưởng.** Giữ nguyên tỉ lệ H:W rồi phóng kích thước từ ×0,5 lên ×10
  cho ra góc giống hệt đến 4 chữ số thập phân (chỉ khoảng cách ước lượng đổi) → hệ thống không cần
  biết model ăng-ten cụ thể, chỉ cần **nhận diện đúng loại**.
- **Sai số tái chiếu nhỏ KHÔNG chứng minh nghiệm đúng.** 4 điểm đồng phẳng với 6 bậc tự do là bài
  toán vừa đủ xác định. Hoán vị thứ tự bốn góc đi một bước làm azimuth lệch **148°** mà RMS gần như
  y hệt (2,10 vs 2,07) và tilt cả hai đều "hợp lý". Tương quan giữa RMS và sai số azimuth: **r = +0,06**.
- **Tiêu cự không phải nút thắt.** Nút thắt thật là **nhiễu keypoint** trên ảnh độ phân giải thấp.
  Đại lượng thật sự tương quan với sai số là kích thước vật trên ảnh (r = −0,45): đường chéo box
  109–132 px → MAE 13,8°; 154–177 px → MAE 6,6°.

### E.6 Các nhánh đã thử rồi LOẠI BỎ vì đo thấy không có tác dụng hoặc có hại

| Nhánh | Kết quả đo |
|---|---|
| Ước lượng tiêu cự bằng Depth-Anything-3 thay EXIF | MAE 17,3° so với 10,8° của f35 cố định — **tệ hơn hẳn**; fx biến thiên 90–214 mm trong cùng một buổi chụp là phi lý vật lý |
| Trọng số theo độ bất định Monte-Carlo (nhiễu keypoint 2 px) | sai số sector 6,96 so với 4,46 — **hại** |
| Trung bình đa tỉ lệ (TTA qua imgsz 640/768/896/1024) | 6,19 so với 4,46 — chỉ xê dịch bias, không giảm nhiễu (mô hình train ở 640) |
| Khử nhập nhằng nghiệm gương bằng EM tự do | MAE 23 — tạo cụm chặt giả |
| Trọng số theo kích thước khung bao | tốt hơn ở LAN0890 nhưng **làm mất hẳn một sector ở VTU0510** → đã revert về trọng số RMS |
| Dùng `tilt` để chọn nghiệm gương | bắt được ca 5G vô hại, **bỏ lọt ca 4G chết người** → chỉ dùng làm cờ cảnh báo |
| Làm nét / cornerSubPix / upscale Lanczos | không thêm thông tin thật → không cải thiện |

> **Bài học chung:** hai trạm có thể cho kết luận ngược nhau về cùng một tuỳ chọn ⇒ không chốt một
> cơ chế từ dữ liệu một trạm.

### E.7 Vì sao KHÔNG báo cáo góc cụp (tilt)

`tilt` mà thuật toán trả về là **góc giữa mặt ăng-ten và trục quang camera**, tức ≈ góc ngước của
camera + góc cụp cơ khí — không phải downtilt. Đo tại VTU0510: 7/8 ăng-ten cho 47,0–58,9° (trung vị
51,6°), khớp góc ngước khi đứng chân cột chụp lên, trong khi hồ sơ trạm ghi downtilt thật chỉ **0–12°**.
EXIF không có trường pitch nên không tách được hai thành phần. → **Đề tài chỉ báo cáo azimuth.**

---

## Chạy lại

```
1. Chuẩn bị dữ liệu nhận diện   E:\TIEN_XU_LI_ANH_ANTENNA\Pipeline_Dataset.ipynb
2. Train nhận diện              E:\KAGGLE_UPLOAD\train_yolo26*.ipynb              (Kaggle, 2×T4)
3. Chuẩn bị dữ liệu pose        split_pose_clean_211.ipynb
4. Train pose                   NOTE_BOOK_POSE\TRAIN_YOLO26{m,l}_POSE.ipynb       (Kaggle, 2×T4)
5. Chạy azimuth                 Azimuth_Folder_BatKy.ipynb  (sửa mỗi ô cấu hình đầu tiên)
```

Môi trường: Python 3.10 · `ultralytics` (nhận diện 8.4.75–8.4.102, pose 8.4.107) ·
`opencv-python`, `numpy`, `scipy`, `Pillow`, `albumentations`.

Ba điều dễ sai khi chạy lại:
- **`imgsz` phải là 640** ở cả huấn luyện lẫn suy diễn.
- **Ghim đúng phiên bản `ultralytics`** đã dùng lúc huấn luyện; đổi phiên bản làm chỉ số xê dịch ở
  chữ số thập phân thứ ba.
- Trên Windows, khi gọi `YOLO(...).val(...)` phải đặt `workers=0` và bọc trong `if __name__=='__main__'`;
  GPU 4 GB thì **chỉ chạy một tiến trình đánh giá tại một thời điểm**.

---

## Hạn chế đã biết

1. **Hằng số hiệu chuẩn hướng chưa chốt** — xem E.2.
2. **Thứ tự bốn keypoint là điểm yếu duy nhất không có lưới an toàn.** Không cơ chế hình học nào phía
   sau bắt được lỗi hoán vị thứ tự; thứ duy nhất bảo vệ là mô hình chấm đúng thứ tự góc.
3. **Lớp `anten-5G` là nút thắt.** Mặt bức xạ gần vuông (tỉ lệ 1,10) nên hình chiếu ít thay đổi khi
   xoay → rất nhạy với nhiễu keypoint. Đã quét tỉ lệ giả định từ 1,00 đến 2,76: MAE phẳng 15,7→22,9,
   tức **kích thước giả định không phải nguyên nhân** — nguyên nhân là box 5G chỉ ~110 px trên ảnh 480×720.
4. **Ảnh chụp xa hoặc xiên cho sai số lớn.** Khuyến nghị quy trình khảo sát: mỗi ăng-ten cần ít nhất
   hai ảnh, trong đó có một ảnh tương đối trực diện và đủ gần, **giữ EXIF** (không gửi qua Zalo).

---

## Về dữ liệu

Ảnh và nhãn **không công bố kèm mã nguồn** — đây là ảnh khảo sát thực địa tại các trạm BTS đang vận
hành, mang theo mã trạm, toạ độ GPS, cấu hình thiết bị và góc phương vị thiết kế. Chi tiết ở
mục 6 của [README.md](README.md).
