# -*- coding: utf-8 -*-
r"""
azimuth_optimized.py  —  PIPELINE AZIMUTH ANTEN (BAN TOI UU 2026-06-26)
=======================================================================
Toi uu sau khi DO BANG SO tren bo anh mau (azimuth that = {10, 100, 200} do).

Ket luan da kiem chung (xem memory azimuth-fusion-optimization):
  * Nut that THUC SU = nhieu keypoint tren anh 480x720, KHONG phai mirror/tieu cu.
  * Disambiguation mirror KHONG cuu (oracle 10.1 vs 10.8; RMS-min da dung 11/12).
    -> giu RMS-min + "tilt-sanity guard" (chi lat khi RMS-min co tilt phi ly).
  * Trong so MC-uncertainty va multi-scale TTA: do thay HAI -> bo (MC chi lam co bao cao).
  * Phan gom cum / trung binh robust nhieu goc nhin DA BO khoi module nay: dau ra bay gio
    la azimuth CUA TUNG ANG-TEN. Muon doi chieu giua cac phep do thi lam o lop tren.

Cong thuc azimuth:  true_az = (heading - az_rel + AZIMUTH_OFFSET) % 360
  - heading        = goc la ban cua VI TRI CHUP (watermark, ten tep, hoac GPSImgDirection).
  - az_rel         = goc ang-ten so voi truc quang camera, suy tu phap tuyen mat buc xa.
  - AZIMUTH_OFFSET = hang so hieu chuan, MAC DINH 0.

Ve hang so hieu chuan (can doc ky truoc khi doi):
  * MAC DINH = 0 theo quy uoc "watermark = huong ang-ten nhin ra nguoi chup".
    Day la quy uoc dang dung trong luan van (Muc 4.12, bo anh mau).
  * Rieng bo anh mau (tram DUY NHAT co azimuth do that, GT = 10/100/200) can offset
    ~196-201 moi khop GT. Mau thuan nay CHUA giai quyet dut diem: hoac quy uoc watermark
    nguoc lai, hoac lech la ban thiet bi. Khi co it nhat 1 azimuth that thi dung
    calibrate_offset() de do lai, dung doan.

Tieu chi nghiem thu: moi ang-ten "dat" neu azimuth lech khong qua 15 do so voi azimuth that.
"""
import os, glob, math
import numpy as np
import cv2
from PIL import Image, ExifTags
try:
    from scipy.optimize import linear_sum_assignment  # Hungarian (gan toi uu detection<->pose)
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------------------------------------------------------------------- cau hinh
# Kich thuoc theo LOAI (chi TI LE H:W anh huong azimuth voi vat phang). Calibrate bang RMS.
DIMS_BY_NAME = {
    'anten-4G': (874.0, 380.0, 150.0),   # H:W = 2.30
    'anten-5G': (418.0, 380.0, 200.0),   # H:W = 1.10 (gan vuong -> az nhay keypoint)
    'rrh':      (380.0, 380.0, 150.0),
    'rru':      (380.0, 380.0, 150.0),
    'viba':     (380.0, 380.0, 150.0),
}
SKIP_NAMES = {'none'}

# ----------------------------------------------------------------------------- tien ich goc
def circ_dist(a, b):
    """Khoang cach goc nho nhat (deg), luon >= 0."""
    return abs(((a - b) + 180) % 360 - 180) # ví dụ 359 và 1 chỉ các nhau 2 thôi , phép tính thường sẽ bảo là cách 358

# -------HÀM ĐỔI TIÊU CỰ CAMERA SANG TIÊU CỰ DẠNG PIXEL------

def estimate_focal_px(image_path, img_w, img_h, f35_mm=27.0, fov_long_deg=None):
    """Uu tien EXIF > FOV > F35. (DA3 per-anh da do la HAI -> bo, xem memory.)"""
    long_side = float(max(img_w, img_h)) # DO DAI CUA ANH (px) THEO CHIEU DAI NHAT
    try:
        ex = Image.open(image_path)._getexif() or {}
        tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        f35 = tags.get('FocalLengthIn35mmFilm') or tags.get('FocalLengthIn35mmFormat')
        # DÙng khi có ảnh có tham chiếu là focal length in 35mm film (EXIF). Nếu không có thì dùng fov_long_deg. Nếu cũng không có thì dùng f35_mm mặc định.
        if f35:
            return long_side * float(f35) / 36.0, f'EXIF f35={f35}mm' # tINH TOAN focal px tu EXIF
    except Exception:
        pass
    if fov_long_deg is not None:
        return (long_side / 2.0) / math.tan(math.radians(fov_long_deg / 2.0)), f'FOV={fov_long_deg}deg'
    return long_side * float(f35_mm) / 36.0, f'f35={f35_mm}mm'
# dùng ma trận K vì yêu cầu kích thước dạng pixel ( mà ảnh chụp từ cam đc tính bằng mm)

# ----------------------------------------------------------------------------- pose
def box_iou(a, b): # tọa độ của 2 bbox , 1 cái do poosse , 1 cái do dêtct
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])# trên trái
    x2, y2 = min(a[2], b[2]), min(a[3], b[3]) # dưới phải
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)  # tính chiều rộng và chiều cao vùng giao
    inter = iw * ih  # tính diện tích phần giao 
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter # tính phần hợp , bao gồm tổng diện tích của 2 bbox trừ đi phần giao
    return inter / ua if ua > 0 else 0.0  # IoU = giao / hợp (0 nếu hợp <= 0)
# Quy uoc box: [0]=x1,[1]=y1 goc tren-trai ; [2]=x2,[3]=y2 goc duoi-phai.

def _tilt_plausible(tilt):
    """Anten that: downtilt co hoc + chup tu duoi len -> tilt ~ 0..80. Phi ly: am nhieu / >88."""
    return -12.0 <= tilt <= 88.0

def reprojection_rms(obj, ip, rvec, tvec, K, dist):
    """RMS reprojection error (px): chieu 4 goc 3D voi (rvec,tvec) roi so voi keypoint anh.
    Tach rieng de dung LAI cho ca 'truoc LM' lan 'sau LM' (khoi lap code, do dung 1 cong thuc)."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = proj.reshape(-1, 2) - ip
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

def calculate_pose(kp_2d, img_w, img_h, H_mm, W_mm, focal_px):
    """IPPE cho 2 nghiem guong. Chon RMS-min; CHI lat sang nghiem kia khi RMS-min co
    tilt PHI LY ma nghiem kia hop ly va rms khong te hon 1.5x (tilt-sanity guard).
    Tra ve dict: az_rel, tilt, rms (SAU LM, dung de chon & tinh trong so), rms_before
    (nghiem IPPE tho), rms_improve (=rms_before-rms, LUON >=0 vi LM la descent co damping),
    ambiguous, cands(list ca 2), rvec, tvec, K."""
    obj = np.array([[-W_mm/2,  H_mm/2, 0.0], [ W_mm/2,  H_mm/2, 0.0],
                    [ W_mm/2, -H_mm/2, 0.0], [-W_mm/2, -H_mm/2, 0.0]], dtype=np.float64)
    ip = np.array(kp_2d, dtype=np.float64)
    K = np.array([[focal_px, 0, img_w/2.0], [0, focal_px, img_h/2.0], [0, 0, 1]], dtype=np.float64) # ma trận K
    dist = np.zeros((4, 1)) # gia su meo anh bang 0
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, ip, K, dist, flags=cv2.SOLVEPNP_IPPE)# goi ham ippe với các tham số đầu vào 
    cands = []
    # Chọn 1 trong 2 nghiệm guong (rvec, tvec) dựa trên RMS-min và tilt-sanity guard
    for rvec, tvec in zip(rvecs, tvecs):
        # (1) RMS TRUOC LM = do khop cua nghiem IPPE tho (dung lam moc so sanh, in bao cao).
        rms_before = reprojection_rms(obj, ip, rvec, tvec, K, dist)
        # (2) Tinh chinh Levenberg-Marquardt: cuc tieu reprojection error tu diem khoi IPPE.
        #     LM la descent CO DAMPING -> bước làm tăng cost bị bỏ -> rms_after LUON <= rms_before
        #     (da do 0/4000 lan LM lam tang RMS). LM van CAN vi keo nghiem guong ve day that,
        #     giup so RMS-min giua 2 nghiem cong bang (co truong hop LM doi az toi hang chuc do).
        rvec, tvec = cv2.solvePnPRefineLM(obj, ip, K, dist, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        az_rel = np.degrees(np.arctan2(R[0, 2], -R[2, 2]))
        tilt = np.degrees(np.arctan2(R[1, 1], R[2, 1])) + 90.0
        # (3) RMS SAU LM = con so DUNG dung de chon nghiem (RMS-min) va tinh trong so fusion.
        rms = reprojection_rms(obj, ip, rvec, tvec, K, dist)
        cands.append(dict(az_rel=az_rel, tilt=tilt, rms=rms, rms_before=rms_before,
                          rms_improve=rms_before - rms, rvec=rvec, tvec=tvec, K=K))
    cands.sort(key=lambda c: c['rms'])
    best = dict(cands[0]); best['cands'] = cands; best['flipped'] = False
    best['ambiguous'] = False; best['az_alt'] = cands[1]['az_rel'] if len(cands) > 1 else None
    if len(cands) > 1:
        d_az = circ_dist(cands[0]['az_rel'], cands[1]['az_rel'])
        if cands[1]['rms'] < cands[0]['rms'] * 1.5 and d_az > 10:
            best['ambiguous'] = True
        # tilt-sanity guard: RMS-min phi ly nhung nghiem kia hop ly -> lat
        if (not _tilt_plausible(cands[0]['tilt']) and _tilt_plausible(cands[1]['tilt'])
                and cands[1]['rms'] < cands[0]['rms'] * 1.5):
            alt = dict(cands[1]); alt['cands'] = cands; alt['flipped'] = True
            alt['ambiguous'] = best['ambiguous']; alt['az_alt'] = cands[0]['az_rel']
            best = alt
    return best

def mc_uncertainty(kp_2d, img_w, img_h, H_mm, W_mm, focal_px, sigma=2.0, N=80, seed=0):
    """Std cua az_rel khi jitter keypoint ~sigma px (deg). CHI dung lam CO tin cay bao cao
    (vat gan vuong / xien -> std lon). KHONG dung lam trong so fusion (do thay hai)."""
    base = calculate_pose(kp_2d, img_w, img_h, H_mm, W_mm, focal_px)['az_rel']
    kp = np.array(kp_2d, float); rng = np.random.default_rng(seed); ds = []
    for _ in range(N):
        try:
            a = calculate_pose(kp + rng.normal(0, sigma, kp.shape), img_w, img_h, H_mm, W_mm, focal_px)['az_rel']
            ds.append(((a - base + 180) % 360) - 180)
        except Exception:
            pass
    return float(np.sqrt(np.mean(np.square(ds)))) if ds else float('nan')

# ----------------------------------------------------------------------------- YOLO -> detections
def detect_antennas(image_path, pose_model, det_model, focal_px,
                    pose_conf=0.25, det_conf=0.55, imgsz=640, iou_match=0.3,
                    with_mc=False, strict_match=True):
    """Chay pose + detect, ghep box theo IoU. Tra ve list det:
    dict(name, az_rel, tilt, rms, conf signals, kp, box, cx, cy).

    strict_match=True (mac dinh, CHONG GHEP NHAM o tram anten day):
      - ghep 1-1 TOI UU TOAN CUC bang Hungarian (linear_sum_assignment): cuc dai
        TONG IoU tren cac cap hop le -> moi pose CHI phuc vu 1 detection va nguoc lai,
        khong con 1 pose bi gan cho nhieu anten, va khong "ich ky" nhu greedy;
      - cong dieu kien TAM-TRONG-BOX (tam pose nam trong box detect HOAC nguoc lai)
        -> loai cap chi chong meo canh (anten ben canh) du IoU > nguong;
      - thieu scipy -> tu dong fallback greedy 1-1 (ket qua gan nhu Hungarian).
    strict_match=False: quay ve cach cu (moi detection lay pose IoU cao nhat, cho
      dung lai) de doi chieu A/B."""
    img = cv2.imread(image_path)
    if img is None:
        return [], (0, 0)
    img_h, img_w = img.shape[:2]
    rp = pose_model(img, conf=pose_conf, imgsz=imgsz, verbose=False)[0] # pose detection -> Xuất ra class + box + keypoint
    rd = det_model(img, conf=det_conf, imgsz=imgsz, verbose=False)[0] # object detection -> Xuất ra class + box
    # Chia rõ ràng 2 model để dễ kiểm soát , dùng detect để nhận diện class , dinfg pose để xác định 4 góc 
    # sau đó gộp lại theo IoU để xác định anten nào có pose và loại anten đó là gì
    pose_dets = [] # danh sách lưu kết quả pose detection (box và keypoint) của anten
    # posebbox(box và keypoint)-> 4 điểm keypoint của anten 
    # detectionbbox(class và box )-> anten đó là loại gì
    if rp.boxes is not None and rp.keypoints is not None: # đóng gói anten pose thành 1 cặp ( box,keypoint)
        pbx = rp.boxes.xyxy.cpu().numpy(); pkp = rp.keypoints.xy.cpu().numpy()
        for i in range(len(pbx)):
            pose_dets.append((pbx[i], pkp[i]))
    out = []
    if rd.boxes is None or len(rd.boxes) == 0: # nếu không detect được anten nào thì out sớm
        return out, (img_w, img_h)
    dbx = rd.boxes.xyxy.cpu().numpy(); dcl = rd.boxes.cls.cpu().numpy().astype(int)
    det_idx = [j for j in range(len(dbx)) if det_model.names[dcl[j]] not in SKIP_NAMES] # bỏ SKIP_NAMES

    def _center_in(b, box): # tâm của b có nằm trong box không (chống box chồng lệch cạnh)
        cx, cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    # ---- GHÉP detection <-> pose: chọn cách theo strict_match ----
    matched = {} # matched[j] = i (chỉ số pose khớp cho detection j)
    if strict_match:
        # Ghép 1-1 TỐI ƯU (Hungarian): dựng ma trận điểm IoU cho các cặp HỢP LỆ
        # (đủ 4 kp + IoU>=ngưỡng + tâm-trong-box), rồi cực đại TỔNG IoU -> gán 1-1 toàn cục
        # (tránh "ích kỷ" của greedy khi IoU giằng co ở trạm dày). Ô cấm = điểm tệ nhất.
        npose = len(pose_dets)
        score = np.full((len(det_idx), npose), -1.0)   # -1 = cặp cấm
        for r, j in enumerate(det_idx):
            db = dbx[j]
            for i, (pb, kp) in enumerate(pose_dets):
                if len(kp) != 4:
                    continue
                v = box_iou(db, pb)
                if v < iou_match:
                    continue
                if not (_center_in(pb, db) or _center_in(db, pb)): # tâm lệch -> nghi anten bên cạnh
                    continue
                score[r, i] = v
        has_valid = score.size and bool((score >= 0).any())
        if _HAS_SCIPY and has_valid:
            cost = np.where(score >= 0, -score, 1.0)   # cực đại IoU = cực tiểu (-IoU); ô cấm = 1.0 (tệ nhất)
            rows, cols = linear_sum_assignment(cost)
            for r, i in zip(rows, cols):
                if score[r, i] >= 0:                   # LOẠI cặp bị Hungarian ép gán vào ô cấm
                    matched[det_idx[r]] = int(i)
        elif has_valid:
            # fallback greedy nếu không có scipy (kết quả gần Hungarian, chỉ khác khi IoU giằng co)
            pairs = sorted(((score[r, i], r, i) for r in range(len(det_idx)) for i in range(npose)
                            if score[r, i] >= 0), reverse=True)
            used_pose = set()
            for v, r, i in pairs:
                j = det_idx[r]
                if j in matched or i in used_pose:
                    continue
                matched[j] = i; used_pose.add(i)
    else:
        # Cách CŨ (A/B): mỗi detection lấy pose IoU cao nhất, cho phép trùng pose, không xét tâm.
        for j in det_idx:
            db = dbx[j]
            bi, bv = -1, 0.0 # bv : bbox iou cao nhất ; #bi: vị trí pose cao nhất
            for i, (pb, kp) in enumerate(pose_dets):
                v = box_iou(db, pb)
                if v > bv:
                    bv, bi = v, i
            if bi >= 0 and bv > iou_match and len(pose_dets[bi][1]) == 4: # điều kiện : phải có box , iou > 0.3, có 4 điểm trong kp
                matched[j] = bi

    # ---- xuất kết quả theo bảng ghép ----
    for j in det_idx:
        name = det_model.names[dcl[j]]; db = dbx[j]
        rec = dict(name=name, box=db.tolist(), cx=float((db[0]+db[2])/2), cy=float((db[1]+db[3])/2),
                   has_pose=False)
        i = matched.get(j, -1)
        if i >= 0: # có pose khớp -> tính azimuth
            kp = pose_dets[i][1]
            H_c, W_c, _ = DIMS_BY_NAME.get(name, (380.0, 380.0, 150.0))
            best = calculate_pose(kp, img_w, img_h, H_c, W_c, focal_px)
            rec.update(has_pose=True, az_rel=best['az_rel'], tilt=best['tilt'], rms=best['rms'],
                       rms_before=best['rms_before'], rms_improve=best['rms_improve'],
                       ambiguous=best['ambiguous'], flipped=best['flipped'], kp=np.array(kp).tolist())
            rec['mc_std'] = (mc_uncertainty(kp, img_w, img_h, H_c, W_c, focal_px) if with_mc else None)
        out.append(rec)
    return out, (img_w, img_h)

# ----------------------------------------------------------------------------- FUSION (loi toi uu)
def calibrate_offset(dets, known_azimuths, base_offset=0.0):
    """Do lech la ban b: cho moi det 1 azimuth THAT (known, gan theo GT), tim offset toan cuc
    toi thieu sai so. Quet off 0..360 tu do (KHONG ep +180). Tra ve (offset_tot_nhat, b=offset-base, MAE)."""
    raw = [((d['heading'] - d['az_rel']) % 360) for d in dets if d.get('has_pose')]
    best_off, best_mae = base_offset, 1e9
    for off in np.arange(0, 360, 0.5):
        errs = [min(circ_dist((r + off) % 360, g) for g in known_azimuths) for r in raw]
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae, best_off = mae, float(off)
    return best_off, (best_off - base_offset) % 360, best_mae

# ----------------------------------------------------------------------------- driver
def run_folder(folder, pose_weights, det_weights, headings, known_azimuths=None,
               azimuth_offset=0.0, f35_mm=27.0, fov_long_deg=None,
               imgsz=640, det_conf=0.55, pose_conf=0.20, iou_match=0.3,
               with_mc=False, auto_offset=False, strict_match=True):
    """Chay het mot thu muc anh -> in va tra ve GOC PHUONG VI CUA TUNG ANG-TEN.

    Khong gom cum, khong lay trung binh: moi ang-ten trong moi anh la mot ket qua doc lap.
    Muon doi chieu giua cac phep do (vd so cap 4G/5G cung gia do) thi lam o lop tren.

    Tham so dang chu y:
      headings        dict {ten_tep: goc_la_ban_cua_vi_tri_chup}
      known_azimuths  list azimuth THAT neu co -> dung de cham sai so tung ang-ten
      auto_offset     True + known_azimuths -> tu do lai hang so hieu chuan (calibrate_offset)

    Tra ve: list dict, moi dict la mot thiet bi; xem khoa 'has_pose' de biet co tinh duoc goc.
    """
    from ultralytics import YOLO
    if isinstance(pose_weights, (list, tuple)):
        raise ValueError('pose_weights chi nhan MOT duong dan (nhanh ensemble da bo).')
    pose_model = YOLO(pose_weights)
    det_model = YOLO(det_weights)
    files = [f for f in glob.glob(os.path.join(folder, '*.*'))
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_dets = []
    print(f'== {folder} | offset={azimuth_offset} | imgsz={imgsz} | pose_conf={pose_conf} ==')

    for fp in sorted(files):
        nm = os.path.basename(fp)
        heading = float(headings.get(nm, 0.0))
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        fpx, fsrc = estimate_focal_px(fp, w, h, f35_mm, fov_long_deg)
        dets, _ = detect_antennas(fp, pose_model, det_model, fpx, pose_conf, det_conf,
                                  imgsz, iou_match, with_mc=with_mc, strict_match=strict_match)
        for d in dets:
            d['heading'] = heading
            d['image'] = nm
            if d.get('has_pose'):
                d['true_az'] = (heading - d['az_rel'] + azimuth_offset) % 360
        all_dets.extend(dets)
        npose = sum(1 for d in dets if d.get('has_pose'))
        print(f'  {nm[:26]:28} hd={heading:5.0f} | {len(dets)} thiet bi -> {npose} tinh duoc goc | {fsrc}')

    if auto_offset and known_azimuths:
        off, b, mae = calibrate_offset(all_dets, known_azimuths)
        print(f'\n[calibrate] offset tot nhat = {off:.1f}  (lech la ban = {b:.1f})  MAE = {mae:.1f}')
        azimuth_offset = off
        for d in all_dets:
            if d.get('has_pose'):
                d['true_az'] = (d['heading'] - d['az_rel'] + azimuth_offset) % 360

    # ---- bang ket qua tung ang-ten ----
    # rms0 = RMS truoc buoc tinh chinh LM ; rms = sau LM (con so dung de chon nghiem).
    co = [d for d in all_dets if d.get('has_pose')]
    print('\n--- GOC PHUONG VI TUNG ANG-TEN ---')
    cot = f'{"anh":<22}{"hd":>5}{"lop":>10}{"az_rel":>8}{"azimuth":>9}{"rms0":>6}{"rms":>6}{"mc":>6}'
    print(cot + (f'{"sai so":>8}  co' if known_azimuths else '  co'))
    for d in sorted(co, key=lambda x: (x['image'], x['true_az'])):
        fl = []
        if d.get('flipped'):
            fl.append('LAT')
        if d.get('ambiguous'):
            fl.append('nhap-nhang')
        if d.get('mc_std') and d['mc_std'] > 25:
            fl.append(f"mc!{d['mc_std']:.0f}")
        mc = f"{d['mc_std']:.0f}" if d.get('mc_std') is not None else '-'
        dong = (f"{d['image'][:20]:<22}{d['heading']:>5.0f}{d['name']:>10}{d['az_rel']:>8.1f}"
                f"{d['true_az']:>9.1f}{d.get('rms_before', d['rms']):>6.1f}{d['rms']:>6.1f}{mc:>6}")
        if known_azimuths:
            e = min(circ_dist(d['true_az'], g) for g in known_azimuths)
            d['err'] = e
            dong += f'{e:>8.1f}'
        print(dong + '  ' + ' '.join(fl))

    if known_azimuths and co:
        errs = [d['err'] for d in co]
        dat = sum(1 for e in errs if e <= 15)
        print(f'\n  >> {dat}/{len(errs)} ang-ten sai so <= 15 do | trung binh {np.mean(errs):.2f} do'
              f' | trung vi {np.median(errs):.2f} do')

    thieu = [d for d in all_dets if not d.get('has_pose')]
    if thieu:
        print(f'\n  {len(thieu)} thiet bi do duoc nhung khong ghep duoc keypoint: '
              + ', '.join(sorted({d['name'] for d in thieu})))
    return all_dets


if __name__ == '__main__':
    # Bo anh khao sat tram mau: 3 anh chup o 3 huong, ten tep chinh la goc la ban.
    # EXIF con nguyen -> tieu cu doc RIENG cho tung anh (90.jpg chup zoom 115mm, hai anh
    # con lai 69mm), nen KHONG dat f35_mm chung cho ca thu muc.
    FOLDER = r'E:\AZIMUTH_TEST_LAN1\TRAM_MAU'
    # DUNG DUNG 2 MO HINH DA CHOT TRONG LUAN VAN (Chuong 4) - doi la khong con tai lap duoc so.
    POSE = r'E:\AZIMUTH_TEST_LAN1\KLTN_HUYNHNGUYENQUAN\MO_HINH_POSE\FILE_KET_QUA\training_results_yolo26m\pose_model\weights\best.pt'
    DET = r'E:\AZIMUTH_TEST_LAN1\KLTN_HUYNHNGUYENQUAN\best.pt'
    HEADINGS = {'90.jpg': 90.0, '180.jpg': 180.0, '270.jpg': 270.0}

    # Tram nay CHUA co azimuth do that -> khong cham duoc sai so tuyet doi.
    # Kiem chung thay the: moi gia do co 1 ang-ten 4G va 1 ang-ten 5G phai huong gan trung
    # nhau, nen doc do lech cua tung cap trong bang ket qua.
    run_folder(FOLDER, POSE, DET, HEADINGS, azimuth_offset=0.0, with_mc=False)
