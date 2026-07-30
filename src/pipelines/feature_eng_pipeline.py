# -*- coding: utf-8 -*-
r"""split_pose_dataset.py - chia KLTN_POSE_FROM_SEG thanh train/val/test (80/10/10)
theo ANH (616 anh nguon deu khac nhau -> khong ro ri) + tao dataset.yaml chuan V4.
Ket qua: KLTN_POSE_V5_SEG/{train,val,test}/{images,labels} + dataset.yaml -> zip len Kaggle."""
import os, glob, shutil, random

SRC = r'E:\AZIMUTH_TEST_LAN1\KLTN_POSE_FROM_SEG'
OUT = r'E:\AZIMUTH_TEST_LAN1\KLTN_POSE_V5_SEG'
RATIO = (0.8, 0.1, 0.1)   # train / val / test
SEED = 42

def main():
    random.seed(SEED)
    imgs = sorted(glob.glob(os.path.join(SRC, 'images', '*.jpg')))
    random.shuffle(imgs)
    n = len(imgs)
    n_tr = int(n * RATIO[0]); n_va = int(n * RATIO[1])
    splits = {'train': imgs[:n_tr], 'val': imgs[n_tr:n_tr + n_va], 'test': imgs[n_tr + n_va:]}
    for sp, files in splits.items():
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(OUT, sp, sub), exist_ok=True)
        miss = 0
        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            lbl = os.path.join(SRC, 'labels', base + '.txt')
            if not os.path.exists(lbl):
                miss += 1; continue
            shutil.copy(fp, os.path.join(OUT, sp, 'images', os.path.basename(fp)))
            shutil.copy(lbl, os.path.join(OUT, sp, 'labels', base + '.txt'))
        print(f'{sp:6}: {len(files)} anh (thieu label: {miss})')
    with open(os.path.join(OUT, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write('# KLTN Pose V5 (tu COCO segmentation -> 4 keypoint, chia 80/10/10 theo anh)\n'
                'path: ' + OUT.replace('\\', '/') + '\n'
                'train: train/images\nval: val/images\ntest: test/images\n\n'
                'kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n\n'
                'names:\n  0: anten-4G\n  1: anten-5G\n  2: rrh\n  3: rru\nnc: 4\n')
    print(f'\nXONG. Dataset -> {OUT}\\dataset.yaml')
    print('Nen thu muc KLTN_POSE_V5_SEG thanh .zip roi Add Input len Kaggle.')

if __name__ == '__main__':
    main()
