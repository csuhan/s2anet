import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmdet.core import poly_to_rotated_box_single

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter']

label_ids = {name: i + 1 for i, name in enumerate(wordname_15)}


def parse_ann_info(label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name + '.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = tuple(poly_to_rotated_box_single(bbox).tolist())
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(label_ids[class_name])
            elif difficult == 1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label_ids[class_name])
    return bboxes, labels, bboxes_ignore, labels_ignore


def convert_dota_to_mmdet(src_path, out_path, trainval=True, filter_empty_gt=True, ext='.png'):
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output pkl file path
        trainval: trainval or test
    """
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)

    data_dict = []
    for id, img in enumerate(img_lists):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = os.path.join(label_path, img_name + '.txt')
        img = Image.open(osp.join(img_path, img))
        img_info['filename'] = img_name + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        if trainval:
            if not os.path.exists(label):
                print('Label:' + img_name + '.txt' + ' Not Exist')
                continue
            # filter images without gt to speed up training
            if filter_empty_gt & (osp.getsize(label) == 0):
                continue
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(label_path, img_name)
            ann = {}
            ann['bboxes'] = np.array(bboxes, dtype=np.float32)
            ann['labels'] = np.array(labels, dtype=np.int64)
            ann['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)
            img_info['ann'] = ann
        data_dict.append(img_info)

    mmcv.dump(data_dict, out_path)


if __name__ == '__main__':
    convert_dota_to_mmdet('data/dota_1024/trainval_split',
                         'data/dota_1024/trainval_split/trainval_s2anet.pkl')
    convert_dota_to_mmdet('data/dota_1024/test_split',
                         'data/dota_1024/test_split/test_s2anet.pkl', trainval=False)
    print('done!')
