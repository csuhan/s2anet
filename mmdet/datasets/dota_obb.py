import json

import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class DotaOBBDataset(CustomDataset):
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    def load_annotations(self, ann_file):
        '''
        load annotations from .json ann_file
        '''
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.CLASSES)
        }
        with open(ann_file, 'r') as f_ann_file:
            self.data_dicts = json.load(f_ann_file)
            self.img_infos = []
            self.img_names = []
            for data_dict in self.data_dicts:
                img_info = {}
                img_info['filename'] = data_dict['filename']
                img_info['height'] = data_dict['height']
                img_info['width'] = data_dict['width']
                img_info['id'] = data_dict['id']
                self.img_infos.append(img_info)
                self.img_names.append(data_dict['filename'])
        return self.img_infos

    def get_ann_info(self, idx):
        ann_dict = self.data_dicts[idx]['annotations']
        ann = {}
        ann['bboxes'] = np.array(ann_dict['bboxes'])
        ann['bboxes_ignore'] = np.array(ann_dict['bboxes_ignore'])
        if len(ann_dict['labels']):
            ann['labels'] = np.array([self.cat2label[label]
                                      for label in ann_dict['labels']])
        else:
            ann['labels'] = np.array([])

        if len(ann_dict['labels_ignore']):
            ann['labels_ignore'] = np.array(
                [self.cat2label[label] for label in ann_dict['labels_ignore']])
        else:
            ann['labels_ignore'] = np.array([])

        return ann

    def get_cat_ids(self, idx):
        ann_dict = self.data_dicts[idx]['annotations']
        if len(ann_dict['labels']):
            return [self.cat2label[label] for label in ann_dict['labels']]
        else:
            return []
