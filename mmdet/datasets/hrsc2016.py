import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from DOTA_devkit.hrsc2016_evaluation import voc_eval
from mmdet.core import norm_angle
from mmdet.core import rotated_box_to_poly_single
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class HRSC2016Dataset(XMLDataset):
    CLASSES = ('ship',)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        self.img_names = []
        for img_id in img_ids:
            filename = f'AllImages/{img_id}.bmp'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)
            self.img_names.append(img_id)
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
            label = self.cat2label['ship']
            difficult = int(obj.find('difficult').text)
            bbox = []
            for key in ['mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h', 'mbox_ang']:
                bbox.append(obj.find(key).text)
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            cx, cy, w, h, a = list(map(float, bbox))
            # set w the long side and h the short side
            new_w, new_h = max(w, h), min(w, h)
            # adjust angle
            a = a if w > h else a + np.pi / 2
            # normalize angle to [-np.pi/4, pi/4*3]
            a = norm_angle(a)
            bbox = [cx, cy, new_w, new_h, a]

            ignore = False
            if self.min_size:
                assert not self.test_mode
                if bbox[2] < self.min_size or bbox[3] < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
            label = self.cat2label['ship']
            cat_ids.append(label)

        return cat_ids

    def evaluate(self, results, work_dir=None, gt_dir=None, imagesetfile=None):
        results_path = osp.join(work_dir, 'results_txt')
        mmcv.mkdir_or_exist(results_path)

        print('Saving results to {}'.format(results_path))
        self.result_to_txt(results, results_path)

        detpath = osp.join(results_path, '{:s}.txt')
        annopath = osp.join(gt_dir, '{:s}.xml')  # data/HRSC2016/Test/Annotations/{:s}.xml

        classaps = []
        map = 0
        for classname in self.CLASSES:
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            print(classname, ': ', ap)
            classaps.append(ap)

        map = map / len(self.CLASSES)
        print('map:', map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        # Saving results to disk
        with open(osp.join(work_dir, 'eval_results.txt'), 'w') as f:
            res_str = 'mAP:' + str(map) + '\n'
            res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
            f.write(res_str)
        return map

    def result_to_txt(self, results, results_path):
        img_names = [img_info['id'] for img_info in self.img_infos]

        assert len(results) == len(img_names), 'len(results) != len(img_names)'

        for classname in self.CLASSES:
            f_out = open(osp.join(results_path, classname + '.txt'), 'w')
            print(classname + '.txt')
            # per result represent one image
            for img_id, result in enumerate(results):
                for class_id, bboxes in enumerate(result):
                    if self.CLASSES[class_id] != classname:
                        continue
                    if bboxes.size != 0:
                        for bbox in bboxes:
                            score = bbox[5]
                            bbox = rotated_box_to_poly_single(bbox[:5])
                            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                osp.splitext(img_names[img_id])[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                                bbox[5], bbox[6], bbox[7])
                            f_out.write(temp_txt)
            f_out.close()
