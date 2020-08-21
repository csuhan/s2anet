import argparse
import os
import os.path as osp

import mmcv

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.core import result2dota_task1,result2dota_task2
from DOTA_devkit.ResultMerge_multi_process import mergebypoly, mergebyrec


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('result_path', help='test config file path')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    
    outputs = mmcv.load(args.result_path)
    dst_path = args.out
    dst_raw_path = osp.join(dst_path, 'result_raw')
    dst_merge_path = osp.join(dst_path, 'result_merge')
    if not osp.exists(dst_path):
        os.mkdir(dst_path)
    if not osp.exists(dst_raw_path):
        os.mkdir(dst_raw_path)
    if not osp.exists(dst_merge_path):
        os.mkdir(dst_merge_path)
    print('convert result to dota result format at {}'.format(dst_raw_path))
    result2dota_task1(outputs, dst_raw_path, dataset)
    print('merge result')
    mergebyrec(dst_raw_path, dst_merge_path)
    
    
if __name__ == '__main__':
    main()
