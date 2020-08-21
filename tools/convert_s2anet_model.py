import argparse
import re
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict.items():
        if 'fam_cls' in key:
            continue
        out_state_dict[key] = val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)

def main():
    parser = argparse.ArgumentParser(description='RRoIDet Model Resave')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
