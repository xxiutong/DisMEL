import argparse
from omegaconf import OmegaConf


def setup_parser(default_val=None):
    default_val = default_val if default_val is not None else '/data/zyt/MMOR/config/WikiMEL.yaml'
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--config', type=str, default='/data/zyt/MMOR/config/WikiMEL.yaml')
    parser.add_argument('--m', type=str, default='', help='Description to save in read.txt')
    parser.add_argument('--config', type=str, default=default_val)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    args.m = _args.m
    return args
