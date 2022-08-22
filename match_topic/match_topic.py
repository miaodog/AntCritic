#! /usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
from numpy.linalg import norm
from restruct_result import restruct
from embedding import apply_aidesk_3


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-i', '--input', type=str, default='', help='ref file')
    parser.add_argument('-p', '--pred', type=str, default='', help='pred file')
    parser.add_argument('-t', '--topic', type=str, default='', help='pred file')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args





if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/argument_mining/match_topic/match_topic.py \
    -i /mnt/fengyao.hjj/transformers/data/topic_pgc/0427_2022042702000000348601.csv \
    -p /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0427_2022042702000000348601.prediction.csv \
    -t '大反弹!创指涨超5%'
    """
    args = cmd()
    input = args.input
    pred = args.pred
    topic = args.topic
    res = restruct(input, pred)
    top_n_res = match_topic(res, topic, 10)
    for t, re in res.items():
        print('\n\n')
        print(f"==========={t}===========")
        for bin, bl in enumerate(re['blocks']):
            if (t, bin) in top_n_res:
                print(top_n_res[(t, bin)])
                print(f"block {bin}: {bl} {re['cinblocks'][bin]}")