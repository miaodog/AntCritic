#! /usr/bin/env python
# -*- coding:utf-8 -*-
import copy
import csv
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from restruct_result import restruct
from embedding import apply_aidesk_3
#显示所有列
# pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 30)


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


import numpy as np
from numpy.linalg import norm
def cosines_imilarity(a, b):
    A = np.array(a)
    B = np.array(b)
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    print("Cosine Similarity:", cosine)
    return cosine


def match_topic(res, topic, top_n=10):
    topic_prob = apply_aidesk_3(topic)
    cosines = {}
    for t, re in res.items():
        re['block_probs'] = []
        for i, bl in enumerate(re['blocks']):
            print(t, i, ''.join(list(bl.values())))
            block_prob = apply_aidesk_3(''.join(list(bl.values())))
            re['block_probs'].append(block_prob)
            cosine = cosines_imilarity(topic_prob, block_prob)
            cosines[(t, i)] = cosine

    cosines = {k: v for k, v in sorted(cosines.items(), key=lambda item: item[1])}
    print(cosines)
    return res


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/argument_mining/match_topic.py \
    -i /mnt/fengyao.hjj/transformers/data/topic_pgc/0425_2022042402000000344601.csv \
    -p /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0425_2022042402000000344601.prediction.csv \
    -t '三大指数均跌超5%'
    """
    args = cmd()
    input = args.input
    pred = args.pred
    topic = args.topic
    res = restruct(input, pred)
    topn = match_topic(res, topic, 10)
