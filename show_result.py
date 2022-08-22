#! /usr/bin/env python
# -*- coding:utf-8 -*-
import copy
import csv
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
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
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


def raw2sentence(raw_dict):
    sentence = []
    for key, val in raw_dict.items():
        sentence.append(val)
    return sentence


def res2dict(titles, srcs, sents, preds, grids):
    title = ''
    bres = {}
    for t, src, s, p, g in zip(titles, srcs, sents, preds, grids):
        if title != t:
            bres[t] = {'sents': [], 'preds': [], 'grids': [], 'src': src}
            title = t
        bres[t]['sents'].append(s)
        bres[t]['preds'].append(p)
        bres[t]['grids'].append(g)
    return bres

def beautiful_claim_premise(res):
    for t, re in res.items():
        claims = []
        premises = []
        print('\n\n')
        print(f"=================={t}====================")
        print(re['src'])
        for i, (p, g) in enumerate(zip(re['preds'], re['grids'])):
            if 'Claim' == p:
                claim = None
                for cl in claims:
                    if i in cl:
                        claim = cl
                        claims.remove(cl)
                        break
                claim = {i: re['sents'][i]} if claim is None else claim
                for r in g:
                    z, c, relation = int(r[1:-1].split('-')[0]), int(r[1:-1].split('-')[1]), '-'.join(r[1:-1].split('-')[2:])
                    if i == z and relation == 'Co-reference' and re['preds'][c] == 'Claim':
                        claim[c] = re['sents'][c]
                if claim not in claims:
                    claims.append({k: v for k, v in sorted(list(claim.items()))})
            if 'Premise' == p:
                premise = None
                for pre in premises:
                    if i in pre:
                        premise = pre
                        premises.remove(pre)
                        break
                premise = {i: re['sents'][i]} if premise is None else premise
                for r in g:
                    z, c, relation = int(r[1:-1].split('-')[0]), int(r[1:-1].split('-')[1]), '-'.join(r[1:-1].split('-')[2:])
                    if i == z and relation == 'Co-reference' and re['preds'][c] == 'Premise':
                        premise[c] = re['sents'][c]
                if premise not in premises:
                    premises.append({k: v for k, v in sorted(list(premise.items()))})

        # clean duplication: 把被其他论据包含的论据删除, 前面的代码只能保证列表后面的不被前面的包含，下面则是保证列表前面的不被后面的包含
        pop_indexes = []
        for i in range(len(claims)):
            for j in range(i+1, len(claims)):
                if True in [key in list(claims[j].keys()) for key in list(claims[i].keys())]:
                    claims[j].update(claims[i])
                    pop_indexes.append(i)
        claims = [{k: v for k, v in sorted(list(c.items()))} for i, c in enumerate(claims) if i not in pop_indexes]
        pop_indexes = []
        for i in range(len(premises)):
            for j in range(i+1, len(premises)):
                if True in [key in list(premises[j].keys()) for key in list(premises[i].keys())]:
                    premises[j].update(premises[i])
                    pop_indexes.append(i)
        premises = [{k: v for k, v in sorted(list(p.items()))} for i, p in enumerate(premises) if i not in pop_indexes]
        # 上面一个论点论据的segment聚合的时候，只考虑Co-reference的关系；
        # 下面为一个论点找论据的时候，考虑所有的关系，即只要相关，正确的应该是只考虑论据到论点的Affiliation，但是目前的模型预测的Affiliation非常少；
        # 所以即考虑论点到论据的三种关系，也考虑论据到论点的三种关系；

        # 为每个claim找premise
        claims2premises = {}
        premises2claims = {}
        for cin, cl in enumerate(claims):
            raw_indexes, new_indexes = [], []
            for k, v in cl.items():
                for r in re['grids'][k]:
                    z, c, relation = int(r[1:-1].split('-')[0]), int(r[1:-1].split('-')[1]), '-'.join(r[1:-1].split('-')[2:])
                    # 跟此claim相关的segment indexes
                    raw_indexes.append(c)
                    raw_indexes.append(z)
            raw_indexes = list(set(raw_indexes))
            for i in raw_indexes:
                if re['preds'][i] == 'Premise':
                    for pin, pre in enumerate(premises):
                        if i in pre and pin not in new_indexes:
                            new_indexes.append(pin)
            claims2premises[cin] = new_indexes
        # 为每个premise找claim
        for pin, pre in enumerate(premises):
            raw_indexes, new_indexes = [], []
            for k, v in pre.items():
                for r in re['grids'][k]:
                    z, c, relation = int(r[1:-1].split('-')[0]), int(r[1:-1].split('-')[1]), '-'.join(r[1:-1].split('-')[2:])
                    # 跟此claim相关的segment indexes
                    raw_indexes.append(c)
                    raw_indexes.append(z)
            raw_indexes = list(set(raw_indexes))
            for i in raw_indexes:
                if re['preds'][i] == 'Claim':
                    for cin, cl in enumerate(claims):
                        if i in cl and cin not in new_indexes:
                            new_indexes.append(cin)
            premises2claims[pin] = new_indexes
        re['rclaims'] = copy.deepcopy(claims)
        re['rpremises'] = copy.deepcopy(premises)
        re['claims'] = claims
        re['premises'] = premises
        re['premises2claims'] = premises2claims
        re['claims2premises'] = claims2premises
        # print("===========claim===========")
        # for cin, cl in enumerate(re['rclaims']):
        #     print(f'claim {cin}: {cl} {claims2premises[cin]}')
        # print("===========premise===========")
        # for pin, pre in enumerate(re['rpremises']):
        #     print(f'premise {pin}: {pre} {premises2claims[pin]}')
    return res


def merge_claim_premise(res):
    for t, re in res.items():
        # print('\n\n')
        # print(f"++++++++++++++++++{t}+++++++++++++++++++")
        # print(re['src'])
        claims, premises = re['claims'], re['premises']
        premises2claims, claims2premises = re['premises2claims'], re['claims2premises']
        for cin, cl in enumerate(claims):
            # 对于每个论点，将其指向的论据加入，同时将指向该论点的论据加入
            for pin in claims2premises[cin]:
                cl.update(premises[pin])
                premises[pin] = {}
            for k, v in premises2claims.items():
                if cin in v:
                    cl.update(premises[k])
                    premises[k] = {}
            claims[cin] = {k: v for k, v in sorted(list(cl.items()))}
        # print("===========claim===========")
        # for cin, cl in enumerate(claims):
        #     print(f'claim {cin}: {cl} {claims2premises[cin]}')
        # print("===========premise===========")
        # for pin, pre in enumerate(premises):
        #     print(f'premise {pin}: {pre} {premises2claims[pin]}')
    return res

def adj(n, l):
    if n >= l[0] - 1 and n <= l[-1] + 1:
        return True
    else:
        return False

def merge_premises(res):
    for t, re in res.items():
        # print('\n\n')
        # print(f"++++++++++++++++++{t}+++++++++++++++++++")
        # print(re['src'])
        claims, premises = re['claims'], re['premises']
        for pin in range(1, len(premises)):
            if len(premises[pin-1]) > 0 and len(premises[pin]) > 0 and adj(list(premises[pin-1].keys())[-1], list(premises[pin].keys())):
                premises[pin].update(premises[pin-1])
                premises[pin] = {k: v for k, v in sorted(list(premises[pin].items()))}
                premises[pin-1] = {}
        # print("===========claim===========")
        # for cin, cl in enumerate(claims):
        #     print(f'claim {cin}: {cl}')
        # print("===========premise===========")
        # for pin, pre in enumerate(premises):
        #     print(f'premise {pin}: {pre}')
    return res


def merge_premise2claim(res):
    for t, re in res.items():
        re['blocks'] = []
        # print('\n\n')
        # print(f"++++++++++++++++++{t}+++++++++++++++++++")
        # print(re['src'])
        claims, premises = re['claims'], re['premises']
        for pin, pre in enumerate(premises):
            # 对于每个论据，将其加入到拥有相邻segment的论点里面
            flag = False
            for cin, cl in enumerate(claims):
                if flag is False:
                    for k, v in pre.items():
                        if adj(k, list(cl.keys())):
                            cl.update(pre)
                            claims[cin] = {k: v for k, v in sorted(list(cl.items()))}
                            premises[pin] = {}
                            flag = True
                            break
        # print("===========claim===========")
        # for cin, cl in enumerate(claims):
        #     print(f'claim {cin}: {cl}')
        # print("===========premise===========")
        # for pin, pre in enumerate(premises):
        #     print(f'premise {pin}: {pre}')
    return res


def merge_claims(res):
    for t, re in res.items():
        print('\n\n')
        print(f"==========={t}===========")
        print(re['src'])
        claims, premises = re['claims'], re['premises']
        for cin in range(1, len(claims)):
            if len(claims[cin-1]) > 0 and len(claims[cin]) > 0 and adj(list(claims[cin-1].keys())[-1], list(claims[cin].keys())):
                claims[cin].update(claims[cin-1])
                claims[cin] = {k: v for k, v in sorted(list(claims[cin].items()))}
                claims[cin-1] = {}
        re['blocks'] = [c for c in claims if len(c) > 0]
        rclaims, re['cinblocks'] = [], []
        for c in re['rclaims']:
            rclaims = rclaims + list(c.keys())
        for b in re['blocks']:
            re['cinblocks'].append([k for k, v in b.items() if k in rclaims])
        print("===========claim & premise = blocks===========")
        for bin, bl in enumerate(re['blocks']):
            print(f"block {bin}: {bl} {re['cinblocks'][bin]}")
    return res


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/argument_mining/show_result.py \
    -i /mnt/fengyao.hjj/transformers/data/topic_pgc/0425_2022042402000000344601.csv \
    -p /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0425_2022042402000000344601.prediction.csv
    """
    args = cmd()
    input = args.input
    pred = args.pred
    # with open(pred, 'r') as pred:
    #     preds = [l.strip().split(' ')[1] for l in pred]
    # print(preds, len(preds))
    srcs = []
    sents = []
    titles = []
    csv_file = pd.read_csv(input, encoding='utf-8', delimiter=',')
    df = pd.read_csv(pred, encoding='utf-8', delimiter=',')
    for row_idx in tqdm(range(len(csv_file))):
        row = csv_file.loc[row_idx]
        sentences, tags, _ = raw2sentence(eval(row[2])), eval(row[3]), eval(row[4])
        titles = titles + [row[0]] * len(sentences)
        srcs = srcs + [row[1]] * len(sentences)
        sents = sents + sentences
    df['sents'] = sents
    df['titles'] = titles
    df['grid'] = df['grid'].map(lambda x: [r for r in x.strip()[1:-1].split(', ') if 'No-Relation' not in r])
    # print(tabulate(df[['titles', 'preds', 'sents', 'grid']], showindex=False, headers=df.columns))
    res = res2dict(titles, srcs, sents, list(df['preds']), list(df['grid']))
    res = beautiful_claim_premise(res)
    res = merge_claim_premise(res)
    res = merge_premises(res)
    res = merge_premise2claim(res)
    res = merge_claims(res)






