#! /usr/bin/env python
# -*- coding:utf-8 -*-

def split_into_subsents(raw_text):
    clean_srcs = []
    clean_src = ''
    punctuation1 = ['。', '？', '?', '！', '!', '；', ';', '|']  # 必须分割
    # punctuation2 = ['：', ':', '，', ','] # 分割但是看长度
    punctuation2 = [] # 分割但是看长度
    pres_w = ''
    for n, w in enumerate(raw_text):
        if w == '\u3000' and pres_w == '\u3000' and len(clean_src) > 10:
            clean_srcs.append(clean_src)
            clean_src = ''

        # 以 \n 分割
        if len(clean_src) > 0 and w == '\\' and n < len(raw_text) - 2 and raw_text[n + 1] == 'n':
            clean_srcs.append(clean_src)
            clean_src = ''

        # \n 出现在标点符号分割后的句首，则去掉
        if len(clean_src) == 0 and w == '\\':
            pres_w = w
            continue
        elif len(clean_src) == 0 and pres_w == '\\' and w == 'n':
            pres_w = w
            continue
        else:
            clean_src = clean_src + w
        pres_w = w

        # 以标点符号分割
        if w in punctuation1:
            if n < len(raw_text) - 2 and (raw_text[n + 1] not in punctuation1 and raw_text[n + 1] != ')' and raw_text[n + 1] != '）'):
                clean_srcs.append(clean_src)
                clean_src = ''

        if (w in punctuation2 and len(clean_src) >= 20):
            clean_srcs.append(clean_src)
            clean_src = ''

        # 以 1、分割
        # 以 1[whitespace]、 分割
        if (w not in punctuation1 + punctuation2
                and n < len(raw_text) - 2
                and raw_text[n + 2] == '、'
                and raw_text[n + 1] in {'1'}) \
            or (w not in punctuation1 + punctuation2
                and n < len(raw_text) - 3
                and raw_text[n + 3] == '、'
                and raw_text[n + 2] == ' '
                and raw_text[n + 1] in {'1'}):
            clean_srcs.append(clean_src)
            clean_src = ''
    if clean_src != '':
        clean_srcs.append(clean_src.replace('\\n', ''))
    return clean_srcs


if __name__=="__main__":
    split_into_subsents('')