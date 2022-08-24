#! /usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import re
from selectolax import parser
from split2subsents import split_into_subsents
from bs4 import BeautifulSoup, Tag



def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f1', '--file', type=str, default=None, help='')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


def special_span(line):
    replace_tags = []
    for i in re.finditer('<span[^<>]+? data-name=[^<>]+?></span>', line):
        data_name = \
        [t.split('=')[1].split('"')[1] for t in line[i.span()[0]: i.span()[1]].split(' ') if 'data-name' in t][0]
        replace_tags.append(data_name)
    line = re.sub('<span[^<>]+? data-name=[^<>]+?></span>', ' _PLACEHOLDER_ ', line)
    i = 0
    fill = []
    for t in line.split(' '):
        if t == '_PLACEHOLDER_':
            t = replace_tags[i]
            i = i + 1
            fill.append(t)
        else:
            fill.append(t)
    line = ''.join(fill)
    return line


def html_extractor(line):
    text = re.sub('<[^<>]+?>', '', line)
    text = re.sub('\s+', '', text).strip()
    return text


def find_lcsubstr(s1, s2):
    # 最长公共子串
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度


def html_features_extractor(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    # print(soup.prettify())
    result = {}
    po = 0  # 段编号
    for c in soup.children:
        for subp in c.body:
            clean_text = html_extractor(special_span(str(subp)))
            if clean_text.strip() != '' and clean_text.strip() != '\\n':
                po += 1  # 以p，blockquote, None 分段
                result[po] = {}  # 初始化段内句子属性信息
                sentences = split_into_subsents(clean_text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&'))
                for j, s in enumerate(sentences):
                    result[po][j + 1] = {'text': s, 'font-size': -1, 'color': -1, 'background-color': -1,
                                        'strong': 0, 'sns-small-title': 0, 'sns-blob-tl': 0, 'supertalk': 0,
                                         'blockquote': 0, 'h4': 0}
                    if subp.name != 'p' and isinstance(subp, Tag):
                        result[po][j + 1][subp.name] = 1

                if not isinstance(subp, Tag):
                    continue

                # sns-small-title(<p>tag的属性): 这里所有的句子都应该具有这个属性；
                if 'class' in subp.attrs and 'sns-small-title' in subp.attrs['class']:
                    key, value = 'sns-small-title', 1
                    for j, s in enumerate(sentences):
                        result[po][j + 1][key] = value

                # strong: 当分句s2和加粗句子s1的最长公共子串长度至少为s1的一半以上时，就设置该分句s2为加粗句子；
                key, value = 'strong', 1
                strongs = subp.findAll('strong') + subp.findAll('b')
                for k, s1 in enumerate(strongs):
                    for j, s2 in enumerate(sentences):
                        substr, max = find_lcsubstr(s1.text, s2)
                        if max > len(s1.text) / 2 or max > len(s2) / 2:
                            result[po][j + 1][key] = value

                # font-size/color(该属性在span内): 当分句s一半以上在该span里面或者span内一半以上在s中时，就设置该分句s的属性为该span的属性；
                for k, sp in enumerate(subp.findAll('span')):
                    span_text = html_extractor(special_span(str(sp)))
                    if 'class' in sp.attrs and sp.attrs['class'][0] == 'stock-name':
                        # 产品名字不设置属性
                        continue
                    if 'style' in sp.attrs:
                        for j, s in enumerate(sentences):
                            substr, max = find_lcsubstr(s, span_text)
                            if max > len(s) / 2 or max > len(span_text) / 2:
                                styles = sp.attrs['style'].strip().split(';')
                                for s in styles:
                                    if len(s) > 0:
                                        key, value = s.split(':')[0].strip(), s.split(':')[1].strip()
                                        result[po][j + 1][key] = value
                    elif 'supertalk' in ' '.join(sp.attrs['class']):
                        # supertalk(该属性在span内): 当该span的内容都在分句s里时，就设置该分句s的supertalk为1；
                        key, value = 'supertalk', 1
                        for j, s in enumerate(sentences):
                            if span_text in s or ''.join(sentences) == span_text:
                                result[po][j + 1][key] = value
                    elif 'sns-blob-tl' in ' '.join(sp.attrs['class']):
                        key, value = 'sns-blob-tl', 1
                        for j, s in enumerate(sentences):
                            if span_text in s:
                                result[po][j + 1][key] = value
                    else:
                        print('Warning: check other spans: ', sp, sp.attrs['class'])
    return result


def reclean_text(html_result):
    clean_text = ''
    for po, r in html_result.items():
        for pi, t in r.items():
            clean_text = clean_text + t['text']
    return clean_text


if __name__=="__main__":
    args = cmd()
    html_text = "<p>中药股{{{S_0}}}<strong>开盘一字跌停</strong>，临近中午强势拉升翻红，尾盘<strong>封上涨停直至收盘，强势拿下四连板，上演惊人的“地天板”一幕</strong>。</p>{{{P_0}}} <p><img src=\"https://mdn.alipayobjects.com/afts/img/fT9WSoShHr42rO1IjnU8KAAAAZgwAQBr/original?bz=wealth_community_transfer\"></p> <p>早在3月9日，大理药业就上演过类似走势，如今又“如法炮制”。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/b7dSQYiDF9kZYWdArI3G9wAAAZgwAQBr/original?bz=wealth_community_transfer\"></p> <p>拉长时间来看，大理药业近两个月股价累计最高涨幅达103.4%**。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/Hwl-R7LSRuYX3mlLvAJD1QAAAZgwAQBr/original?bz=wealth_community_transfer\"></p> <p>从大理药业所属的中药板块来看，<strong>年初至今涨幅位列第三</strong>，仅次于{{{S_1}}}和{{{S_2}}}。</p>{{{P_1}}} <p><img src=\"https://mdn.alipayobjects.com/afts/img/zDw3TJivX4Ba0TBAuPu4aAAAAZgwAQBr/original?bz=wealth_community_transfer\"></p> <p>背后是那些资金在<strong>疯狂涌入</strong>呢？大理药业盘后公布的<strong>一日榜数据显示</strong>，龙虎榜资金净卖出771.8万元。<strong>有“散户大本营”之称的东方财富证券拉萨营业部位列买四席位。</strong>知名游资东莞证券北京分公司、东亚前海证券江苏分公司位列买三和卖一席位。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/LWFhT59EDmRvM_pOi_UQrQAAAZgwAQBr/original?bz=wealth_community_transfer\"></p> <p>大理药业盘后公布的<strong>三日榜数据显示</strong>，龙虎榜资金净卖出114.75万元。<strong>有“散户大本营”之称的东方财富证券拉萨营业部位列卖五席位。</strong>知名游资东亚前海证券江苏分公司位列买一和卖一席位；东莞证券北京分公司位列买五席位。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/oLkXQKacdv-ejMBIs39qcgAAK5gwAQBr/original?bz=wealth_community_transfer\"></p> <p>大理药业晚间最新公告，公司的产品醒脑静注射液拟备选广东联盟清开灵等中成药集中带量采购。</p> <p>公开资料显示，大理药业主营业务系中西药注射剂的生产与销售。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/pFYSQoEnNAiAVfNNqC5YQQAAK5gwAQBr/original?bz=wealth_community_transfer\"></p> <p>据悉，大理药业主要产品醒脑静注射液和参麦注射液均被国家卫健委和国家中医药管理局<strong>列入第六版、第七版、第八版和第九版《新型冠状病毒肺炎诊疗方案》的推荐用药</strong>。</p> <p>消息面上，近期，世卫组织表示：<strong>中药能有效治疗新冠肺炎</strong>，降低轻型、普通型病例转为重症，缩短病毒清除时间，<strong>鼓励成员国考虑吸纳中医药</strong>。</p> <p>1月29日，大理药业公布业绩预告，<strong>2021年度净利润为-4500万元到-4000万元</strong>。</p> <p>值得注意的是，卖方机构对大理药业兴趣寥寥，<strong>公司自2017年9月22日上市至今五年以来，皆无券商研报覆盖</strong>。</p> <p><img src=\"https://mdn.alipayobjects.com/afts/img/MZTEQawsKX_dTbpFs_oGtAAAAZgwAQBr/original?bz=wealth_community_transfer\"></p>"
    html_result = html_features_extractor(html_text.replace('\\"', '"'))
    clean_text = reclean_text(html_result)
    clean_text = re.sub('\{\{\{[^<>]+?\}\}\}', '', clean_text)
