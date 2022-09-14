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
                                    if len(s) > 0 and ':' in s:
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
                        pass
                        # print('Warning: check other spans: ', sp, sp.attrs['class'])
    return result


def reclean_text(html_result):
    clean_text = ''
    for po, r in html_result.items():
        for pi, t in r.items():
            clean_text = clean_text + t['text']
    return clean_text


if __name__=="__main__":
    args = cmd()
    html_text = "<p></p><blockquote>  <p>买入机会，卖出风险，不盲目操作，只做自己认知内的交易。少操作，多思考，不断提升自己，如果不能克服心理障碍，再多的失败也不能转化为成功。</p></blockquote><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);\">今天在大金融的护盘拉升中，大盘一路震荡向上，早盘黄白两线还呈分化状态，但临近午盘时黄线快速拉升，中小盘跌幅也相继缩小。板块上，房地产、医疗、证券涨幅居前，光伏、电池、酒店餐饮跌幅居前。今天主要是大金融的行情，大消费因为口罩问题的影响，预期有所降低，目前低位的大金融就成资金首选对象了。</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">\n</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">今日操作：</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\"></span></p><p><span style=\"background-color: rgb(255, 255, 255);\">1、半导体 </span></p><p>半导体回补缺口后震荡回升，走势还不错，我半导体本身仓位不少了，今天就小幅低吸下。\n</p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">个人操作：加仓一万</span></p><p>\n</p><p><span style=\"background-color: rgb(255, 255, 255);\">2、白酒 </span></p><p>白酒今天只是超跌反弹，上方缺口还没有回补，目前还是弱势震荡，没有走出确定性，继续观察。\n</p><p><span style=\"color: rgb(219, 80, 43); background-color: rgb(255, 255, 255);\">个人操作：底仓不操作</span></p><p><span style=\"color: rgb(219, 80, 43); background-color: rgb(255, 255, 255);\">\n</span></p><p>3、医疗</p><p>医疗今天反弹还行，但下方留了个缺口，回补的概率比较大，想做T也可以。</p><p><span style=\"color: rgb(219, 80, 43);\">个人操作：底仓不操作</span></p><p>\n</p><p>4、猪肉</p><p>猪肉下午持续拉升，走的还不错，有资金进场的迹象，但目前还在趋势线下，可以再观察一天，看下周一能否重新站回。\n</p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">个人操作：底仓不操作</span></p><p>\n</p><p>5、军工 </p><p>军工探底后回升，目前跌幅缩小，有拒绝调整的迹象，但调整并没有结束，不着急进场。\n</p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">个人操作：底仓不操作</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">\n</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);\">6、光伏</span></p><p>光伏观点和军工一样，继续耐心等待买点。\n</p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">个人操作：底仓不操作</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">\n</span></p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);\">7、基建、地产</span></p><p>地产今天涨幅很大，是个小的卖点，但今天回补缺口，突破压力位也有继续走加速的可能，可以部分落袋为安，剩下的继续博弈。\n</p><p><span style=\"color: rgb(219, 80, 43);\">个人操作：底仓不操作</span></p><p>\n</p><p>8、恒生科技</p><p>恒生科技今天强势反弹，重新站回了箱体，目前还没到上方压力位，继续拿好。\n</p><p><span style=\"color: rgb(219, 80, 43);\">个人操作：持仓不动</span></p><p>\n</p><p><span style=\"background-color: rgb(255, 255, 255); color: rgb(219, 80, 43);\">如果尾盘有什么变更，我会评论留言置顶或者发小动态~~</span></p><p><span style=\"background-color: rgb(255, 233, 222);\">好了，就说这么多了，分析不易，路过的小伙伴们顺手给阿坤点个赞啊，评论留言\"冲冲冲\"，大家一起互相加油！大家的支持就是阿坤分享的最大动力！谢谢了！</span></p><p><span style=\"background-color: rgb(222, 242, 255);\">\n</span></p><p></p><p><span style=\"background-color: rgb(222, 242, 255);\"></span></p><p>股市有风险，投资需谨慎\n在市场面前我们永远都是学生，要懂得敬畏市场，做好仓位管理，找到适合自己的交易模式。</p><p>\n</p><p>发文时间：</p><p>早评：09：00前</p><p>午评：12：30前</p><p>操作：14：50前</p><p>复盘：23：00前</p><p>\n</p><p>个人部分持仓，仅是个人纪录，不做为投资参考。</p><p><span style=\"color: rgb(219, 80, 43);\">集火波段区：</span></p><p><span class=\"stock-name\" data-name=\"银河创新成长混合C\" data-code=\"014143\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 4w</p><p><span class=\"stock-name\" data-name=\"南方中证全指证券公司ETF联接C\" data-code=\"004070\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 2.1w</p><p><span class=\"stock-name\" data-name=\"天弘恒生科技指数(QDII)C\" data-code=\"012349\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 1.1w</p><p><span style=\"color: rgb(219, 80, 43);\">底仓观察区：</span></p><p><span style=\"color: rgb(219, 80, 43);\"><span class=\"stock-name\" data-name=\"鹏华酒指数C\" data-code=\"012043\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> </span><span style=\"color: rgb(0, 0, 0);\"> 0.1w</span></p><p><span style=\"color: rgb(0, 0, 0);\"><span class=\"stock-name\" data-name=\"招商中证白酒指数C\" data-code=\"012414\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 0.1w</span></p><p><span style=\"color: rgb(219, 80, 43);\"><span class=\"stock-name\" data-name=\"华夏国证半导体芯片ETF联接C\" data-code=\"008888\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> <span style=\"color: rgb(0, 0, 0);\">0.1w</span></span></p><p><span class=\"stock-name\" data-name=\"长盛医疗行业量化配置股票\" data-code=\"002300\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 0.2w</p><p><span class=\"stock-name\" data-name=\"招商中证新能源汽车指数C\" data-code=\"013196\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 0.1w</p><p><span class=\"stock-name\" data-name=\"汇添富中证新能源汽车产业指数(LOF)C\" data-code=\"501058\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 0.1w</p><p><span class=\"stock-name\" data-name=\"广发中证全指汽车指数C\" data-code=\"004855\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\"></span> 0.1w</p><p><span class=\"stock-name\" data-name=\"国泰中证畜牧养殖ETF联接C\" data-code=\"012725\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">国泰中证畜牧养殖ETF联接C</span> 500</p><p><span class=\"stock-name\" data-name=\"易方达中证军工指数(LOF)C\" data-code=\"012842\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">易方达中证军工指数(LOF)C</span> 0.1w</p><p><span class=\"stock-name\" data-name=\"鹏华中证空天军工指数(LOF)C\" data-code=\"010364\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">鹏华中证空天军工指数(LOF)C</span> 0.1w</p><p><span class=\"stock-name\" data-name=\"广发中证基建工程ETF联接C\" data-code=\"005224\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">广发中证基建工程ETF联接C</span> 0.1w</p><p><span class=\"stock-name\" data-name=\"中欧医疗健康混合C\" data-code=\"003096\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">中欧医疗健康混合C</span> 0.2w</p><p><span class=\"stock-name\" data-name=\"招商中证畜牧养殖ETF联接C\" data-code=\"014415\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">招商中证畜牧养殖ETF联接C</span> 500</p><p><span class=\"stock-name\" data-name=\"诺安成长混合\" data-code=\"320007\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">诺安成长混合</span> 0.1w</p><p><span class=\"stock-name\" data-name=\"南方中证全指房地产ETF联接C\" data-code=\"004643\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">南方中证全指房地产ETF联接C</span> 0.1w</p><p></p><p> <span class=\"supertalk hide\" data-name=\"南京证券涨停!\" data-code=\"2022090902000000525201\">#南京证券涨停!</span> </p>"
    html_result = html_features_extractor(html_text.replace('\\"', '"'))
    clean_text = reclean_text(html_result)
    clean_text = re.sub('\{\{\{[^<>]+?\}\}\}', '', clean_text)
