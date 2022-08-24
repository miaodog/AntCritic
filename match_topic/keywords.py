#!/usr/bin/python
# -*- coding:utf-8 -*-
import requests
import json
from datetime import datetime


def get_hotwords(content):
    """
    调用cto线的关键词服务；
    :param id:
    :param content:
    :return:
    """
    params = {"serviceCode": "datacube-8181",
              "uri": "mkg_inc_hot_event",
              "params": {"unique_id": "xxx", "type": "KWD", "content": content},
              "appId": "insurance_textsim", "appName": "南朔", "attributes": {"_ROUTE_": "UA"}
              }
    res = {}
    try:
        url = "http://cv-cross.gateway.alipay.com/ua/invoke"
        # url = "http://cv.gateway.alipay.com/ua/invoke"
        headers = {'Content-Type': 'application/json'}
        datas = json.dumps(params)
        r = requests.post(url, data=datas, headers=headers, timeout=10)
        res = json.loads(r.text)
    except Exception as e:
        print('{} - ERROR get_hotwords Exception {}'.format(datetime.now(), e.__str__()))

    if res.get('resultCode', 'Fail') == 'SUCCESS' and res.get('resultMap', None) \
            and res['resultMap'].get('algo_result', None) and res['resultMap']['algo_result'].strip() != "":
        rst_keywords_list = res['resultMap']['algo_result'].strip().split('\t')
        rst_keywords_list = [(val.split(':')[0], float(val.split(':')[1])) for val in rst_keywords_list if ':' in val]
        rst_keywords_list.sort(key=lambda x: x[1], reverse=True)
        rst_keywords_list = ["{}:{}".format(val[0], val[1]) for val in rst_keywords_list]
        return '\t'.join(rst_keywords_list)
    else:
        return ""


def main():
    id = '1222'
    content = "林志颖驾特斯拉出车祸"
    results =get_hotwords(id, content)
    print('results: ', results)


if __name__ == '__main__':
    main()
