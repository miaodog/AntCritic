#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import requests


def get_seg(content):
    url = "http://cv-cross.gateway.alipay.com/ua/invoke"
    headers = {
        "Content-Type": 'application/json',
    }
    params = {"uri": "ner_antfin_high_gpu",
              "serviceCode": "datacube-8585",
              "params": {
                  "content": content,
                  "output_type": "all",
                  "source": "kezun.zkz_caifu_classify"

              },
              "appId": "antfcu",
              "appName": "",
              "attributes": {
                  "_ROUTE_": "UA"
              }
              }

    response = requests.request("POST",
                                url,
                                data=json.dumps(params),
                                headers=headers)
    tokens = []
    entities = []

    response = json.loads(response.text)
    if 'success' in response and response['success'] and "resultMap" in response:
        ret = response["resultMap"]
        if "seg_list" in ret and "entities" in ret:
            tokens.extend([[item[0], item[1]] for item in ret['seg_list']])
            entities.extend(ret["entities"])
    return entities, tokens


if __name__ == '__main__':
    entities = get_seg('这是一个测试，马云今天在新加坡会见李克强总理')
    print(entities)
