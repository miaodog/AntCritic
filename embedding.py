#! /usr/bin/env python
# -*- coding:utf-8 -*-
import requests
import json
import time


SERVICE_URL = "http://cv-cross.gateway.alipay.com/ua/invoke"
serviceCode = "datacube-8033"


def apply_aidesk_3(text):
    request_type = "embedding"
    if not isinstance(text,str):
        query = str(text)
    if len(text) == 0:
        return None
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "params":
        {
            "type": request_type,
            "text": text
        },
        "appId": "f5bda3e2884a1075",
        "appName": "sentence_embedding",
        "serviceCode": serviceCode,
        "attributes":
        {
            "_TIMEOUT_": "60000",
            "_ROUTE_": "UA"
        }
    }
    json_str = json.dumps(data)
    response = requests.post(SERVICE_URL, headers=headers, data=json_str)

    result_dict = response.json()
    prob = eval(result_dict['resultMap']['algo_result'])['result']
    return prob[0]


if __name__ == "__main__":
    text = '这种时候，会有个很神奇的现象，就是：每天都有你想不到的事，都有打破你预期的事。无数种利空，但是最终都会疲劳的，也就是冲击会一波一波减弱最重要的就一句话：相信常识。'
    res = apply_aidesk_3(text)
    print(res)
