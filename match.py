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


request_type = "embedding"
topic = '三大指数均跌超5%'
prob = apply_aidesk_3(topic)
print(len(prob))