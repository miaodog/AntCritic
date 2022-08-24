# coding:utf-8
# @author: fengyao
# @aile: inference_ops.py

import tensorflow as tf
import cognition.framework as cf
from cognition.framework import logger, op_logger
import json
import argument_mining_op

cur_version = 'latest'


class ArgumentMiningSession(cf.CfSession):
    def __init__(self):
        op_logger.info("init ArgumentMiningSession")
        self.support_versions = [cur_version]
        self.params_dict_str = ""
        self.content_id_str = ""
        self.content_str = ""


@cf.service("inference")
class BlurDetService:
    @cf.session
    def create_session(self):
        return ArgumentMiningSession()

    @cf.decoder
    def parser(self, data, headers):
        json_data = json.loads(data)
        if 'params' in json_data and 'itemId' in json_data['params']:
            item_id = json_data['params']['itemId']
        elif 'itemId' in json_data:
            item_id = json_data['itemId']
        else:
            item_id = ''
            op_logger.info('No item_id field in params json string.')

        if 'params' in json_data and 'title' in json_data['params']:
            title = json_data['params']['title']
        elif 'content' in json_data:
            title = json_data['title']
        else:
            title = ''
            op_logger.info('No title field in params json string.')

        if 'params' in json_data and 'content' in json_data['params']:
            content = json_data['params']['content']
        elif 'content' in json_data:
            content = json_data['content']
        else:
            content = ''
            op_logger.info('No content field in params json string.')

        headers_format = {}
        for key, val in headers.items():
            if 'Version' == key:
                headers_format['version'] = val
            else:
                headers_format[key] = val
        if 'version' not in headers_format:
            headers_format['version'] = cur_version
            op_logger.info('need config model version, run in default version "latest"')
        op_logger.info("[%s]request headers:%s", cf.get_session_ctx(cf.CTX_SESSION_ID), json.dumps(headers_format))
        return [item_id, title, content, str.encode(json.dumps(headers_format, ensure_ascii=False))]

    @cf.service_init
    def build_service(self):
        self.item_id_str = tf.placeholder(tf.string)
        self.title_str = tf.placeholder(tf.string)
        self.content_str = tf.placeholder(tf.string)
        self.params_dict_str = tf.placeholder(tf.string)
        # 调用op，注意此处要传入的参数
        argument_mining = cf.add_node(cf.OpNode("argument_mining"),
                                             [self.content_id_str, self.title_str, self.content_str, self.params_dict_str])
        return (argument_mining, [self.item_id_str, self.content_str, self.params_dict_str])