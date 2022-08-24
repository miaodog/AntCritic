# coding:utf-8
# @author: fengyao
import sys
import os
import copy
import numpy as np
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
cog_path = os.path.join(os.path.dirname(BASEDIR), "cognition/src")
sys.path.append(BASEDIR)
sys.path.append(cog_path)
from preprocess.second_stage import transform_tags, zero_pad, generate_char_tokenizer, generate_word_tokenizer, generate_models, load_model
import models
from config.config import get_cfg_defaults
from torch import nn
from html_tag_parser import html_features_extractor, reclean_text
from match_topic.result import res2dict, beautiful_claim_premise
import torch
import re
import time
import json
import tensorflow as tf
import cognition.framework as cf
from cognition.framework import logger, op_logger
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
op_logger.info(f'torch_device: {torch_device}')
MAX_PASSAGE_LEN = 400
MAX_TEXT_LEN = 50

def convert_config(d):
    op_logger.info(d)
    config = type('new', (object,), d)
    config.char_model_path = os.path.join(BASEDIR, config.char_model_path)
    config.word_model_path = os.path.join(BASEDIR, config.word_model_path)
    config.model_name_or_path = os.path.join(BASEDIR, config.model_name_or_path)
    config.model_config = os.path.join(BASEDIR, config.model_config)
    return config

@cf.op("argument_mining", tf.string)
class ArgumentMining:
    @cf.op_init
    def init_model(self, raw_config):
        self.cur_model_version = "latest"
        self.config = convert_config(raw_config)
        cfg = get_cfg_defaults()
        if self.config.model_config is not None:
            cfg.merge_from_file(os.path.join(BASEDIR, self.config.model_config))
        cfg.freeze()
        self.model_config = cfg.model
        print('self.model_config: ', self.model_config)
        self.char_model_path = self.config.char_model_path
        self.word_model_path = self.config.word_model_path
        self.model_name_or_path = self.config.model_name_or_path

        char_path = os.path.join(BASEDIR, "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch")
        word_path = os.path.join(BASEDIR, 'pretrained_model/paraphrase-xlm-r-multilingual-v1')
        char_base = BertTokenizer.from_pretrained(char_path, do_lowcheckpointser_case=True)
        self.char_tokenizer = generate_char_tokenizer(char_base)
        self.word_tokenizer = generate_word_tokenizer(word_path)

        char_model_path = os.path.join(BASEDIR, self.char_model_path)
        word_model_path = os.path.join(BASEDIR, self.word_model_path)
        char_pretrained = os.path.join(BASEDIR, "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch")
        word_pretrained = os.path.join(BASEDIR, "pretrained_model/paraphrase-xlm-r-multilingual-v1")

        self.char_model, self.word_model = generate_models(char_model_path, word_model_path, char_pretrained, word_pretrained)
        self._init_model(self.model_config)
        print('self.model: ', self.model)
        # state_dict = torch.load(self.model_name_or_path, map_location=torch.device(torch_device))
        # parameters = state_dict['model_parameters']
        # self.model.load_state_dict(parameters)
        # self.model.eval()
        op_logger.info("Completed loading model in {}......".format(self.model_name_or_path))
        state_dict = torch.load(self.model_name_or_path, map_location=torch.device(torch_device))
        parameters = state_dict['model_parameters']
        self.model.module.load_state_dict(parameters)
        self.model.eval()
        print('Load models from {}.'.format(self.model_name_or_path))

    def _init_model(self, model_config):
        self.model = getattr(models, self.model_config["name"])(**model_config).to(torch_device)
        if 'cuda' in torch_device:
            self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.model = nn.DataParallel(self.model)

    @cf.op_preprocessing
    def pre_processing(self, session, content_id, title, content, params_dict_str):
        session.params_dict_str = params_dict_str
        content_id = content_id.decode(errors='replace')
        title = title.decode(errors='replace')
        content = content.decode(errors='replace')
        content = re.sub('\{\{[^<>]+?\}\}', ' ', content)
        try:
            html_result = html_features_extractor(content)
        except Exception as e:
            # todo: 处理
            raise Exception
        source = reclean_text(html_result)
        sentences, tags = [], []
        for po, pre in html_result.items():
            for pi, res in pre.items():
                sentences.append(res['text'])
                res.pop('text')
                res.update({'po': po, 'pi': pi})
                tags.append(res)

        def get_sentence_results(sentences):
            char_ids, char_masks, word_ids, word_masks = [], [], [], []
            for sentence in sentences:
                char_id, char_mask, char_len = self.char_tokenizer(sentence)
                word_id, word_mask, word_len = self.word_tokenizer(sentence)
                char_ids.append(char_id)
                char_masks.append(char_mask)
                word_ids.append(word_id)
                word_masks.append(word_mask)
            char_id, char_mask = torch.stack(char_ids, dim=0), torch.stack(char_masks, dim=0)
            word_id, word_mask = torch.stack(word_ids, dim=0), torch.stack(word_masks, dim=0)
            with torch.no_grad():
                char_logit, char_embedding = self.char_model(char_id, char_mask, word_id, word_mask)
                word_logit, word_embedding = self.word_model(char_id, char_mask, word_id, word_mask)
            overall_logit = ((char_logit.softmax(-1) + word_logit.softmax(-1))).log_softmax(-1)
            overall_embedding = torch.cat((char_embedding, word_embedding), dim=-1)
            return overall_logit.cpu().numpy(), overall_embedding.cpu().numpy()

        def annotation_transform(sentences, tags):
            annotation = {"embedding": [], "coarse_logit": [], "is_major": [],
                          "sentence_mask": [], "paragraph_order": [], "sentence_order": [],
                          "reflection": [], "font_size": [], "style_mark": [],
                          "sentences": sentences, "tags": tags}
            result = copy.deepcopy(annotation)

            if len(tags) > MAX_PASSAGE_LEN:
                print(f"Too Long Passage")
            para_order, sent_order, font_size, style_mark = transform_tags(tags)
            # 0: others, 1: claim, 2: premise
            logits, embedding = get_sentence_results(sentences)
            sentence_mask = np.zeros((MAX_PASSAGE_LEN)).astype(np.int32)
            sentence_mask[:len(para_order)] = 1
            result["embedding"].append(zero_pad(embedding, MAX_PASSAGE_LEN))
            result["coarse_logit"].append(zero_pad(logits, MAX_PASSAGE_LEN))
            result["sentence_mask"].append(sentence_mask)
            result["paragraph_order"].append(zero_pad(para_order, MAX_PASSAGE_LEN))
            result["sentence_order"].append(zero_pad(sent_order, MAX_PASSAGE_LEN))
            result["font_size"].append(zero_pad(font_size, MAX_PASSAGE_LEN))
            result["style_mark"].append(zero_pad(style_mark, MAX_PASSAGE_LEN))
            return result
        inputs = annotation_transform(sentences, tags)
        return (content_id, title, inputs)

    @cf.op_predict
    def do_predict(self, content_id, title, inputs):
        self.start = time.time()
        result = {}
        model_inputs = {'sentence_embedding': torch.tensor(inputs['embedding']).to(torch_device),
                  'sentence_mask': torch.tensor(inputs['sentence_mask']).to(torch_device),
                  'paragraph_order': torch.tensor(inputs['paragraph_order']).to(torch_device),
                  'sentence_order': torch.tensor(inputs['sentence_order']).to(torch_device),
                  'font_size': torch.tensor(inputs['font_size']).to(torch_device),
                  'style_mark': torch.tensor(inputs['style_mark']).to(torch_device),
                  'coarse_logit': torch.tensor(inputs['coarse_logit']).to(torch_device)}
        output = self.model(**model_inputs)
        mask_1d = model_inputs["sentence_mask"]
        mask_2d = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(-1)
        major_logits = output["major_logit"].masked_select(mask_1d == 1)
        pred_sentence = output["label_logit"].max(-1)[1].masked_select(mask_1d == 1)
        n = int(math.sqrt(output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).shape[0]))
        pred_grids = output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).reshape(n, n)
        pred_grid = []
        grid_map = {0: 'No-Relation', 1: 'Co-occurence', 2: 'Co-reference', 3: 'Affiliation'}
        for i, item in enumerate(pred_grids.tolist()):
            pred_grid.append([f'{i}-{j}-{grid_map[re]}' for j, re in enumerate(item)])
        map = {0: 'Others', 1: 'Claim', 2: 'Premise', 3: 'Major'}
        d = {'major': major_logits.tolist(),
             'preds': [map[p] for p in pred_sentence.tolist()],
             'grid': pred_grid,
             'sents': inputs['sentences'],
             'titles': [title] * len(inputs['sentences']),
             'ids': [content_id] * len(inputs['sentences']),
             'srcs': [''] * len(inputs['sentences']),
             'tags': inputs['tags'],
             'users': [''] * len(inputs['sentences'])}
        df = pd.DataFrame(data=d)
        df['grid'] = df['grid'].map(lambda x: [r for r in x if 'No-Relation' not in r])
        res = res2dict(df)
        res = beautiful_claim_premise(res)
        result = []
        for t, resu in res.items():
            print(f"major claim: {resu['major']}")
            if len(resu['major']) > 0:
                result.append({
                    'tagValue': 'majorClaim',
                    'tagName': '主论点',
                    'text': ''.join(list(resu['major'][0].values())),
                    'score': max(major_logits.tolist()),
                    'extTag': {'id': f'{content_id}_majorclaim', 'sents': [str(o) for o in list(resu['major'][0].keys())]}
                  })
            for cin, cl in enumerate(resu['rclaims'][0:8]):
                print(f"claim {cin}: {cl} {resu['claims2premises'][cin]}")
                cpremises = []
                for pin, pre in enumerate(resu['rpremises']):
                    if cin in resu['premises2claims'][pin]:
                        cpremises.append({'text': ''.join(list(pre.values())), 'sents': [str(o) for o in list(pre.keys())], 'relation': -1})
                result.append({
                    'tagValue': f'claim{cin+1}',
                    'tagName': f'子论点{cin+1}',
                    'text': ''.join(list(cl.values())),
                    'score': 0,
                    'extTag': {'id': f'{content_id}_claim{cin+1}', 'sents': [str(o) for o in list(cl.keys())], 'relation': -1, 'premises': cpremises}
                })
            for pin, pre in enumerate(resu['rpremises']):
                print(f"premise {pin}: {pre} {resu['premises2claims'][pin]}")
        for re in result:
            print(json.dumps(re, indent=4, ensure_ascii=False))
        self.end = time.time()
        return result

    @cf.op_postprocessing
    def post_processing(self, session, output):
        result = {"success": "True", "error_msg": "", 'model_name': 'AntCritic',
                  'model_version': self.cur_model_version, 'elapsed': format(self.end - self.start, '.6f')}
        try:
            params_dict_str = session.params_dict_str.decode('utf-8').replace("'", "\"")
            params_dict = json.loads(params_dict_str)
            op_logger.info("params_dict: {}".format(params_dict))
            if params_dict['version'] not in session.support_versions:
                op_logger.info("model version invalid")
                result["success"] = "False"
                result["error_msg"] = "model version invalid"
                return json.dumps({"ret": 0, "result": result}, ensure_ascii=False)

            if params_dict['version'] != self.cur_model_version:
                op_logger.info("model version don't match")
                result["success"] = "False"
                result["error_msg"] = "model version don't match"
                return json.dumps({"ret": 0, "result": result}, ensure_ascii=False)

            session.result_data = output
            result['ArgumentMining'] = session.result_data
            session.cf_response = result
        except:
            s = sys.exc_info()
            op_logger.info("Error '%s' happened in file test_ops.py on line %d " % (s[1], s[2].tb_lineno))
            result["success"] = "False"
            result["error_msg"] = ("postprocess error: %s" % s[1])
            session.cf_response = result
        response = json.dumps({"ret": 0, "result": result}, ensure_ascii=False)
        return response

    def make_examples(self, style_tag, sentence):
        examples = {self.question_column_name: [style_tag], self.context_column_name: [sentence]}
        return examples

    def preprocess_function(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        ).to(torch_device)
        # for k, v in tokenized_examples.items():
        #     tokenized_examples[k] = torch.tensor(tokenized_examples[k])
        batch_sise = len(tokenized_examples["input_ids"])

        tokenized_examples["global_attention_mask"] = torch.tensor([[0] * self.max_seq_length] * batch_sise, device=torch_device)
        tokenized_examples["global_attention_mask"][:, [1, 2, 3] + list(range(4, self.max_seq_length, 200))] = 1
        return tokenized_examples


if __name__ == "__main__":
    from inference_ops import ArgumentMiningSession
    session = ArgumentMiningSession()
    config = {}
    config_file = os.path.join(BASEDIR, 'services', 'config.json')
    with open(config_file, 'r') as j:
        raw_config = json.loads(j.read())
        config['device'] = torch_device
        config['model_config'] = raw_config['ops'][0]['configs']['model_config']
        config['char_model_path'] = raw_config['ops'][0]['configs']['char_model_path']
        config['word_model_path'] = raw_config['ops'][0]['configs']['word_model_path']
        config['model_name_or_path'] = raw_config['ops'][0]['configs']['model_name_or_path']
    params = json.dumps({})
    content_id = str.encode("2018052360251343bc5ee809122949698d80a41cc72f0c93")

    # title = '刺激！指数快要翻红了！新能源车光伏怎么办？'
    # content = "<p><span style=\\\"color: rgb(219, 80, 43);\\\"><strong></strong></span></p><p><strong></strong></p><p><strong></strong></p><p>{{G_0}}上午波动结束，成长方向又一次集体挨锤，无聊了一周的行情，今天终于来点刺激的，富婆准备上班，午后发力！</p><p>\\n</p><p>{{G_1}}截止到现在，指数的调整还没有结束，一个多月了，没有像样的放量中阳线这种信号，所以对行情的基本预期就是弱势调整，并非赚钱的好行情。</p><p>\\n</p><p>不过现在也没跌破，创业板依然良好的运行在60日均线之上，这段时间正在尝试回升，再挣扎一段，先别主观看空。</p><p>\\n</p><p><span style=\\\"color: rgb(219, 80, 43);\\\"><strong>今天最需要关注的还是新能源车！</strong></span></p><p>{{G_2}}盘中对60日均线的破位，杀伤力还是挺大的，而且是一路回升，方向选择之后的结果，情绪再差一点，就容易新发新一轮的下跌。</p><p>\\n</p><p>这段时间我们的基础策略就是保持观望，中间可以有小仓位的高抛低吸，但是尽量不要大幅度操作，等方向选择结束之后再说。午后看有没有像样的修复。</p><p>\\n</p><p>{{G_3}}光伏问题倒不是很大，一路以来强势惯了，即使回踩幅度也不会太深，往下只需关注30日均线不破就行。前两天还有很多朋友觉得没有办法上车，今天出现回踩，不知道还有没有兴趣…光伏比新能源走势强很多，给出了不少利润缓冲，放松松点，别紧张。</p><p>\\n</p><p>{{G_4}}军工芯片窄幅震荡相当于回吐了昨天的上涨，波动不大，没有太多值得关注的。如果这一轮指数选择向下，那谁都跑不了，向上的话自然都会收益。</p><p>\\n</p><p>{{G_5}}白酒医疗还算不错，至少没有再继续下跌，附近认为会有止跌，打磨到了现在，已经来到企稳区域的下缘了。</p><p>\\n</p><p><strong>午间市场还是有不错的回升，刺激，下午继续，冲！</strong></p><p>{{G_6}}</p><p>个人部分持仓，其他见图：\\n</p><p> <span class=\\\"stock-name\\\" data-name=\\\"招商中证白酒指数(LOF)A\\\" data-code=\\\"161725\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_0}}</span> 3.1万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"中欧医疗健康混合A\\\" data-code=\\\"003095\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_1}}</span> 3.1万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"人保沪深300指数\\\" data-code=\\\"006600\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_2}}</span> 2.9万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"华宝券商ETF联接C\\\" data-code=\\\"007531\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_3}}</span> 2.7万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"富国中证科创创业50ETF联接C\\\" data-code=\\\"013314\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_4}}</span> 3.2万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"鹏华中证空天军工指数(LOF)C\\\" data-code=\\\"010364\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_5}}</span> 2.7万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"嘉实恒生港股通新经济指数(LOF)A\\\" data-code=\\\"501311\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_6}}</span> 2.5万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"天弘中证光伏产业指数C\\\" data-code=\\\"011103\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_7}}</span> 1.84万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"华安创业板50ETF联接C\\\" data-code=\\\"160424\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_8}}</span> 1.8万</p><p> <span class=\\\"stock-name\\\" data-name=\\\"诺安成长混合\\\" data-code=\\\"320007\\\" data-market=\\\"OF\\\" data-type=\\\"fund\\\" style=\\\"color: rgb(55, 133, 255);\\\">{{F_9}}</span> 2.0万</p><p>{{G_7}}{{G_8}}</p><p> <span class=\\\"supertalk hide\\\" data-name=\\\"白酒急跌 为何?\\\" data-code=\\\"2022081802000000495801\\\">#白酒急跌 为何?</span> </p><p>\\n</p><p>\\n</p><p>\\n</p>"
    title = '【30万实盘】煤炭领涨突破压力位！减仓吗？明天行情会如何？'
    content = "<p><strong style=\"color: rgb(219, 168, 43);\">注意了！注意了！注意了！你们想知道的在这里，白酒 、 新能源、 半导体、 医疗、 煤炭哪个板块是机会？</strong>\n</p><p><span style=\"color: rgb(219, 168, 43); background-color: rgb(255, 247, 222);\"><strong>{{G_0}}\n</strong></span></p><p><span style=\"color: rgb(219, 168, 43); background-color: rgb(255, 247, 222);\"><strong>\n</strong></span></p><p><span style=\"color: rgb(219, 168, 43); background-color: rgb(255, 247, 222);\"><strong>收益情况：昨日盈亏情况 。 <span style=\"background-color: rgb(255, 247, 222); color: rgb(219, 80, 43);\">实盘情况：八哥总资金是30万，其实这个实盘是比较少的，比不过人家百万、千万，只能做好自己的每一步，认真对待每一个板块，因为对于百万、千万，人家买入1万，就跟我们买入1000或者100是一样的道理，所以不要在意人家买入多少。</span></strong></span></p><p>{{G_1}}</p><p>\n</p><p><strong style=\"color: rgb(219, 168, 43);\">盘面情况：</strong>\n</p><p><strong style=\"color: rgb(219, 168, 43);\">{{G_2}}</strong></p><p><strong style=\"color: rgb(219, 168, 43);\"><strong>\n</strong><span style=\"font-size: 18px; color: rgb(219, 80, 43);\">煤炭板块分析：\n</span><strong>\n</strong></strong><span style=\"font-size: 18px;\"><span style=\"color: rgb(0, 0, 0);\">煤炭今天放量拉升突破了短期的压力位，这个位置从做短线重仓的角度分析的话，如果前面是逢跌左侧分批低吸进来的，做短线波段，这个位置八哥认为盈利到达自己的收益目标可以低吸高抛分批减仓，如果是按着右侧交易策略分析的话，煤炭这个板块上升趋势有所形成，今天突破了压力位向上拉升，若今天激进的话八哥觉得今天应该不少人去追突破行情，但是今天行情是有所突破了，只不过从今天走势有所回落，没能真正的突破站上压力位的这位置，说明短期的趋势只突破了一半，八哥虽然做趋势之上的板块，追突破行情，但是也要看整天的走势，稳不稳，稳的话，八哥才敢去做，不稳的话，八哥躺着休息。当然八哥没有去追煤炭这个板块，主要八哥还是担心短期煤炭这个板块明天会出现放量过后的分化行情。八哥先观察一下，若明天没有出现分化行情，走出探底回升，下方有踩着多头行情支撑没有回落，八哥会考虑右侧参与一下\n</span></span>\n{{G_3}}</p><p><span style=\"font-size: 18px; color: rgb(0, 0, 0);\"><strong>相关基金：</strong></span> <span style=\"font-size: 18px;\"><span class=\"stock-name\" data-name=\"国泰中证煤炭ETF联接C\" data-code=\"008280\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_0}}</span> </span></p><p><span class=\"stock-name\" data-name=\"金信行业优选灵活配置混合\" data-code=\"002256\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_1}}</span> 61000元。\n</p><p>\n</p><p><span class=\"stock-name\" data-name=\"招商中证白酒指数C\" data-code=\"012414\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_2}}</span> 26000元。</p><p>\n</p><p><span class=\"stock-name\" data-name=\"大成纳斯达克100指数(QDII)\" data-code=\"000834\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_3}}</span> 7300元。</p><p>\n</p><p><span class=\"stock-name\" data-name=\"中欧碳中和混合C\" data-code=\"014766\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_4}}</span> 7000元。</p><p>\n</p><p><span class=\"stock-name\" data-name=\"南方中证全指证券公司ETF联接C\" data-code=\"004070\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_5}}</span> 30000元。</p><p>{{G_4}}</p><p>{{G_5}}</p><p>{{G_6}}</p><p>\n</p><p><strong><strong><strong style=\"color: rgb(219, 168, 43);\">仓位管理\n</strong></strong></strong></p><p>因每个人资金情况不一样，所以在资金方面，八哥简单介绍一下，每个板块的资金除以10，就是每层的资金。 <span class=\"stock-name\" data-name=\"中欧碳中和混合C\" data-code=\"014766\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_6}}</span> </p><p>\n</p><p><span style=\"color: rgb(219, 168, 43);\"><strong>例如：</strong></span></p><p>白酒板块布局总金额是5万，那每层的金额是5000元。</p><p>半导体板块布局总金额是5万，那每层的金额是5000元。</p><p>医疗板块布局总金额是5万，那每层的金额是5000元。</p><p>新能源板块布局总金额是5万，那每层的金额是5000元。</p><p><span style=\"color: rgb(219, 168, 43);\">（以此类推即可，那分布四个板块的总金额是20万元，假如你总资金是10万，那么分散布局四个板块，那每个板块金额就是2.5万，那每层金额就是2500元，这是仓位的管理与细分，对于新手需要了解并熟知，当看好行情时，仓位可以适当加仓，当短期不看时，要注意控制仓位） </span></p><p> <span class=\"supertalk hide\" data-name=\"白酒急跌 为何?\" data-code=\"2022081802000000495801\">#白酒急跌 为何?</span> </p>"
    # title = '昨天减仓为主，今天加仓为主，8.18芯片医药白酒军工操作策略'
    # content = "<p> <span class=\"supertalk hide\" data-name=\"白酒急跌 为何?\" data-code=\"2022081802000000495801\">#白酒急跌 为何?</span> 各位小伙伴,下午好,盘面比较混乱，市场现在不缺钱，缺的是好的投资标的(要么估值偏高，要么看不到未来)，只能通过短线赚点小钱。还需要继续忍耐，更没必要去瞎折腾,继续做好均衡布局，把握好节奏，仓位控制在7成 ，保守的控制在6成，这样进可攻，跌可守。 </p><p>【行情分析】</p><p>昨晚美股三大指数集体收跌， <span class=\"stock-name\" data-name=\"纳斯达克\" data-code=\"IXIC.USI\" data-market=\"USI\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_0}}</span> 收跌1.25%，三大指数早盘低开后分化，截至目前， <span class=\"stock-name\" data-name=\"上证指数\" data-code=\"1A0001.SH\" data-market=\"SH\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_1}}</span> 跌0.47%， <span class=\"stock-name\" data-name=\"深证成指\" data-code=\"2A01.SZ\" data-market=\"SZ\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_2}}</span> 跌0.63% ， <span class=\"stock-name\" data-name=\"创业板指\" data-code=\"399006.SZ\" data-market=\"SZ\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_3}}</span> 跌0.07% ，盘面上个股跌多涨少，两市超3200只个股下跌。板块方面，热泵、光热、TOPCON电池、一体化压铸，市场热点在赛道股各细分方向之间快速轮动。此外，宁德时代、亿纬锂能等新能源权重股继续上涨带动创业板指走强。下跌方面，云游戏、先进封装、猪肉、农业等板块跌幅居前。</p><p><strong>操作分享：（个人操作分享，不作为投资依据，谨慎跟投）</strong></p><p>1<strong>、半导体芯片</strong></p><p> <span class=\"stock-name\" data-name=\"天弘中证芯片产业指数C\" data-code=\"012553\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_0}}</span> 半导体芯片今天确实有探底回升，再次收长下影线，唯一就是量能不足！理论上，半导体只要不跌破支撑线，就问题不大,保持仓位即可。</p><p>\n</p><p>{{G_0}}</p><p>\n</p><p><strong>2、医药板块</strong></p><p> 今日+1000元<span class=\"stock-name\" data-name=\"天弘国证生物医药ETF联接C\" data-code=\"011041\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_1}}</span> 医药板块昨天T了一部分了,反正都跌到这里了,我也不慌了,会继续坚持死磕到底的,尤其疫情下抗疫药物研发等推动生物医药发展，生物医药我感觉还是大概率有机会拉升的,这样的黄金坑.，加点仓降低下持仓成本也不错的。</p><p>{{G_1}}</p><p>\n</p><p><strong>3、白酒消费板块</strong></p><p> 今日+1000元<span class=\"stock-name\" data-name=\"天弘中证食品饮料ETF联接C\" data-code=\"001632\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_2}}</span> <span class=\"stock-name\" data-name=\"鹏华酒指数C\" data-code=\"012043\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_3}}</span> 舍得业绩下滑，白酒食品今日继续急跌，又回落不少，白酒昨天没T今天补点仓，只要不破支撑就没问题 。</p><p>{{G_2}}</p><p>\n</p><p><strong>4、海外基港股版块</strong></p><p> 今日+500元<span class=\"stock-name\" data-name=\"天弘恒生科技指数(QDII)C\" data-code=\"012349\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_4}}</span> 港股恒生科技目前跌1%左右，具体以16点收盘为准，恒生科技逢跌加一点 <span class=\"stock-name\" data-name=\"天弘越南市场股票(QDII)C\" data-code=\"008764\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_5}}</span> 参考越南VN30指数跌0.2%，具体以16点收盘为准，昨天说T后面忙忘记了，那就继续拿稳吧。</p><p>{{G_3}}</p><p>{{G_4}}</p><p>\n</p><p><strong>5、宽基+大金融板块（银行、保险、券商）：</strong></p><p> <span class=\"stock-name\" data-name=\"天弘中证全指证券公司ETF联接C\" data-code=\"008591\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_6}}</span> 券商昨天我清仓了，真的是渣男一枚,回回都是一日游,跌几天后面开始拉升,那么我们找到了策略也就不慌了,继续逢跌加仓,高抛低吸,耐心等待下一波就是.</p><p>\n</p><p>{{G_5}}</p><p>\n</p><p><strong>6、固收+、债基</strong></p><p>震荡行情，控制仓位，配置债基显得尤为重要，今年我都会以防御为主，大家可以多去财富群、讨论区及群里逛一逛，把资金轮动用起来 今日 +500元 <span class=\"stock-name\" data-name=\"新华鑫日享中短债债券A\" data-code=\"004981\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_7}}</span> ，其余债基有福利也加了一些。</p><p>{{G_6}}</p><p>\n</p><p>7、军工板块</p><p><span class=\"stock-name\" data-name=\"鹏华中证空天军工指数(LOF)C\" data-code=\"010364\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_8}}</span> 昨天加了军工，今日不动。</p><p>{{G_7}}</p><p>\n</p><p>今日加仓图如下：</p><p>{{G_8}}</p><p>相关持仓图如下，发文需要：</p><p>\n</p><p>{{G_9}}</p><p>\n</p><p>仅供参考，文中所涉及的产品非推荐，投资须谨慎，请大家一定要慎重选择！</p><p>码字不易，记得点赞+关注哦，点赞关注的小伙伴，好运连连</p><p>\n</p><p>{{G_10}}</p><p>\n</p><p>\n</p><p>\n</p><p>\n</p><p>\n</p>"

    title = str.encode(title)
    content = str.encode(content)
    argument_mining = ArgumentMining()
    argument_mining.init_model(raw_config=config)
    content_id, title, inputs = argument_mining.pre_processing(session, content_id, title, content, params)
    result = argument_mining.do_predict(content_id, title, inputs)
