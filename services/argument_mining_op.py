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
from preprocess.second_stage import transform_tags, zero_pad, generate_char_tokenizer, generate_word_tokenizer, generate_models
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
    return config

@cf.op("argument_mining", tf.string)
class ArgumentMining:
    @cf.op_init
    def init_model(self, raw_config):
        self.cur_model_version = "latest"
        self.config = convert_config(raw_config)
        cfg = get_cfg_defaults()
        self.model_config = cfg.model
        print('self.model_config: ', self.model_config)
        self.char_model_path = self.config.char_model_path
        self.word_model_path = self.config.word_model_path
        self.model_name_or_path = self.config.model_name_or_path
        # todo: 加载word&char&second_stage的预测模型
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
        op_logger.info("Completed loading model in {}......".format(self.model_name_or_path))
        self._init_model(self.model_config)
        print('self.model: ', self.model)
        state_dict = torch.load(self.model_name_or_path, map_location=torch.device(torch_device))
        parameters = state_dict['model_parameters']
        self.model.module.load_state_dict(parameters)
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
        start = time.time()
        result = {}
        model_inputs = {'sentence_embedding': torch.tensor(inputs['embedding']).to(torch_device),
                  'sentence_mask': torch.tensor(inputs['sentence_mask']).to(torch_device),
                  'paragraph_order': torch.tensor(inputs['paragraph_order']).to(torch_device),
                  'sentence_order': torch.tensor(inputs['sentence_order']).to(torch_device),
                  'font_size': torch.tensor(inputs['font_size']).to(torch_device),
                  'style_mark': torch.tensor(inputs['style_mark']).to(torch_device),
                  'coarse_logit': torch.tensor(inputs['coarse_logit']).to(torch_device)}
        # print('model_inputs: ', model_inputs)
        output = self.model(**model_inputs)
        mask_1d = model_inputs["sentence_mask"]
        mask_2d = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(-1)
        major_logits = output["major_logit"].masked_select(mask_1d == 1)
        pred_sentence = output["label_logit"].max(-1)[1].masked_select(mask_1d == 1)
        n = int(math.sqrt(output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).shape[0]))
        pred_grids = output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).reshape(n, n)
        pred_label = torch.tensor(pred_sentence)
        major_logits = torch.tensor(major_logits)
        pred_grid = []
        grid_map = {0: 'No-Relation', 1: 'Co-occurence', 2: 'Co-reference', 3: 'Affiliation'}
        for i, item in enumerate(pred_grids.tolist()):
            pred_grid.append([f'{i}-{j}-{grid_map[re]}' for j, re in enumerate(item)])
        map = {0: 'Others', 1: 'Claim', 2: 'Premise', 3: 'Major'}
        d = {'major': major_logits.tolist(),
             'preds': [map[p] for p in pred_label.tolist()],
             'grid': pred_grid,
             'sents': inputs['sentences'],
             'titles': [title] * len(inputs['sentences']),
             'ids': [content_id] * len(inputs['sentences']),
             'srcs': [''] * len(inputs['sentences']),
             'tags': inputs['tags'],
             'users': [''] * len(inputs['sentences'])}
        df = pd.DataFrame(data=d)
        df['grid'] = df['grid'].map(lambda x: [r for r in x if 'No-Relation' not in r])
        print(df)
        res = res2dict(df)
        res = beautiful_claim_premise(res)
        for t, resu in res.items():
            print(f"===========major claim===========")
            print(f"major claim: {resu['major']}")
            print("===========raw_claim===========")
            for cin, cl in enumerate(resu['rclaims']):
                print(f"claim {cin}: {cl} {resu['claims2premises'][cin]}")
            print("===========raw_premise===========")
            for pin, pre in enumerate(resu['rpremises']):
                print(f"premise {pin}: {pre} {resu['premises2claims'][pin]}")
        end = time.time()
        result['clean_elapsed'] = format(end - start, '.6f')
        return df

    @cf.op_postprocessing
    def post_processing(self, session, output):
        result = {"success": "True", "error_msg": "", 'model_name': 'argument_mining',
                  'model_version': self.cur_model_version}
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
            result['data'] = session.result_data
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
    config_file = 'config.json'
    with open(config_file, 'r') as j:
        raw_config = json.loads(j.read())
        print(raw_config)
        config['device'] = torch_device
        config['char_model_path'] = raw_config['ops'][0]['configs']['char_model_path']
        config['word_model_path'] = raw_config['ops'][0]['configs']['word_model_path']
        config['model_name_or_path'] = raw_config['ops'][0]['configs']['model_name_or_path']
    params = json.dumps({})
    content_id = str.encode("2018052360251343bc5ee809122949698d80a41cc72f0c93")

    title = '白酒板块领跌行情分析'
    content = "<p></p>{{G_0}}<p></p><p>行情回顾：上证指数下跌0.46%报3277.54，深圳成指下跌0.62%，创业板指下跌0.08%。涨幅靠前的板块有电机，光伏设备，光伏等电子，消费电子，半导体，电子元件电源设备汽车零部件，电池等行业。跌幅靠前的板块有贵金属，旅游酒店，酿酒行业，保险，煤炭行业等板块。</p><p>资金流向：北上资金流出46.13亿，主力资金流出312.37亿，两市合计成交9869亿元与上一交易日减少2.49%，盘面上1974只个股上涨，涨停79只，2778只个股下跌。</p>{{G_1}}<p></p><p>行情分析：今日白酒行业跌幅靠前主要是舍得酒业发布半年业绩报显示净利润同比下滑29.67%。进入7月中下旬白酒上市公司陆续公布业绩，其中顺鑫控股，水井坊等多家公司业绩不及预期。数据显示近期以来多个机构和明星基金都有减持白酒相关个股。白酒板块是消费行业中非常重要的细分行业，毛利率高，需求量大业绩表现也是非常亮眼的，上半年受多重因素的影响白酒板块表现弱势，带动消费行业表现一般。但是我们必须要相信突发情况肯定会得到有效控制，当前大幅下跌白酒消费行业的机遇大于风险。</p><p></p><p>感谢小伙伴的点赞、留言、关注，投资路上我们共同成长进步。</p><p></p>"

    # title = '昨天减仓为主，今天加仓为主，8.18芯片医药白酒军工操作策略'
    # content = "<p> <span class=\"supertalk hide\" data-name=\"白酒急跌 为何?\" data-code=\"2022081802000000495801\">#白酒急跌 为何?</span> 各位小伙伴,下午好,盘面比较混乱，市场现在不缺钱，缺的是好的投资标的(要么估值偏高，要么看不到未来)，只能通过短线赚点小钱。还需要继续忍耐，更没必要去瞎折腾,继续做好均衡布局，把握好节奏，仓位控制在7成 ，保守的控制在6成，这样进可攻，跌可守。 </p><p>【行情分析】</p><p>昨晚美股三大指数集体收跌， <span class=\"stock-name\" data-name=\"纳斯达克\" data-code=\"IXIC.USI\" data-market=\"USI\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_0}}</span> 收跌1.25%，三大指数早盘低开后分化，截至目前， <span class=\"stock-name\" data-name=\"上证指数\" data-code=\"1A0001.SH\" data-market=\"SH\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_1}}</span> 跌0.47%， <span class=\"stock-name\" data-name=\"深证成指\" data-code=\"2A01.SZ\" data-market=\"SZ\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_2}}</span> 跌0.63% ， <span class=\"stock-name\" data-name=\"创业板指\" data-code=\"399006.SZ\" data-market=\"SZ\" data-type=\"stock\" style=\"color: rgb(55, 133, 255);\">{{S_3}}</span> 跌0.07% ，盘面上个股跌多涨少，两市超3200只个股下跌。板块方面，热泵、光热、TOPCON电池、一体化压铸，市场热点在赛道股各细分方向之间快速轮动。此外，宁德时代、亿纬锂能等新能源权重股继续上涨带动创业板指走强。下跌方面，云游戏、先进封装、猪肉、农业等板块跌幅居前。</p><p><strong>操作分享：（个人操作分享，不作为投资依据，谨慎跟投）</strong></p><p>1<strong>、半导体芯片</strong></p><p> <span class=\"stock-name\" data-name=\"天弘中证芯片产业指数C\" data-code=\"012553\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_0}}</span> 半导体芯片今天确实有探底回升，再次收长下影线，唯一就是量能不足！理论上，半导体只要不跌破支撑线，就问题不大,保持仓位即可。</p><p>\n</p><p>{{G_0}}</p><p>\n</p><p><strong>2、医药板块</strong></p><p> 今日+1000元<span class=\"stock-name\" data-name=\"天弘国证生物医药ETF联接C\" data-code=\"011041\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_1}}</span> 医药板块昨天T了一部分了,反正都跌到这里了,我也不慌了,会继续坚持死磕到底的,尤其疫情下抗疫药物研发等推动生物医药发展，生物医药我感觉还是大概率有机会拉升的,这样的黄金坑.，加点仓降低下持仓成本也不错的。</p><p>{{G_1}}</p><p>\n</p><p><strong>3、白酒消费板块</strong></p><p> 今日+1000元<span class=\"stock-name\" data-name=\"天弘中证食品饮料ETF联接C\" data-code=\"001632\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_2}}</span> <span class=\"stock-name\" data-name=\"鹏华酒指数C\" data-code=\"012043\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_3}}</span> 舍得业绩下滑，白酒食品今日继续急跌，又回落不少，白酒昨天没T今天补点仓，只要不破支撑就没问题 。</p><p>{{G_2}}</p><p>\n</p><p><strong>4、海外基港股版块</strong></p><p> 今日+500元<span class=\"stock-name\" data-name=\"天弘恒生科技指数(QDII)C\" data-code=\"012349\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_4}}</span> 港股恒生科技目前跌1%左右，具体以16点收盘为准，恒生科技逢跌加一点 <span class=\"stock-name\" data-name=\"天弘越南市场股票(QDII)C\" data-code=\"008764\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_5}}</span> 参考越南VN30指数跌0.2%，具体以16点收盘为准，昨天说T后面忙忘记了，那就继续拿稳吧。</p><p>{{G_3}}</p><p>{{G_4}}</p><p>\n</p><p><strong>5、宽基+大金融板块（银行、保险、券商）：</strong></p><p> <span class=\"stock-name\" data-name=\"天弘中证全指证券公司ETF联接C\" data-code=\"008591\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_6}}</span> 券商昨天我清仓了，真的是渣男一枚,回回都是一日游,跌几天后面开始拉升,那么我们找到了策略也就不慌了,继续逢跌加仓,高抛低吸,耐心等待下一波就是.</p><p>\n</p><p>{{G_5}}</p><p>\n</p><p><strong>6、固收+、债基</strong></p><p>震荡行情，控制仓位，配置债基显得尤为重要，今年我都会以防御为主，大家可以多去财富群、讨论区及群里逛一逛，把资金轮动用起来 今日 +500元 <span class=\"stock-name\" data-name=\"新华鑫日享中短债债券A\" data-code=\"004981\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_7}}</span> ，其余债基有福利也加了一些。</p><p>{{G_6}}</p><p>\n</p><p>7、军工板块</p><p><span class=\"stock-name\" data-name=\"鹏华中证空天军工指数(LOF)C\" data-code=\"010364\" data-market=\"OF\" data-type=\"fund\" style=\"color: rgb(55, 133, 255);\">{{F_8}}</span> 昨天加了军工，今日不动。</p><p>{{G_7}}</p><p>\n</p><p>今日加仓图如下：</p><p>{{G_8}}</p><p>相关持仓图如下，发文需要：</p><p>\n</p><p>{{G_9}}</p><p>\n</p><p>仅供参考，文中所涉及的产品非推荐，投资须谨慎，请大家一定要慎重选择！</p><p>码字不易，记得点赞+关注哦，点赞关注的小伙伴，好运连连</p><p>\n</p><p>{{G_10}}</p><p>\n</p><p>\n</p><p>\n</p><p>\n</p><p>\n</p>"

    title = str.encode(title)
    content = str.encode(content)
    argument_mining = ArgumentMining()
    argument_mining.init_model(raw_config=config)
    content_id, title, inputs = argument_mining.pre_processing(session, content_id, title, content, params)
    result = argument_mining.do_predict(content_id, title, inputs)
