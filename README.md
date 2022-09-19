# AntCritic

论文: [AntCritic: Argument Mining for Free-Form and Visually-Rich Financial Comments](http://arxiv.org/abs/2208.09612)

数据:
* 训练集: antcritic/train.1.csv
* 测试集: antcritic/test.1.csv
* 验证集: antcritic/dev.1.csv

使用的预训练模型:
* 词级别: pretrained_model/paraphrase-xlm-r-multilingual-v1
* 字级别: pretrained_model/FinBERT_L-12_H-768_A-12_pytorch

使用antcritic数据集finetune的预训练模型:
* 词级别: checkpoints/char/models-9.pt
* 字级别: checkpoints/word/models-12.pt

使用antcritic数据集训练的论文中Figure4所示模型:
* checkpoints/GRU/models-7.pt
