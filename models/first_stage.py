import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import BertModel

from utils.helper import masked_operation


class WordBasedExtractor(nn.Module):
    def __init__(self, word_path):
        super().__init__()
        self.core = SentenceTransformer(word_path)

    def forward(self, word_id, word_mask):
        # word_id: (B, L), word_mask: (B, L)
        results = self.core({"input_ids": word_id, "attention_mask": word_mask})
        # all_states = results["all_layer_embeddings"]
        # all_embedding = all_states[0] + all_states[-1]
        all_embedding = results["token_embeddings"]
        overall_embedding = masked_operation(all_embedding, word_mask, dim=1, operation="mean")
        return overall_embedding  # (B, D)

    def save_model(self, save_path):
        self.core.save(save_path)

class CharBasedExtractor(nn.Module):
    def __init__(self, char_path):
        super().__init__()
        self.core = BertModel.from_pretrained(char_path)

    def forward(self, char_id, char_mask):
        # char_id: (B, L), char_mask: (B, L)
        all_states = self.core(input_ids=char_id, attention_mask=char_mask, output_hidden_states=True).hidden_states
        all_embedding = all_states[1] + all_states[-1]
        overall_embedding = masked_operation(all_embedding, char_mask, dim=1, operation="mean")
        return overall_embedding

    def save_model(self, save_path):
        self.core.save_pretrained(save_path)


class FirstStageModel(nn.Module):
    def __init__(self, module_cfg, use_word, **kwargs):
        super().__init__()
        self.use_word = use_word
        if not use_word:
            self.extractor = CharBasedExtractor(module_cfg["char_path"])
        else:
            self.extractor = WordBasedExtractor(module_cfg["word_path"])
        self.out_linear = nn.Linear(768, 3)

    def forward(self, char_id, char_mask, word_id, word_mask):
        if self.use_word:
            input_id, input_mask = word_id, word_mask
        else:
            input_id, input_mask = char_id, char_mask
        origin = self.extractor(input_id, input_mask)
        logits = self.out_linear(origin)
        return logits, origin










