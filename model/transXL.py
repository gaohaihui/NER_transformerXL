# -*- coding: utf-8 -*-

from fastNLP.modules import ConditionalRandomField, allowed_transitions
from pytorch_pretrained_bert import BertModel
from torch import nn
import torch
import torch.nn.functional as F
from transformers.modeling_transfo_xl import TransfoXLConfig, TransfoXLModel

class TransXL(nn.Module):
    def __init__(self, tag_vocab, bert_config, bi_embed=None):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()

        self.embed = BertModel.from_pretrained(bert_config)
        embed_size = self.embed.embeddings.word_embeddings.weight.shape[1]
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size
        self.configuration = TransfoXLConfig(d_model=768, d_head=16, n_head=16, n_layer=4, mem_len=1000)
        self.xl_model = TransfoXLModel(self.configuration)
        self.liner = nn.Linear(768, len(tag_vocab))
        # trans = allowed_transitions(tag_vocab, include_start_end=True, encoding_type = "bioes")
        #TODO: trans 为限制转移的数组，非常有用，过后加上
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=None)

    def _forward(self, sentence, target=None, mems=None):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        mask = sentence.ne(0)

        embeds, _ = self.embed(sentence, attention_mask=None,
                                   output_all_encoded_layers=False)
        trans_out = self.xl_model(None, mems, inputs_embeds=embeds)[:2]
        feats, mems = trans_out[0], trans_out[1]
        feats = self.liner(feats.contiguous().view(-1, 768))
        feats = feats.contiguous().view(batch_size, seq_length, -1)
        logits = F.log_softmax(feats, dim=-1)
        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': [paths, mems]}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': [loss, mems]}

    def forward(self, chars, target=None, mems=None):
        return self._forward(chars, target, mems)

    def predict(self, chars, mems=None):
        return self._forward(chars, target=None, mems=mems)
