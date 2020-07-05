# -*- coding: utf-8 -*-

import traceback
import torch

from config import Config
from utils import (decode_tags_biose, decode_tags_bio,
                   get_text_line_feature, id2tag,
                   load_vocab, load_model, load_train_model)
from model.bert_lstm_crf import BERT_LSTM_CRF


try:
	print('loading corpus')
	config = Config()
	use_gpu = config.use_cuda
	vocab_mask = load_vocab(config.vocab)
	label_dict = load_vocab(config.label_file)
	tagset_size = len(label_dict)
	label_mode = config.label_mode
	device_ids = config.gpu
	batch_size = 8
	max_str_len = 512
	model = BERT_LSTM_CRF(
		config.bert_path, tagset_size, config.bert_embedding,
		config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio,
		dropout1=config.dropout1, use_cuda=config.use_cuda, config=config)
	model = load_train_model(model)
	if config.use_cuda and len(config.gpu) > 1:
		if not isinstance(model, torch.nn.DataParallel):
			model = torch.nn.DataParallel(model, device_ids=config.gpu)
	model.eval()
except Exception:
	traceback.format_exc()
	pass