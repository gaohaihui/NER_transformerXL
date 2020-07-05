# -*- coding: utf-8 -*-


import torch

from config import Config
from utils import (decode_tags_io, decode_tags_bio,decode_tags_biose,
                   get_text_line_feature, id2tag,
                   load_vocab, load_model, load_train_model)
from model.transXL import TransXL

def test():
  """
  Background test method.
  """
  config = Config()
  device = torch.device("cpu")
  # if config.use_cuda:
  #   device = torch.cuda.set_device(config.gpu[0])
  print('loading corpus')
  vocab_mask = load_vocab(config.vocab)
  label_dic = load_vocab(config.label_file)
  model = TransXL(tag_vocab=dict([val, key] for key, val in label_dic.items()),
          bert_config=config.bert_path)
  model = load_train_model(model)
  #   model.crf.use_cuda = False
  model.to(device)
  model.eval()

  while True:
    line = input("input sentence, please:")
    mems = None
    feature = get_text_line_feature(line, vocab_mask, max_length=512)
    input_id = torch.LongTensor(feature.input_id).unsqueeze(0)
    ids, mems = model.predict(input_id, mems)["pred"]
    ids = ids.squeeze(0).numpy().tolist()
    pre_tags = id2tag(label_dic, ids)
    if config.label_mode == "BIOSE":
      result = decode_tags_io(line, pre_tags[1:-1])
    else:
      result = decode_tags_bio(line, pre_tags[1:-1])
    print(result)


if __name__ == '__main__':
  test()
