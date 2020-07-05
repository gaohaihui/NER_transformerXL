# -*- coding: utf-8 -*-

import os
import random
import torch

from torch.utils.data import Dataset
from utils import InputFeatures, iob_iobes


class BioData(Dataset):
  """
  BIO data structure for training NER model.
  """
  def __init__(self, root, mode,
    max_length, label_dic, vocab, label_mode="BIOSE"):
    """
    BIO data structure for training NER model.

    Args:
      root(str): Train data root path.
      mode(str): Train mode or test mode.
      max_length(int): Character max length.
      label_dic(dict): Label and index map.
      vocab(dict): All character and index map.
      label_mode(str): Label mode, contain BIOSE or BIO.
    """
    super(BioData, self).__init__()
    self.root = root
    self.mode = mode
    self.max_length = max_length
    self.label_dic = label_dic
    self.vocab = vocab
    self.data = []
    self.label_mode = label_mode
    print("label mode: " + label_mode)
    if self.mode == 'train':
      self.data = self.read_data("example.train")
    elif self.mode == 'dev':
      self.data = self.read_data("example.dev")
    else:
      self.data = self.read_data("example.test")
    self.input_ids = torch.LongTensor([temp.input_id for temp in self.data])
    self.masks = torch.LongTensor([temp.input_mask for temp in self.data])
    self.tags = torch.LongTensor([temp.label_id for temp in self.data])

  def __len__(self):
    """
    Gets data length.

    Returns:
      len(int): Entire data length.
    """
    return len(self.data)

  def __getitem__(self, idx):
    """
    Gets data by idx.

    Args:
      idx(int): Character id.

    Returns:
      input_ids(int): Character.
      masks(int): Mask label.
      tags(int): Label index.
    """
    return self.input_ids[idx], self.masks[idx], self.tags[idx]

  def read_data(self, path):
    """
    Reads data from path.

    Args:
      path(str): Data path.

    Returns:
      result(list): List of train data has been formated.
    """
    full_path = os.path.join(self.root, path)
    with open(full_path, encoding='utf-8') as file:
      content = file.readlines()
    result = []
    tokens = []
    labels = []
    i = 0
    for line in content:
      i+=1
      text = line.strip().split(' ')
      if len(text) != 2:
        text = text[0].split("\t")
        if len(text) != 2:
          continue
      text, label = text[0], text[1]
      tokens.append(text)
      labels.append(label)
      if len(tokens) == self.max_length:
        tokens_f = tokens
        if self.label_mode == "BIOSE":
          labels = iob_iobes(labels)
        label_f = labels
        input_ids = [int(self.vocab[i]) if i in self.vocab else
                     int(self.vocab['[UNK]']) for i in tokens_f]
        label_ids = [self.label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == self.max_length
        assert len(input_mask) == self.max_length
        assert len(label_ids) == self.max_length
        feature = InputFeatures(
          input_id=input_ids, input_mask=input_mask, label_id=label_ids)
        result.append(feature)
        tokens = []
        labels = []
    if tokens:
      tokens_f = tokens
      if self.label_mode == "BIOSE":
        labels = iob_iobes(labels)
      label_f = labels
      input_ids = [int(self.vocab[i]) if i in self.vocab
                   else int(self.vocab['[UNK]']) for i in tokens_f]
      label_ids = [self.label_dic[i] for i in label_f]
      input_mask = [1] * len(input_ids)
      while len(input_ids) < self.max_length:
        input_ids.append(int(self.vocab["[PAD]"]))
        input_mask.append(0)
        label_ids.append(self.label_dic['<pad>'])
      assert len(input_ids) == self.max_length
      assert len(input_mask) == self.max_length
      assert len(label_ids) == self.max_length
      feature = InputFeatures(
        input_id=input_ids, input_mask=input_mask, label_id=label_ids)
      result.append(feature)
    return result
