# -*- coding: utf-8 -*-

import logging
import os
import torch

from TransXL.metrics import classification_report
from TransXL.metrics import f1_score


class InputFeatures(object):
  """
  Input feature for BERT.
  """
  def __init__(self, input_id, label_id, input_mask):
    """
    Args:
      input_id(int): Id of character.
      label_id(str): Id of label for classify.
      input_mask(int): Mask flag, 0 or 1.
    """
    self.input_id = input_id
    self.label_id = label_id
    self.input_mask = input_mask


def load_vocab(vocab_file):
  """
  Loads a vocabulary file into a dictionary.

  Args:
    vocab_file(str): file path.

  Returns:
    vocab(dict): Character and index map.
  """
  vocab = {}
  index = 0
  with open(vocab_file, "r", encoding="utf-8") as reader:
    while True:
      token = reader.readline()
      if not token:
          break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def get_text_line_feature(sentense, vocab, max_length=512):
  """
  Gets text feature.

  Args:
    sentense(str): Input string.
    vocab(dict): Character and index map.
    max_length(int): Input content max length.

  Returns:
    feature(InputFeatures): Feature of sentense.
  """
  tokens = list(sentense.replace(" ", ""))
  tokens_f = ['[CLS]'] + tokens + ['[SEP]']
  input_ids = [
    int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_length:
    input_ids.append(0)
    input_mask.append(0)
  feature = InputFeatures(
    input_id=input_ids, input_mask=input_mask, label_id=None)
  return feature


def id2tag(tag_id, ids):
  """
  Index to tag.

  Args:
    tag_id(dict): Label and index map.
    ids(list): List of tag.

  Returns:
    tags(list): Tags of ids.
  """
  id_tag = dict([val, key] for key, val in tag_id.items())
  tags = [id_tag.get(id) for id in ids]
  return tags


def decode_tags_biose(text, tags):
  """
  Decodes BIOSE to string with tag.

  Args:
    text(str): Origin string.
    tags(list): Tag of string.

  Returns:
    item(dict): Decoded result of current string.
  """
  item = {"string": text, "entities": []}
  entity_name = ""
  entity_start = 0
  idx = 0
  for char, tag in zip(text, tags):
    if tag[0] == "S":
      item["entities"].append(
        {"word": char, "start": idx, "end": idx+1, "type": tag[2:]})
    elif tag[0] == "B":
      entity_name += char
      entity_start = idx
    elif tag[0] == "I":
      if idx - 1 >= 0:
        if tags[idx-1] == "O":
          entity_start = idx
      entity_name += char
    elif tag[0] == "E":
      if entity_name:
        entity_name += char
        if len(set([t[2:] for t in tags[entity_start:idx +1]])) == 1:
          item["entities"].append(
            {"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
        entity_name = ""
    else:
      entity_name = ""
      entity_start = idx
    idx += 1
  return item


def decode_tags_io(text, tags):
  """
  Decodes IO to string with tag.

  Args:
    text(str): Origin string.
    tags(list): Tag of string.

  Returns:
    item(dict): Decoded result of current string.
  """
  item = {"string": text, "entities": []}
  entity_name = ""
  entity_start = 0
  last_tag = ""
  for idx, (char, tag) in enumerate(zip(text, tags)):
    if tag not in ["O", "<pad>", "<start>", "<eos>"]:
      entity_tag = tag[2:]
    else:
      entity_tag = "O"
    if entity_tag != "O":
      # entity_tag = tag[2:]
      if last_tag != entity_tag:
        entity_name = char
        entity_start = idx
      else:
        entity_name += char
      if idx == len(tags)-1 or tags[idx+1][2:] != entity_tag:
        if len(set([t[2:] for t in tags[entity_start:idx + 1]])) == 1:
          if text[entity_start:idx + 1] != entity_name:
            print("tttttttttttt")
          item["entities"].append(
            {"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
        entity_name = ""
    else:
      entity_name = ""
      entity_start = idx
    last_tag = entity_tag
  return item


def compute(pre, groud_true):
  """
  Computes correct.

  Args:
    pre(list): Predicted tags.
    groud_true(list): Tags of groud true.

  Returns:
    correct_num(int): Num of correct.
  """
  correct_num = 0
  use_num = 0
  for p, t in zip(pre, groud_true):
    if p == "o":
      continue
    use_num += 1
    if p == t:
      correct_num += 1
  return correct_num


def evaluate(true_tags, pred_tags, verbose=False):
  """
  Evaluate method.

  Args:
    true_tags(list): Tags of groud true.
    pred_tags(list): Tags of predict.
    verbose(bool): Whether need verbose f1 report.

  Returns:
    metrics(dict): Metrics of evaluation.
  """
  metrics_all = f1_score(true_tags, pred_tags)
  # logger.debug("Metrics: {}".format(metrics_all))
  metrics = metrics_all.get("avg")

  metrics_str = "; ".join("{}: {:05.2f}".format(k, v)
    for k, v in metrics.items())
  logger.debug("- {} metrics: ".format(metrics_str))
  print("- {} metrics: ".format(metrics_str))
  if verbose:
    report = classification_report(true_tags, pred_tags)
    logger.debug(report)
  return metrics


def decode_tags_bio(text, tags):
  """
  Decodes BIO to string with tag.

  Args:
    string(str): Origin string.
    tags(list): Tag of string.

  Returns:
    item(dict): Decoded result of current string.
  """
  item = {"text": text, "entities": []}
  entity_name = ""
  entity_start = 0
  count = 0
  entity_tag = ""
  for c_idx in range(len(text)):
    c, tag = text[c_idx], tags[c_idx]
    if c_idx < len(tags)-1:
      tag_next = tags[c_idx+1]
    else:
      tag_next = ''
    if tag[0] == 'B':
      entity_tag = tag[2:]
      entity_name = c
      entity_start = count
      if tag_next[2:] != entity_tag:
        item["entities"].append({"word": c, "start": count, "end": count + 1, "type": tag[2:]})
    elif tag[0] == "I":
      if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
        tags[c_idx] = 'O'
        pass
      else:
        entity_name = entity_name + c
        if tag_next[2:] != entity_tag:
          item["entities"].append(
            {"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
          entity_name = ''
    count += 1
  return item


def save_model(model, epoch, path='result', **kwargs):
  """
  Saves model.

  Args:
    model(BERT_LSTM_CRF): Model of BERT_LSTM_CRF.
    epoch(int): Number of epoch.
    path(str): Model saved path.
  """
  if not os.path.exists(path):
    os.mkdir(path)
  if kwargs.get('name', None) is None:
    f1 = 0.0
    if kwargs.get("f1"):
      f1 = kwargs.get("f1")
    name = ('epoch_' + str(epoch) +
      "_loss_" + str('%.3f' % kwargs.get("loss")) + "f1_" + str(f1) + ".pkl")
    full_name = os.path.join(path, name)
    if isinstance(model, torch.nn.DataParallel):
      model = model.module
    torch.save(model, full_name)
    print('Saved model at epoch {} successfully'.format(epoch))
    with open('{}/checkpoint'.format(path), 'w') as file:
      file.write(name)
      print('Write to checkpoint')
    file_list = os.listdir(path)
    file_list = sorted(
      file_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    surplus = file_list[:-6]
    for file_name in surplus:
      os.remove(os.path.join(path, file_name))


def save_train_model(model, epoch, path='result', **kwargs):
  """
  Saves model.

  Args:
    model(BERT_LSTM_CRF): Model of BERT_LSTM_CRF.
    epoch(int): Number of epoch.
    path(str): Model saved path.
  """
  if not os.path.exists(path):
    os.mkdir(path)
  if kwargs.get('name', None) is None:
    f1 = 0.0
    if kwargs.get("f1"):
      f1 = kwargs.get("f1")
    name = ('epoch_' + str(epoch) +
      "_loss_" + str('%.3f' % kwargs.get("loss")) + "f1_" + str(f1) + ".pth")
    full_name = os.path.join(path, name)
    if isinstance(model, torch.nn.DataParallel):
      model = model.module

    print('Saved model at epoch {} successfully'.format(epoch))
    with open('{}/checkpoint'.format(path), 'w') as file:
      file.write(name)
      print('Write to checkpoint')
    file_list = os.listdir(path)
    file_list = sorted(
      file_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    surplus = file_list[:-1]
    for file_name in surplus:
      os.remove(os.path.join(path, file_name))
    torch.save(model.state_dict(), full_name)

def load_model(path='result', **kwargs):
  """
  Loads model.

  Args:
    model(BERT_LSTM_CRF): Model of BERT_LSTM_CRF.
    path(str): Model saved path.

  Returns:
    model(BERT_LSTM_CRF): Model of BERT_LSTM_CRF after added weights.
  """
  if kwargs.get('name', None) is None:
    with open('{}/checkpoint'.format(path)) as file:
      content = file.read().strip()
      name = os.path.join(path, content)
  else:
    name = kwargs['name']
    name = os.path.join(path, name)
  model = torch.load(name, map_location=lambda storage, loc: storage)
  print('load model {} successfully'.format(name))
  return model


def load_train_model(model, path='result', **kwargs):
  if kwargs.get('name', None) is None:
    with open('{}/checkpoint'.format(path)) as file:
      content = file.read().strip()
      name = os.path.join(path, content)
  else:
    name = kwargs['name']
    name = os.path.join(path, name)
  model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
  print('load model {} successfully'.format(name))
  return model

def iob_iobes(tags):
  """
  BIO mode to BIOSE.

  Args:
    tags(list): BIO tags.

  Returns:
    new_tags(list): BIOSE tags.
  """
  new_tags = []
  for i, tag in enumerate(tags):
    if tag == 'O':
      new_tags.append(tag)
    elif tag.split('-')[0] == 'B':
      if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
        new_tags.append(tag)
      else:
        new_tags.append(tag.replace('B-', 'S-'))
    elif tag.split('-')[0] == 'I':
      if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
        new_tags.append(tag)
      else:
        new_tags.append(tag.replace('I-', 'E-'))
    else:
      raise Exception('Invalid IOB format!')
  return new_tags


def iobes_iob(tags):
  """
  BIOSE mode to BIO.

  Args:
    tags(list): BIOSE tags.

  Returns:
    new_tags(list): BIO tags.
  """
  new_tags = []
  for i, tag in enumerate(tags):
    if tag.split('-')[0] == 'B':
      new_tags.append(tag)
    elif tag.split('-')[0] == 'I':
      new_tags.append(tag)
    elif tag.split('-')[0] == 'S':
      new_tags.append(tag.replace('S-', 'B-'))
    elif tag.split('-')[0] == 'E':
      new_tags.append(tag.replace('E-', 'I-'))
    elif tag.split('-')[0] == 'O':
      new_tags.append(tag)
    else:
      raise Exception('Invalid format!')
  return new_tags


def get_logger():
  """
  Gets logger obj.

  Returns:
    logger(loggerClass): Logger util.
  """
  logger = logging.getLogger("train")
  logger.setLevel(logging.DEBUG)
  # Builds filehandler write log, above debug level.
  fh = logging.FileHandler("train.log")
  fh.setLevel(logging.DEBUG)
  # Builds streamhandler CMD window，above error.
  ch = logging.StreamHandler()
  ch.setLevel(logging.ERROR)
  # Sets log format.
  formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  ch.setFormatter(formatter)
  fh.setFormatter(formatter)
  # Adds handler to logger obj.
  logger.addHandler(ch)
  logger.addHandler(fh)
  return logger


logger = get_logger()