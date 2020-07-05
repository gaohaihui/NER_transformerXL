# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.optim as optim
import time

from bio_data import BioData
from config import Config
from model.transXL import TransXL
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import (evaluate, id2tag, load_vocab,
                   load_train_model, logger, save_model, load_model, save_train_model)
from viz_util import VisdomLinePlotter





plotter = VisdomLinePlotter(env_name='train_TransXL')

def train():
  """
  Train method.
  """
  config = Config()
  logger.debug(f'当前设置为: {config}')
  suffix_name = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  device = torch.device("cpu")
  if config.use_cuda:
    torch.cuda.set_device(config.gpu[0])
    device = torch.device(config.gpu[0])
  vocab = load_vocab(config.vocab)
  label_dic = load_vocab(config.label_file)
  tagset_size = len(label_dic)
  # train data build
  logger.debug(f'Loading train data, path is {config.root}.')
  train_dataset = BioData(
    config.root, "train", config.max_length, label_dic, vocab, config.label_mode)
  train_size = int(0.9 * len(train_dataset))
  train_global_step = train_size/config.batch_size
  logger.debug(f'Training global step:  {train_global_step}.')
  dev_size = len(train_dataset) - train_size
  train_dataset, dev_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, dev_size])
  train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=4)
  # dev data build
  dev_loader = DataLoader(
    dev_dataset, shuffle=False, batch_size=1, num_workers=2)
  # test data build
  test_dataset = BioData(
    config.root, "test", config.max_length, label_dic, vocab, config.label_mode)
  test_loader = DataLoader(
    test_dataset, shuffle=False, batch_size=1, num_workers=2)

  model = TransXL(tag_vocab=dict([val,key] for key,val in label_dic.items()),
                bert_config=config.bert_path)
  assert config.load_path is not None
  if config.load_model:
    logger.debug(f"Loads model {config.load_path}, continue training")
    model = load_train_model(model)
  optimizer = getattr(optim, config.optim)
  optimizer = optimizer(model.parameters(), lr=config.lr,
                        weight_decay=config.weight_decay)
  if config.use_cuda and len(config.gpu) > 1:
    if not isinstance(model, torch.nn.DataParallel):
      model = torch.nn.DataParallel(model, device_ids=config.gpu)
    optimizer = torch.nn.DataParallel(optimizer, device_ids=config.gpu)
  model.to(device)
  model.train()
  global_step = 0
  f1 = 0
  logger.debug("Train starting ........")
  mems = None
  for epoch in range(config.base_epoch):
    step = 0
    logger.debug(f"Epoch: {epoch} start.")
    for i, batch in enumerate(train_loader):
      step += 1
      model.zero_grad()
      inputs, masks, tags = batch
      inputs, masks, tags = inputs.long(), masks.long(), tags.long()
      if config.use_cuda:
        inputs, masks, tags = inputs.to(device), masks.to(device), tags.to(device)
      if inputs.shape[0] != config.batch_size:
        continue
      loss, mems = model(inputs, tags, mems)["loss"]
      loss= loss.mean()
      if isinstance(model, torch.nn.DataParallel):
        loss.backward()
        optimizer.module.step()
      else:
        loss.backward()
        optimizer.step()
      plotter.plot('loss', 'train loss', 'Loss_' +suffix_name, global_step, loss.item())
      global_step += 1
      if global_step % 10 == 0:
        logger.debug(f"Epoch :{epoch}, Step: {step}, loss: {loss.item()}")
      if global_step % 300 == 0:
        f1_dev, val_metrics = test(model, dev_loader, epoch, config, label_dic)
        if f1_dev > f1:
          f1 = f1_dev
          pa = {"loss": loss.item(), "f1": f1}
          logger.debug(f"Saved model, Epoch: {epoch}, F1: {f1}.")
          save_train_model(model, epoch, path=config.checkpoint, **pa)
        logger.debug(f"Epoch :{epoch}, Dev data f1: {f1_dev},"
                     f"Precision: {val_metrics['p']}, Recall: {val_metrics['r']}")
        plotter.plot('val', 'F1', 'val status_'+suffix_name, global_step, f1_dev)
        plotter.plot('val', 'Recall', 'val status_'+suffix_name, global_step, val_metrics['r'])
        plotter.plot('val', 'Precision', 'val status_'+suffix_name, global_step, val_metrics['p'])
    f1_cur, val_metrics = test(model, test_loader, epoch, config, label_dic)
    logger.debug(f"Test data f1 result: {f1_cur}, Precision: {val_metrics['p']},"
                 f"Recall: {val_metrics['r']} Epoch: {epoch}.")
    if f1 < f1_cur:
      f1 = f1_cur
      pa = {"loss": loss.item(), "f1": f1}
      logger.debug(f"Saved model, Epoch: {epoch}, F1: {f1}.")
      save_train_model(model, epoch, path=config.checkpoint, **pa)
    plotter.plot('test', 'F1', 'test status_'+suffix_name, epoch, val_metrics['f1'], "Epoch")
    plotter.plot('test', 'Recall', 'test status_'+suffix_name, epoch, val_metrics['r'], "Epoch")
    plotter.plot('test', 'Precision', 'test status_'+suffix_name, epoch, val_metrics['p'], "Epoch")


def test(model, test_loader, epoch, config, label_dic):
  """
  Test method.

  Args:
    model(BERT_LSTM_CRF): Object of BERT_LSTM_CRF model.
    test_loader(DataLoader): Object of DataLoader for test.
    epoch(int): Epoch num.
    config(Config): Config object.
    label_dic(dict): Label and index map.

  Returns:
    val_f1(float): F1 score of test data.
  """
  model.eval()
  true = []
  pred = []
  length = 0
  mems = None
  for i, batch in enumerate(test_loader):
    inputs, masks, tags = batch
    length += inputs.size(0)
    inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
    if config.use_cuda:
      inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
    ids, mems = model.predict(inputs, mems)["pred"]
    if config.use_cuda:
      ids = ids.cpu()
      tags = tags.cpu()
    pred.extend(ids.numpy().reshape(ids.shape[0]*ids.shape[1]).tolist())
    true.extend(tags.numpy().reshape(tags.shape[0]*tags.shape[1]).tolist())
  # compute accuracy
  pre_tags = id2tag(label_dic, pred)
  true_tags = id2tag(label_dic, true)
  true_t = np.array(true_tags).reshape(len(true_tags), 1).tolist()
  pre_t = np.array(pre_tags).reshape(len(pre_tags), 1).tolist()
  val_metrics = evaluate(true_t, pre_t, True)
  val_f1 = val_metrics['f1']
  model.train()
  return val_f1, val_metrics


if __name__ == '__main__':
  train()
