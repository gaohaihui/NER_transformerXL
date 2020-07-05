# -*- coding: utf-8 -*-



class Config(object):
  """
  Config for training.
  """
  def __init__(self):
    self.label_file = r'../data/train_data/msra_transfer/tag.txt'
    self.vocab = r'../data/bert/vocab.txt'
    self.max_length = 512
    self.use_cuda = True
    self.gpu = [7]
    self.batch_size = 12
    self.bert_path = '../data/bert'
    self.rnn_hidden = 512
    self.bert_embedding = 768
    self.dropout1 = 0.45
    self.rnn_layer = 1
    self.lr = 0.000001
    self.lr_decay = 0.000001
    self.weight_decay = 0.0003
    self.checkpoint = 'result/'
    self.optim = 'Adam'
    self.load_model = True
    self.load_path = r"./result"
    self.base_epoch = 200
    self.root = r"../data/train_data/msra_transfer"
    self.label_mode = "BIOSE"



  def update(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __str__(self):
    return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
  con = Config()
  con.update(gpu=8)
  print(con.gpu)
  print(con)
