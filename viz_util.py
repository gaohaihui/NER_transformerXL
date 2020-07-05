# -*- coding: utf-8 -*-


import numpy as np

from visdom import Visdom


class VisdomLinePlotter(object):
  """
  Plotter object for draw line.
  """
  def __init__(self, env_name='main'):
    """
    Plots to Visdom.

    Args:
      env_name(str): Environment name.
    """
    self.viz = Visdom()
    self.env = env_name
    self.plots = {}

  def plot(self, win_name, split_name, title_name, x, y, x_label= "Steps"):
    """
    Plots line to Visdom.

    Args:
      win_name(str): Window name.
      split_name(str): The diffrent name in same window.
      x(list): Abscissa in table.
      y(list): Y-axis.
      x_label(str): X-axis name.
    """
    if win_name not in self.plots:
      self.plots[win_name] = self.viz.line(
        X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
        opts=dict(legend=[split_name],
                  title=title_name,
                  xlabel=x_label,
                  ylabel=win_name))
    else:
      self.viz.line(X=np.array([x]),
                    Y=np.array([y]),
                    env=self.env,
                    win=self.plots[win_name],
                    name=split_name,
                    update='append')