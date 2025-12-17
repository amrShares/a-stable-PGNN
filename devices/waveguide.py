from math import sqrt
import numpy as np
import torch

class WaveGuide:

  def __init__(self, RIs, lambd, lengths, study, field_type):
    self.RIs = RIs
    self.lengths = lengths
    self.total_length = sum(lengths)
    self.study = study
    self.field_type = field_type
    self.lambd = lambd
    self.k_0 = 2 * np.pi / lambd

  def refractive_index(self, x, format='numpy'):
    if format == 'numpy':
      out = np.ones_like(x)
    else:
      out = torch.ones_like(x)
      
    augmented_lengths = np.concatenate(([np.finfo(np.float64).min], np.cumsum(self.lengths)[:-1], [np.finfo(np.float64).max]))
    for i in range(len(self.RIs)):
      out[(augmented_lengths[i]-self.total_length/2<x)&(x<=augmented_lengths[i+1]-self.total_length/2)] = self.RIs[i]
    return out

  def get_discontinuities(self, format='numpy'):
    return np.cumsum(self.lengths)[:-1] - self.total_length/2 if format=='numpy' else torch.from_numpy(np.cumsum(self.lengths)[:-1] - self.total_length/2)

  def get_discontinuities_index(self, x):
    cumulative_lengths = np.cumsum(self.lengths)[:-1]
    return [
      torch.argmin(torch.abs(x+self.total_length/2-cumulative_lengths[i]))
      for i in range(len(self.RIs)-1)
      ]

  def get_boundaries_index(self, x):
    return [torch.argmin(x), torch.argmax(x)]

  @property
  def central_region(self):
    return self.lengths[0]
