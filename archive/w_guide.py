from math import sqrt
import numpy as np
import torch
from scipy.optimize import fsolve
from scipy.io import loadmat

class SymmetricWGuide:

  def __init__(self, n_base, n_center, n_head, lambd, base_length, center_length, head_length, study, field_type):
    self.n_base = n_base
    self.n_center = n_center
    self.n_head = n_head
    self.base_length = base_length
    self.center_length = center_length 
    self.head_length = head_length
    self.total_length = (base_length + center_length + head_length)*2
    self.study = study
    self.field_type = field_type
    self.lambd = lambd
    self.k_0 = 2 * np.pi / lambd
    self.w_core = base_length / self.k_0

  def refractive_index(self, x, format='numpy'):
    if format == 'numpy':
      out = np.ones_like(x)
    else:
      out = torch.ones_like(x)
    out[abs(x) <= self.base_length] = self.n_base
    out[(self.base_length < abs(x)) & (abs(x) < self.base_length + self.center_length)] = self.n_center
    out[self.base_length + self.center_length <= abs(x)] = self.n_head
    return out

  def get_discontinuities_index(self, x):
    return [
      torch.argmin(torch.abs(x+self.base_length+self.center_length)),
      torch.argmin(torch.abs(x+self.base_length)),
      torch.argmin(torch.abs(x-self.base_length)),
      torch.argmin(torch.abs(x-self.base_length-self.center_length)),    
      ]

  def get_boundaries_index(self, x):
    return [torch.argmin(x), torch.argmax(x)]

  @property
  def central_region(self):
    return self.base_length


