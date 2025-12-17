from math import sqrt
import numpy as np
import torch
from scipy.optimize import fsolve
from scipy.io import loadmat
from .w_guide import SymmetricWGuide

class SymmetricWGuideWithPINNs(SymmetricWGuide):

  def __init__(self, n_base, n_center, n_head, lambd, base_length, center_length, head_length, study, field_type, underlying_model, augmented_input=True):
    SymmetricWGuide.__init__(self, n_base, n_center, n_head, lambd, base_length, center_length, head_length, study, field_type)
    self.underlying_model = underlying_model
    self.augmented_input = augmented_input
    if study=='TM' and field_type=='H':
      self.continuous_field = True
      self.continuous_derivative = False
    elif study=='TM' and field_type=='E':
      self.continuous_field=False
      self.continuous_derivative=True
    elif study=='TE' and field_type=='E':
      self.continuous_field=True
      self.continuous_derivative=True    

  @staticmethod
  def get_rayleigh_quotient(d2u_dx2, n_squared, u, mode_num):
    lu_i = d2u_dx2 + n_squared * u[:, mode_num]
    rayleigh_quotient = (lu_i @ u[:, mode_num]) / torch.sum(u[:, mode_num] ** 2)
    return rayleigh_quotient

  def calculate_eigen_value(self, x, mode_num):
    u = self.underlying_model(x)
    n_squared = self.refractive_index(x[:, 0], format='torch').detach() ** 2
    du_dx = torch.autograd.grad(u[:, mode_num].sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0][:, 0]

    return SymmetricWGuideWithPINNs.get_rayleigh_quotient(d2u_dx2, n_squared, u, mode_num)

  def evaluate(self, x):
    return self.underlying_model(x)