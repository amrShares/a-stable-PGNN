from math import sqrt
import torch

from .buried_channel_waveguide import BuriedChannelWaveguide

class BuriedChannelWaveguideWithPINNs(BuriedChannelWaveguide):
  def __init__(self, RIs, lambd, lengths_x, lengths_y, study, field_type, underlying_model, num_modes, augmented_input=True):
    BuriedChannelWaveguide.__init__(self, RIs, lambd, lengths_x, lengths_y, study, field_type)
    
    self.underlying_model = underlying_model
    self.augmented_input = augmented_input
    self.num_modes = num_modes

    if study not in ['TE', 'TM', 'scalar']:
      raise Exception('this type of study and field are not implemented')

  @staticmethod
  def get_rayleigh_quotient(d2u_dx2, d2u_dy2, n_squared, u, mode_num):
    lu_i = d2u_dx2 + d2u_dy2 + n_squared * u[:, mode_num]
    rayleigh_quotient = (lu_i @ u[:, mode_num]) / (torch.sum(u[:, mode_num] ** 2))
    return rayleigh_quotient

  def calculate_eigen_value(self, xy, mode_num):
    u = self.underlying_model(xy)
    n_squared = self.refractive_index(xy, format='torch').detach() ** 2
    du = torch.autograd.grad(u[:, mode_num].sum(), xy, create_graph=True, retain_graph=True)[0]
    du_dx = du[:, 0]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), xy, create_graph=True)[0][:, 0]
    du_dy = du[:, 1]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), xy, create_graph=True)[0][:, 1]
    return BuriedChannelWaveguideWithPINNs.get_rayleigh_quotient(d2u_dx2, d2u_dy2, n_squared[:, 0], u, mode_num)

  def evaluate(self, xy):
    return self.underlying_model(xy)
