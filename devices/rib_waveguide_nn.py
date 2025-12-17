from math import sqrt
import torch

from .rib_waveguide import RibWaveguide

class RibWaveguideWithPINNs(RibWaveguide):
  def __init__(self, RIs, lambd, lengths_x, lengths_y, study, underlying_model, num_modes=1, augmented_input=True):
    RibWaveguide.__init__(self, RIs, lambd, lengths_x, lengths_y, study)

    self.underlying_model = underlying_model
    self.n_modes = num_modes
    self.augmented_input = augmented_input
    
  @staticmethod
  def get_rayleigh_quotient(d2u_dx2, d2u_dy2, n_squared, u):
      lu_i = d2u_dx2 + d2u_dy2 + n_squared * u
      rayleigh_quotient = (lu_i @ u) / torch.sum(torch.square(u))
      return rayleigh_quotient

  def calculate_eigen_value(self, xy, mode_num):
    u = self.underlying_model(xy)[:, mode_num]

    n_squared = torch.square(self.refractive_index(xy, format='torch').detach())[:, 0]

    du = torch.autograd.grad(u.sum(), xy, create_graph=True, retain_graph=True)[0]
    du_dx = du[:, 0]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), xy, create_graph=True)[0][:, 0]
    du_dy = du[:, 1]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), xy, create_graph=True)[0][:, 1]

    return RibWaveguideWithPINNs.get_rayleigh_quotient(d2u_dx2, d2u_dy2, n_squared, u)

  def __cal__(self, xy):
    return self.underlying_model(xy)


  