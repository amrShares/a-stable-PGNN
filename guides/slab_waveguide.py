from math import sqrt
import numpy as np
import torch
from scipy.optimize import fsolve
from scipy.io import loadmat
from .waveguide import WaveGuide

class SlabWaveguide(WaveGuide):

  def __init__(self, RIs, lambd, lengths, study, field_type):
    WaveGuide.__init__(self, RIs, lambd, lengths, study, field_type)
    self.n_substrate = RIs[0]
    self.n_core = RIs[1]
    self.n_cladding = RIs[2]
    self.substrate_length = lengths[0]
    self.core_length = lengths[1]
    self.cladding_length = lengths[2]

  def refractive_index(self, x, format='numpy'):
    if format == 'numpy':
      out = np.ones_like(x)
    else:
      out = torch.ones_like(x)

    out[x <= (-self.core_length/2)] = self.n_substrate
    out[((-self.core_length/2) < x) & (x < (self.core_length/2))] = self.n_core
    out[((self.core_length/2) <= x)] = self.n_cladding
    return out

  def get_discontinuities(self, format='numpy', dtype='float64'):
    if format == 'numpy':
      return np.array([[-self.core_length/2], [self.core_length/2]])
    else:
      return torch.tensor([[-self.core_length/2], [self.core_length/2]], dtype=torch.float64) if dtype=='float64' else torch.tensor([[-self.core_length/2], [self.core_length/2]], dtype=torch.float32)


  def get_discontinuities_index(self, x):
    return [
      torch.argmin(torch.abs(x+self.core_length/2)),
      torch.argmin(torch.abs(x-self.core_length/2))
      ]


  @property
  def central_region(self):
    return self.core_length/2

  def evaluate_analytical(self, evaluation_points, num_modes):
    w_core = self.core_length / self.k_0

    def func(n_eff):
      gamma_1 = np.sqrt(self.n_core**2 - n_eff**2)
      gamma_2 = np.sqrt(n_eff**2 - self.n_substrate**2)
      gamma_3 = np.sqrt(n_eff**2 - self.n_cladding**2)
      a = self.k_0 * gamma_1 * w_core
      b = mode * np.pi
      if self.study=='TM':
        c = np.arctan(((self.n_core**2)*gamma_3)/((self.n_cladding**2)*gamma_1))
        d = np.arctan(((self.n_core**2)*gamma_2)/((self.n_substrate**2)*gamma_1))
      else:
        c = np.arctan((gamma_3)/(gamma_1))
        d = np.arctan((gamma_2)/(gamma_1))
      return a - b - c - d

    analytical_modes = np.zeros((num_modes, ))
    analytical_fields = np.zeros((num_modes, len(evaluation_points)))
    analytical_other1 = np.zeros((num_modes, len(evaluation_points)))
    analytical_other2 = np.zeros((num_modes, len(evaluation_points)))
    power_flux = np.zeros((num_modes, len(evaluation_points)))

    if self.study == 'TM':
        pdFL = self.n_core**2 / self.n_substrate**2
    else:
        pdFL = 1
    pdFn = 1
        
    thicknesses = np.array([self.core_length])/self.k_0
    hn = np.cumsum(thicknesses)
    er = self.refractive_index(evaluation_points.reshape(-1), format='numpy')**2

    for mode in range(num_modes):
      mode_initial_guess = self.n_core*0.99
      mode_solution = fsolve(func, mode_initial_guess)[0]
      analytical_modes[mode] = mode_solution

      # getting the eigenfunction
      gL = 2 * np.pi * np.sqrt(mode_solution**2 - self.n_substrate**2)
      gR = 2 * np.pi * np.sqrt(mode_solution**2 - self.n_cladding**2)

      kn = 2 * np.pi * np.sqrt(self.n_core**2 - mode_solution**2)

      A = 1
      the = np.arctan(gL / kn * pdFL)
      AL = A * np.cos(-the)
      AR = A * np.cos(kn * hn - the)

      x = evaluation_points.copy().reshape(-1)
      x = x / self.k_0
      x += w_core / 2

      hs = np.concatenate(([0], hn))
      iL = x < 0
      analytical_fields[mode, iL] = AL * np.exp(gL * (x[iL] - hs[0]))
      iR = x >= hn[-1]
      analytical_fields[mode, iR] = AR * np.exp(-gR * (x[iR] - hs[-1]))

      ix = (x >= hs[0]) & (x < hs[1])
      analytical_fields[mode, ix] = A * np.cos(kn * x[ix] - the)

    return evaluation_points, analytical_modes,  analytical_fields