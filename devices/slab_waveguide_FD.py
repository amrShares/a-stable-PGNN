from math import sqrt

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from .slab_waveguide import SlabWaveguide

class SlabWaveguideWithFD(SlabWaveguide):
  def __init__(self, RIs, lambd, lengths, study, field_type, dx):
    SlabWaveguide.__init__(self, RIs, lambd, lengths, study, field_type)
    self.dx = dx

  def assemble_derivative_matrix(self, x, num_nodes):
    if self.study == 'TE':
      diagonal_elements = np.ones((num_nodes, )) * -2
      off_diagonal_elements = np.ones((num_nodes - 1, ))
      d2u_dx2 = diags([diagonal_elements, off_diagonal_elements, off_diagonal_elements], [0, -1, 1], format='csc')

    elif self.study=='TM' and self.field_type=='H':
      n_squared = self.refractive_index(x, format='numpy') ** 2
      
      a = np.copy(n_squared)
      a[:-1] += n_squared[1:]
      a = 1/a

      b = np.copy(n_squared)
      b[1:] += n_squared[:-1]
      b = 1/b

      diagonal_elements = -2 * (a + b) * n_squared
      upper_diagonal_elements = a[:-1] * 2 * n_squared[:-1]
      lower_diagonal_elemnts = b[1:] * 2 * n_squared[1:]
      d2u_dx2 = diags([diagonal_elements, lower_diagonal_elemnts, upper_diagonal_elements], [0, -1, 1], format='csc')

    # elif self.study == 'TM' and self.field_type == 'E':
    #   n_squared = self.refractive_index(x, format='numpy') ** 2
    #
    #   a = np.copy(n_squared)
    #   a[:-1] += n_squared[1:]
    #   a = 1 / a
    #
    #   b = np.copy(n_squared)
    #   b[1:] += n_squared[:-1]
    #   b = 1 / b
    #
    #   b1 = np.copy(n_squared)
    #   b1[1:] += n_squared[:-1] # P_n + P_{n-1}
    #
    #   b2 = np.copy(n_squared)
    #   b2[1:] -= n_squared[:-1] # P_n - P_{n-1}
    #
    #   a1 = np.copy(n_squared)
    #   a1[:-1] += n_squared[1:] # P_n + P_{n+1}
    #
    #   a2 = -np.copy(n_squared)
    #   a2[:-1] += n_squared[1:] # -P_n + P_{n+1}
    #
    #   upper_diagonal_elements = a[:-1] * 2 * n_squared[:-1]
    #   lower_diagonal_elemnts = b[1:] * 2 * n_squared[1:]
    #   diagonal_elements = -2 - b2/b1 + a2/a1
    #   d2u_dx2 = diags([diagonal_elements, lower_diagonal_elemnts, upper_diagonal_elements], [0, -1, 1], format='csc')
    # else:
    #   raise NotImplementedError("Unrecognized study")
    elif self.study=='TM' and self.field_type=='E':
      n_squared = self.refractive_index(x, format='numpy') ** 2

      a1 = np.copy(n_squared)
      a1[:-1] += n_squared[1:] # P_n + P_{n+1}

      a2 = -np.copy(n_squared)
      a2[:-1] += n_squared[1:] # -P_n + P_{n+1}

      b1 = np.copy(n_squared)
      b1[1:] += n_squared[:-1] # P_n + P_{n-1}

      b2 = np.copy(n_squared)
      b2[1:] -= n_squared[:-1] # P_n - P_{n-1}

      diagonal_elements = (a2/a1 - b2/b1) - 2
      upper_diagonal_elements = (a2/a1)[:-1] + 1
      lower_diagonal_elements = -(b2/b1)[1:] + 1

      d2u_dx2 = diags([diagonal_elements, lower_diagonal_elements, upper_diagonal_elements], [0, -1, 1], format='csc')

    else:
      raise NotImplementedError("Unrecognized study")

    d2u_dx2 /= self.dx ** 2
    d2u_dx2[0, 0] = 1
    d2u_dx2[0, 1] = 0
    d2u_dx2[num_nodes - 1, num_nodes - 1] = 1
    d2u_dx2[num_nodes - 1, num_nodes - 2] = 0
    return d2u_dx2


  def assemble_scale_matrix(self, x):
    n_x = diags([self.refractive_index(x, 'numpy')], [0], format='csc')
    n_x[0, 0]=0
    n_x[len(x)-1, len(x)-1]=0
    return n_x

  def evaluate(self, num_modes):
    x = np.arange(0, self.total_length/2, self.dx)
    x = np.concatenate((-x[::-1][:-1], x))
    num_nodes = len(x)
    d2u_dx2 = self.assemble_derivative_matrix(x, num_nodes)
    n_u = self.assemble_scale_matrix(x)
    A = d2u_dx2 + n_u ** 2
    eig_vals, eig_vecs = eigs(A, sigma=self.n_core**2)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    sorting_indices = np.argsort(-eig_vals)
    # if self.study == 'TM' and self.field_type=='E':
    #   assert len(eig_vecs.shape) == 2
    #   eig_vecs /= self.refractive_index(x, format='numpy').reshape(-1, 1)**2
    return eig_vals[sorting_indices[:num_modes]], x, eig_vecs[:, sorting_indices[:num_modes]]