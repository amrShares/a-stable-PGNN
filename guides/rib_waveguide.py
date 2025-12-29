from math import sqrt
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class RibWaveguide:

  def __init__(self, RIs, lambd, lengths_x, lengths_y, study):
    self.n_substrate = RIs[0]
    self.n_core = RIs[1]
    self.n_cladding = RIs[2]

    self.lambd = lambd
    self.lengths_x = lengths_x # substrate core substrate
    self.lengths_y = lengths_y # substrate rib core cladding
    self.study = study

    self.total_length_x = sum(lengths_x)
    self.total_length_y = sum(lengths_y)
    
    self.upward_length = lengths_y[2] + lengths_y[3]
    self.downward_length = lengths_y[0] + lengths_y[1]
    
    self.k_0 = 2 * np.pi / lambd
    
    self.total_area = self.total_length_x * self.total_length_y
    self.effective_area = self.lengths_x[1] * self.lengths_y[1]

  def refractive_index(self, xy: torch.Tensor, format='numpy'):
    if format == 'numpy':
      out = np.ones((len(xy), 1))
    else:
      out = torch.ones(len(xy), 1, dtype=xy.dtype, device=xy.device)
    
    lower_sub_mask = xy[:, 1]<0
    upper_clad_mask = xy[:, 1]>0
    core_mask = (abs(xy[:, 0]) <= self.lengths_x[1]/2) & (xy[:, 1]>=0) & (xy[:, 1]<=self.lengths_y[2])
    rib_mask = (xy[:, 1]<=0) & (xy[:, 1]>=-self.lengths_y[1])

    out[lower_sub_mask] = self.n_substrate
    out[upper_clad_mask] = self.n_cladding
    out[core_mask] = self.n_core
    out[rib_mask] = self.n_core

    return out

  def regions(self, xy: torch.Tensor, format='numpy'):
    if format == 'numpy':
      out = np.ones((len(xy), 1))
    else:
      out = torch.ones(len(xy), 1, dtype=xy.dtype, device=xy.device)
    
    lower_sub_mask = xy[:, 1]<0
    upper_clad_mask = xy[:, 1]>=0
    core_mask = (abs(xy[:, 0]) <= self.lengths_x[1]/2) & (xy[:, 1]>0) & (xy[:, 1]<=self.lengths_y[2])
    rib_mask = (xy[:, 1]<=0) & (xy[:, 1]>=-self.lengths_y[1])
    
    out[lower_sub_mask] = self.n_substrate**2
    out[upper_clad_mask] = self.n_cladding**2
    out[core_mask] = self.n_core**2
    out[rib_mask] = self.n_core**2

    return out

  def plot(self, dx, dy):
    dtype = torch.float32
    device = torch.device('cpu')

    xu = torch.arange(0, self.simulation_lengths_x[0], dx, dtype=dtype, device=device)
    xl = torch.arange(0, self.simulation_lengths_x[1], dx, dtype=dtype, device=device)
    x = torch.cat((torch.flip(-xl, dims=(0,))[:-1], xu))

    yu = torch.arange(0, self.simulation_lengths_y[0], dy, dtype=dtype, device=device)
    yl = torch.arange(0, self.simulation_lengths_y[1], dy, dtype=dtype, device=device)
    y = torch.cat((torch.flip(-yl, dims=(0,))[:-1], yu))

    X, Y = torch.meshgrid(x, y)
    
    xy = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)
    xy = xy.reshape(-1, 2)
    
    out = torch.ones(len(xy), 1, dtype=xy.dtype, device=xy.device)
    
    lower_sub_mask = xy[:, 1]<0
    upper_clad_mask = xy[:, 1]>=0
    core_mask = (abs(xy[:, 0]) <= self.lengths_x[1]/2) & (xy[:, 1]>0) & (xy[:, 1]<=self.lengths_y[2])
    rib_mask = (xy[:, 1]<=0) & (xy[:, 1]>=-self.lengths_y[1])
    
    out[lower_sub_mask] = -1
    out[upper_clad_mask] = 1
    out[core_mask] = 0
    out[rib_mask] = 0
    
    out = out.reshape(X.shape)

    plt.figure(figsize=(10, 8))
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], 3)
    plt.contourf(X, Y, out, cmap=cm);
    plt.clim(-1.5, 1.5)
    cb = plt.colorbar(ticks=[-1, 0, 1])
    cb.ax.tick_params(length=0)

  def get_discontinuities(self, dx, dy, dtype, device='cpu'):
    xu = torch.arange(0, self.simulation_lengths_x[0], dx/2, dtype=dtype, device=device).reshape(-1, 1)
    xl = torch.arange(0, self.simulation_lengths_x[1], dx/2, dtype=dtype, device=device).reshape(-1, 1)
    x = torch.cat((torch.flip(-xl, dims=(0,))[:-1], xu))

    yu = torch.arange(dy/2, self.simulation_lengths_y[0], dy/2, dtype=dtype, device=device).reshape(-1, 1)
    yl = torch.arange(dy/2, self.simulation_lengths_y[1], dy/2, dtype=dtype, device=device).reshape(-1, 1)
    y = torch.cat((torch.flip(-yl, dims=(0,)), yu))

    left_points = yu[(yu<self.lengths_y[2])&(yu>0)].reshape(-1, 1)
    left_interface = torch.cat((torch.full_like(left_points, -self.lengths_x[1]/2), left_points), dim=-1)
    
    right_points = yu[(yu<self.lengths_y[2])&(yu>0)].reshape(-1, 1)
    right_interface = torch.cat((torch.full_like(right_points, self.lengths_x[1]/2), right_points), dim=-1)
    
    bottom_points = x[1:-1].reshape(-1, 1)
    bottom_interface = torch.cat((bottom_points, torch.full_like(bottom_points, -self.lengths_y[1])), dim=-1)
    
    center_points = x[(x>self.lengths_x[1]/2)|(x<-self.lengths_x[1]/2)].reshape(-1, 1)
    center_interface = torch.cat((center_points, torch.full_like(center_points, 0)), dim=-1)

    top_points = x[(x<self.lengths_x[1]/2)&(x>-self.lengths_x[1]/2)].reshape(-1, 1)
    top_interface = torch.cat((top_points, torch.full_like(top_points, self.lengths_y[2])), dim=-1)
    
    return torch.cat((bottom_interface, center_interface, top_interface)), torch.cat((left_interface, right_interface)), [len(bottom_interface), len(center_interface), len(top_interface)], [len(left_interface), len(right_interface)]

  def get_boundaries(self, dx, dy, dtype,  device):
    xu = torch.arange(0, self.simulation_lengths_x[0], dx, dtype=dtype, device=device)
    xl = torch.arange(0, self.simulation_lengths_x[1], dx, dtype=dtype, device=device)
    X = torch.cat((torch.flip(-xl, dims=(0,))[:-1], xu)).reshape(-1, 1)

    yu = torch.arange(0, self.simulation_lengths_y[0], dy, dtype=dtype, device=device)
    yl = torch.arange(0, self.simulation_lengths_y[1], dy, dtype=dtype, device=device)
    Y = torch.cat((torch.flip(-yl, dims=(0,))[:-1], yu)).reshape(-1, 1)

    bd_1 = torch.cat((torch.empty_like(Y).fill_(X[0, 0]), Y), dim=1)
    y_bd1 = torch.zeros_like(Y)

    bd_2 = torch.cat((torch.empty_like(Y).fill_(X[-1, 0]), Y), dim=1)
    y_bd2 = torch.zeros_like(Y)

    bd_3 = torch.cat((X, torch.empty_like(X).fill_(Y[0, 0])), dim=1)
    y_bd3 = torch.zeros_like(X)

    bd_4 = torch.cat((X, torch.empty_like(X).fill_(Y[-1, 0])), dim=1)
    y_bd4 = torch.zeros_like(X)

    bd = torch.cat((bd_1, bd_2, bd_3, bd_4), dim=0)
    u_bd = torch.cat((y_bd1, y_bd2, y_bd3, y_bd4), dim=0)
    return bd, u_bd

  @property
  def central_region(self):
    return self.lengths_x[1]/2, self.lengths_y[1]/2

  @property
  def simulation_lengths_x(self):
    return self.total_length_x/2, self.total_length_x/2

  @property
  def simulation_lengths_y(self):
    return self.upward_length, self.downward_length

