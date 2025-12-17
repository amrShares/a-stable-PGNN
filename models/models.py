from functools import partial
import torch
import torch.nn as nn
import numpy as np

class RBF(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)

class DRBF(nn.Module):
    def forward(self, x):
        s = torch.square(x)
        return s*torch.exp(-s)

class Siren(nn.Module):
  def forward(self, x):
    return torch.cos(x)
  
def vanilla_network(in_features, hidden_features, out_features, num_hidden_layers, nonlinearity='RBF'):
  if nonlinearity=='RBF':
    f = RBF
  elif nonlinearity=='DRBF':
    f = DRBF
  elif nonlinearity=='tanh':
    f = nn.Tanh
  elif nonlinearity=='Swish':
    f = nn.SiLU
  elif nonlinearity=='Siren':
    f = Siren
  layers = [nn.Linear(in_features, hidden_features), f()]
  for i in range(num_hidden_layers - 1):
    layers.extend([nn.Linear(hidden_features, hidden_features), f()])
  layers.append(nn.Linear(hidden_features, out_features))
  return nn.Sequential(*layers)

def discontinuity_capturing_network(in_features, hidden_features, out_features, num_hidden_layers, nonlinearities=['RBF', 'RBF']):
  f = []
  for i in range(len(nonlinearities)):
    nonlinearity = nonlinearities[i]
    if nonlinearity=='RBF':
      f.append(RBF)
    elif nonlinearity=='DRBF':
      f.append(DRBF)
    elif nonlinearity=='tanh':
      f.append(nn.Tanh)
    elif nonlinearity=='Swish':
      f.append(nn.SiLU)
    elif nonlinearity=='Siren':
      f.append(Siren)
    elif nonlinearity=='GeLU':
      f.append(nn.GELU)
  layers = [nn.Linear(in_features + 1, hidden_features), f[0]()]
  for i in range(num_hidden_layers - 1):
    layers.extend([nn.Linear(hidden_features, hidden_features), f[1]()])
  layers.append(nn.Linear(hidden_features, out_features))
  return nn.Sequential(*layers)

def default_init(m):
    """
    Applies Xavier initialization to weights and zeros biases.
    """
    if isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

def get_nonlinearity(nonlinearity):
  if nonlinearity == 'RBF':
    return RBF()
  elif nonlinearity == 'DRBF':
    return DRBF()
  elif nonlinearity == 'tanh':
    return nn.Tanh()
  elif nonlinearity == 'Swish':
    return nn.SiLU()
  elif nonlinearity == 'Siren':
    return Siren()
  else:
    raise Exception('Unknown nonlinearity')

class ResidualBlock(nn.Module):
  """
  A residual fully-connected block: FC -> Activation -> FC + shortcut.
  """

  def __init__(self, features: int, hidden: int = None, nonlinearity='Tanh'):
    super().__init__()
    hidden = hidden or features
    self.fc1 = nn.Linear(features, hidden)
    self.fc2 = nn.Linear(hidden, features)
    self.nonlinearity = get_nonlinearity(nonlinearity)

    # Apply initialization
    self.apply(default_init)

  def forward(self, x):
    identity = x
    out = self.nonlinearity(self.fc1(x))
    out = self.fc2(out)
    return self.nonlinearity(out + identity)


class DCNN(nn.Module):
  """
  A simple deep residual fully-connected network.

  Args:
      in_dim: Input dimension.
      hidden_dim: Hidden layer size.
      num_blocks: Number of residual blocks.
      out_dim: Output dimension.
      activation: Activation function (callable).
  """

  def __init__(self,
               in_features: int,
               hidden_features: int,
               out_features: int,
               num_hidden_layers: int,
               n_core,
               nonlinearities=['tanh', 'tanh']):
    super().__init__()
    self.model = discontinuity_capturing_network(in_features, hidden_features, out_features, num_hidden_layers, nonlinearities)
    self.eigenvalue = nn.Parameter(torch.tensor([n_core]))

    # Initialize input and output layers
    self.apply(default_init)

  def forward(self, x):
    return self.model(x)

class ResidualFCNN(nn.Module):
  """
  A simple deep residual fully-connected network.

  Args:
      in_dim: Input dimension.
      hidden_dim: Hidden layer size.
      num_blocks: Number of residual blocks.
      out_dim: Output dimension.
      activation: Activation function (callable).
  """

  def __init__(self,
               in_dim: int,
               hidden_dim: int,
               num_blocks: int,
               out_dim: int,
               n_core,
               nonlinearity='tanh'):
    super().__init__()
    self.input_layer = nn.Linear(in_dim, hidden_dim)
    self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, nonlinearity=nonlinearity)
                                  for _ in range(num_blocks)])
    self.output_layer = nn.Linear(hidden_dim, out_dim)
    self.nonlinearity = get_nonlinearity(nonlinearity)
    self.eigenvalue = nn.Parameter(torch.tensor([n_core ** 2]))

    # Initialize input and output layers
    self.apply(default_init)

  def forward(self, x):
    x = self.nonlinearity(self.input_layer(x))
    x = self.blocks(x)
    return self.output_layer(x)