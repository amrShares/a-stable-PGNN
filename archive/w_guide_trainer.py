import torch
import numpy as np
import torch.nn as nn
import tqdm
import utils
import matplotlib.pyplot as plt

class WGuideTrainer:
    def __init__(self, dtype, device, memory_type,samplers, optimzer, scheduler, n_upper=None, n_lower=None, dx=None, num_samples=None):
      self.samplers = samplers
      self.optimizer = optimzer
      self.scheduler = scheduler
      self.device = device
      self.dtype = dtype

      if n_lower is None:
        self.n_lower = max(self.device.n_center, self.device.n_head)
      else:
        self.n_lower = n_lower
      if n_upper is None:
        self.n_upper = self.device.n_base
      else:
        self.n_upper = n_upper  

      self.dx = dx
      self.num_samples = num_samples
      self.memory_type = memory_type

    @staticmethod
    def adaptive_sampling(sampled_values, residuals, num, device=torch.device('cpu')):
        res_as_prob = torch.exp(residuals)
        res_as_prob = res_as_prob/torch.sum(res_as_prob)
        sorted_values, sorting_indices = torch.sort(sampled_values)
        cumulative_distribution = torch.cumsum(res_as_prob[sorting_indices], dim=0)
        uniform_samples = torch.rand(num, device=device)
        a = cumulative_distribution.reshape(1, -1) - uniform_samples.reshape(-1, 1)
        mask1 = (a>0) * np.finfo(np.float32).eps
        mask2 = (a<0) * -np.finfo(np.float32).max
        a *= (mask1 + mask2)
        new_sampled_values = sorted_values[torch.topk(a, k=1, dim=-1, largest=False).indices]
        new_sampled_values += torch.randn_like(new_sampled_values, device=device) * 0.01
        return new_sampled_values

    def bd_loss(self, u_bd, u_pred_bd):
        return torch.mean(torch.abs(u_bd - u_pred_bd))

    def continuity_loss(self, x, u, rayleigh_values, num_modes):
        loss = [torch.zeros(1, 1, device=self.memory_type)]

        discontinuities_tensor = torch.tensor(
          [[-self.device.center_length-self.device.base_length], [-self.device.base_length], [self.device.base_length], [self.device.base_length+self.device.center_length]],
          dtype=self.dtype,
          device=self.memory_type,
          requires_grad=True).reshape(-1, 1)
        # discontinuities_tensor = x[[self.device.get_discontinuities_index(x[:, 0])]][:, 0].reshape(-1, 1)
        refractive_indices_squared = torch.tensor(
          [[self.device.n_head], [self.device.n_center], [self.device.n_base], [self.device.n_center], [self.device.n_head]],
          device=self.memory_type,
          dtype=self.dtype).reshape(-1, 1) ** 2

        for i in range(len(discontinuities_tensor)):
            for j in range(num_modes):
                if self.device.augmented_input:
                    interface_input_l = torch.cat((discontinuities_tensor[[i]], refractive_indices_squared[[i]]), dim=-1)
                else:
                    interface_input_l = discontinuities_tensor[[i]]-self.dx
                left_val = self.device.underlying_model(interface_input_l)[:, j]

                if self.device.augmented_input:
                    interface_input_r = torch.cat((discontinuities_tensor[[i]], refractive_indices_squared[[i+1]]), dim=-1)
                else:
                    interface_input_r = discontinuities_tensor[[i]]+self.dx
                right_val = self.device.underlying_model(interface_input_r)[:, j]

                grad_left = torch.autograd.grad(left_val, discontinuities_tensor, create_graph=True)[0]
                grad_right = torch.autograd.grad(right_val, discontinuities_tensor, create_graph=True)[0]

                # if self.device.augmented_input:
                #   loss.append(torch.abs(left_val - right_val))

                # if self.device.study == 'TE' and self.device.augmented_input:
                #     loss.append(torch.sum(torch.abs(grad_right[i] - grad_left[i])))
                # elif self.device.study == 'TM':
                #     loss.append(torch.sum(torch.abs(refractive_indices_squared[i]*grad_right[i] - refractive_indices_squared[i+1]*grad_left[i])))
                      
                      # grad_zero = torch.autograd.grad(u[0, j], x, create_graph=True)[0][0]
                      # mask = x[:, 0]<discontinuities_tensor[i].item()
                      # integrand = (1-(rayleigh_values[j]/x[mask][:, 1])) * u[mask][:, j]
                      # I_x = torch.trapezoid(integrand, x[mask, 0], dim=0) - (1/refractive_indices_squared[0])*grad_zero
                      # gradient_difference_term = (refractive_indices_squared[i+1] - refractive_indices_squared[i]) * I_x
                      # loss.append(torch.sum(torch.abs(grad_right[i] - grad_left[i] + gradient_difference_term)))
                      
                if self.device.augmented_input:
                    if self.device.continuous_field:
                        loss.append(torch.abs(left_val - right_val))
                    else:
                        loss.append(torch.abs(refractive_indices_squared[i]*left_val - refractive_indices_squared[i+1]*right_val))
                    if self.device.continuous_derivative:
                        loss.append(torch.sum(torch.abs(grad_right - grad_left)))
                    else:
                        loss.append(torch.sum(torch.abs(refractive_indices_squared[i]*grad_right - refractive_indices_squared[i+1]*grad_left)))
                else:
                    if not self.device.continuous_field:
                        loss.append(torch.abs(refractive_indices_squared[i]*left_val - refractive_indices_squared[i+1]*right_val))
                    if not self.device.continuous_derivative:
                        loss.append(torch.sum(torch.abs(refractive_indices_squared[i]*grad_right - refractive_indices_squared[i+1]*grad_left)))

        return sum(loss)

    def interior_loss(self, x_rayleigh, x_PDE, u, num_modes, sample_new=False):
        cosine_similarity_fn = nn.CosineSimilarity(dim=0, eps=1e-6)
        losses_1 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_2 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_3 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_4 = [torch.zeros(1, 1, device=self.memory_type)]
        rayleigh_values = []
        new_samples = None
        n_squared = self.device.refractive_index(x_PDE[:, 0], format='torch').detach() ** 2
        
        for i in range(num_modes):
            du_dx = torch.autograd.grad(u[:, i].sum(), x_PDE, create_graph=True, retain_graph=True)[0][:, 0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), x_PDE, create_graph=True)[0][:, 0]

            if len(self.samplers) == 2:
              rayleigh = self.device.calculate_eigen_value(x_rayleigh, i)
            else:
              rayleigh = self.device.get_rayleigh_quotient(d2u_dx2, n_squared, u, i)
            rayleigh_values.append(rayleigh.detach().item())

            if not sample_new :
                losses_1.append(torch.mean(torch.abs(d2u_dx2 + (n_squared - rayleigh) * u[:, i])))
            else:
                residuals = torch.abs(d2u_dx2 + (n_squared - rayleigh) * u[:, i])
                new_samples = WGuideTrainer.adaptive_sampling(x_PDE, residuals, sample_new * len(x_rayleigh), self.device)
                losses_1.append(torch.mean(residuals))

            losses_2.append(torch.square(torch.sum(u[:, i] ** 2) * self.device.total_length / len(u[:]) - 1))
            losses_3.append(torch.maximum(self.n_lower ** 2 - rayleigh, torch.zeros_like(rayleigh)) + torch.maximum(rayleigh - self.n_upper ** 2, torch.zeros_like(rayleigh)))

            for j in range(i + 1, num_modes):
                losses_4.append(torch.square(cosine_similarity_fn(u[:, i], u[:, j])))

        return sum(losses_1), sum(losses_2), sum(losses_3), sum(losses_4), rayleigh_values, new_samples

    def construct_training_data(self):
        x_PDE = self.samplers[0](self.device, self.num_samples if self.num_samples else self.dx, self.dtype, self.memory_type)
        x_PDE.requires_grad = True
        if self.device.augmented_input:
            n_squared = self.device.refractive_index(x_PDE[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            x_PDE = torch.cat((x_PDE, n_squared), dim=1).to(self.memory_type)
        if len(self.samplers)==2:
          x_rayleigh = self.samplers[1](self.device, self.dx, self.dtype, self.memory_type)
          x_rayleigh.requires_grad=True
          if self.device.augmented_input:
              n_squared_rayleigh = self.device.refractive_index(x_rayleigh[:, 0], format='torch').detach().reshape(-1, 1) ** 2
              x_rayleigh = torch.cat((x_rayleigh, n_squared_rayleigh), dim=1).to(self.memory_type)
        else:
          x_rayleigh = x_PDE

        print('num PDE samples', len(x_PDE))
        print('num rayleigh samples', len(x_rayleigh))
        return x_PDE, x_rayleigh


    def train(self, epochs, num_modes, weights=[1, 1, 1, 1, 1, 1], verbose=0, calc_error=False, plot_solution=False, sample_new=False):
        if verbose ==10:
            loop = tqdm.tqdm(range(epochs))
        else:
            loop = range(epochs)

        loss_history = []
        rayleigh_history = [[] for _ in range(num_modes)]

        self.device.underlying_model.to(self.memory_type)
        self.device.underlying_model.to(self.dtype)

        x_PDE, x_rayleigh = self.construct_training_data()

        bd = x_PDE[[self.device.get_boundaries_index(x_PDE[:, 0])]][:, [0]].detach()
        if self.device.augmented_input:
            n_squared_bd = self.device.refractive_index(bd[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            bd = torch.cat((bd, n_squared_bd), dim=1).to(self.memory_type)

        u_bd = torch.zeros(2, 1, device=self.memory_type, dtype=self.dtype)
        u_bd = u_bd.to(self.memory_type)

        mode_errors_list = []
        for i in range(num_modes):
            mode_errors_list.append([])

        for j in loop:
            def closure():
                self.optimizer.zero_grad()
                u_pred = self.device.underlying_model(x_PDE)
                u_pred_bd = self.device.underlying_model(bd)

                loss_1, loss_2, loss_3, loss_4, rayleigh_values, new_points = self.interior_loss(x_rayleigh, x_PDE, u_pred, num_modes, sample_new)
                loss_5 = self.bd_loss(u_bd, u_pred_bd)
                loss_6 = self.continuity_loss(x_rayleigh, u_pred, rayleigh_values, num_modes)

                (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6).backward()
                nn.utils.clip_grad_norm_(self.device.underlying_model.parameters(), 1.0)

                loss_history.append((loss_1 + loss_2 + loss_3 + loss_4 + loss_5).detach().item())

                for i in range(num_modes):
                    rayleigh_history[i].append(rayleigh_values[i])

                if verbose == 10:
                    loop.set_description(f'loss1 is {round(loss_1.detach().item(), 5)}, loss2 is {round(loss_2.detach().item(), 5)}, loss3 is {round(loss_3.detach().item(), 5)}, loss4 is {round(loss_4.detach().item(), 5)}, loss5 is {round(loss_5.detach().item(), 5)}, loss6 is {round(loss_6.detach().item(), 5)}')
                elif verbose > 0:
                    if j%(int(1000/verbose))==0:
                      print((f'iteration {j+1} loss1 is {round(loss_1.detach().item(), 5)}, loss2 is {round(loss_2.detach().item(), 5)}, loss3 is {round(loss_3.detach().item(), 5)}, loss4 is {round(loss_4.detach().item(), 5)}, loss5 is {round(loss_5.detach().item(), 5)}, loss6 is {round(loss_6.detach().item(), 5)}'))
                
                return (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6)

            self.optimizer.step(closure)
            self.scheduler.step()

            if verbose > 0:
                if j%(int(1000/verbose))==0:
                    with torch.no_grad():
                        u = self.device.underlying_model(x_rayleigh)
                    if plot_solution:
                        plt.plot(x_rayleigh[:, 0].detach().cpu().view(-1), u[:, 0].detach().cpu().view(-1))
                        plt.show()
                    if calc_error:
                        mode_errors = utils.compare_against_analytic(self.device, x_rayleigh[:, [0]].clone().detach().cpu().numpy(), u)
                        for i in range(num_modes):
                            mode_errors_list[i].append(mode_errors[i])

        if verbose == 10:
          loop.close()

        return loss_history, rayleigh_history, mode_errors_list