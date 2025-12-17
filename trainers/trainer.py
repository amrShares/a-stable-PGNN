import torch
import numpy as np
import torch.nn as nn
import tqdm
import utils
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, dtype, device, memory_type,samplers, optimzer, scheduler, dx=None, num_samples=None, n_lower=None, n_upper=None):
      self.samplers = samplers
      self.optimizer = optimzer
      self.scheduler = scheduler
      self.device = device
      self.dtype = dtype
      self.n_lower = n_lower
      self.n_upper = n_upper

      if self.n_upper is None:
        self.n_upper = self.device.n_core
      if self.n_lower is None:
        self.n_lower = max(self.device.n_cladding, self.device.n_substrate)
        self.n_lower += 0.1*(self.n_upper - self.n_lower)

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
        return torch.mean(torch.square(u_bd - u_pred_bd))

    def continuity_loss(self, x, u, rayleigh_values, num_modes):
        loss = [torch.zeros(1, 1, device=self.memory_type)]

        discontinuities_tensor = torch.tensor([[-self.device.core_length/2], [self.device.core_length/2]], dtype=self.dtype, device=self.memory_type, requires_grad=True).reshape(-1, 1)
        biases = torch.tensor([[self.dx], [-self.dx]])

        refractive_indices_squared_in = torch.square(self.device.refractive_index(discontinuities_tensor.detach()+biases, format='torch'))
        refractive_indices_squared_out = torch.square(self.device.refractive_index(discontinuities_tensor.detach()-biases, format='torch'))

        if self.device.augmented_input:
            interface_input_in = torch.cat((discontinuities_tensor, 1/refractive_indices_squared_in), dim=-1)
            interface_input_out = torch.cat((discontinuities_tensor, 1/refractive_indices_squared_out), dim=-1)
        else:
            interface_input_in = discontinuities_tensor + biases
            interface_input_out = discontinuities_tensor - biases

        in_vals = self.device.underlying_model(interface_input_in)
        out_vals = self.device.underlying_model(interface_input_out)

        for i in range(num_modes):
            in_val = in_vals[:, [i]]
            out_val = out_vals[:, [i]]

            grad_in = torch.autograd.grad(in_val.sum(), discontinuities_tensor, create_graph=True)[0]
            grad_out = torch.autograd.grad(out_val.sum(), discontinuities_tensor, create_graph=True)[0]

            if self.device.augmented_input:
                if self.device.continuous_field:
                    loss.append(torch.mean(torch.square(in_val - out_val)))
                else:
                    loss.append(torch.mean(torch.square(in_val - (refractive_indices_squared_out/refractive_indices_squared_in)*out_val)))
                if self.device.continuous_derivative:
                    loss.append(torch.mean(torch.square(grad_in - grad_out)))
                else:
                    loss.append(torch.mean(torch.square(grad_in - (refractive_indices_squared_in/refractive_indices_squared_out)*grad_out)))
            else:
                if not self.device.continuous_field:
                    loss.append(torch.mean(torch.square(in_val - (refractive_indices_squared_out/refractive_indices_squared_in)*out_val)))
                if not self.device.continuous_derivative:
                    loss.append(torch.mean(torch.square(grad_in - (refractive_indices_squared_in/refractive_indices_squared_out)*grad_out)))

        return sum(loss)

    def interior_loss(self, x_rayleigh, x_PDE, u, num_modes, sample_new=False, u_ref=[]):
        cosine_similarity_fn = nn.CosineSimilarity(dim=0, eps=1e-5)
        losses_1 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_2 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_3 = [torch.zeros(1, 1, device=self.memory_type)]
        losses_4 = [torch.zeros(1, 1, device=self.memory_type)]
        rayleigh_values = []
        new_samples = None
        n_squared = self.device.refractive_index(x_PDE[:, 0], format='torch').detach() ** 2

        # for optimization
        m = 1e1
        a = (self.n_upper**2 - self.n_lower**2)/(m-1)
        b = (m*self.n_lower**2 - self.n_upper**2)/(self.n_upper**2 - self.n_lower**2)

        for i in range(num_modes):
            du_dx = torch.autograd.grad(u[:, i].sum(), x_PDE, create_graph=True, retain_graph=True)[0][:, 0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), x_PDE, create_graph=True)[0][:, 0]

            if len(self.samplers) == 2:
              rayleigh = self.device.calculate_eigen_value(x_rayleigh, i)
            else:
              rayleigh = self.device.get_rayleigh_quotient(d2u_dx2, n_squared, u, i)

            rayleigh_values.append(rayleigh.detach().item())
            residuals = torch.square(d2u_dx2 + (n_squared - rayleigh) * u[:, i])
            loss_1 = torch.mean(residuals)
            losses_1.append(loss_1)
            if sample_new :
                new_samples = Trainer.adaptive_sampling(x_PDE, residuals, sample_new * len(x_rayleigh), self.device)

            energy = torch.sum(u[:, i] ** 2) * self.device.total_length / len(u[:])
            losses_2.append(1/torch.square(energy)/(i+1))
            if rayleigh < self.n_lower**2:
                losses_3.append((loss_1 + 1e1)*torch.square((rayleigh - self.n_upper**2)/(self.n_upper**2 - self.n_lower**2))/(i+1))
            else:
                losses_3.append(loss_1/(rayleigh/a - b)/(i+1))
            # losses_3.append((loss_1 + 1e1*(rayleigh < self.n_lower**2))/(rayleigh/a - b)/(i+1))

            for j in range(len(u_ref)):
                # losses_4.append(torch.mean(torch.abs(u[:, i]*u[:, j])))
                losses_4.append(torch.square(cosine_similarity_fn(u[:, 0], u_ref[j])))

        return sum(losses_1), sum(losses_2), sum(losses_3), sum(losses_4), rayleigh_values, new_samples

    def construct_training_data(self):
        x_PDE = self.samplers[0](self.device, self.num_samples if self.num_samples else self.dx, self.dtype, self.memory_type)
        x_PDE.requires_grad = True
        if self.device.augmented_input:
            n_squared = self.device.refractive_index(x_PDE[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            x_PDE = torch.cat((x_PDE, 1/n_squared), dim=1).to(self.memory_type)
        if len(self.samplers)==2:
          x_rayleigh = self.samplers[1](self.device, self.dx, self.dtype, self.memory_type)
          x_rayleigh.requires_grad=True
          if self.device.augmented_input:
              n_squared_rayleigh = self.device.refractive_index(x_rayleigh[:, 0], format='torch').detach().reshape(-1, 1) ** 2
              x_rayleigh = torch.cat((x_rayleigh, 1/n_squared_rayleigh), dim=1).to(self.memory_type)
        else:
          x_rayleigh = x_PDE

        print('num PDE samples', len(x_PDE))
        print('num rayleigh samples', len(x_rayleigh))
        return x_PDE, x_rayleigh


    def train(self, epochs, num_modes, weights=[1, 1, 1, 1, 1, 1], verbose=0, calc_error=False, plot_solution=False, sample_new=False, u_ref=[]):
        if verbose ==10:
            loop = tqdm.tqdm(range(epochs))
        else:
            loop = range(epochs)

        loss_history = []
        rayleigh_history = [[0] for _ in range(num_modes)]

        self.device.underlying_model.to(self.memory_type)
        self.device.underlying_model.to(self.dtype)

        x_PDE, x_rayleigh = self.construct_training_data()

        bd = x_PDE[[self.device.get_boundaries_index(x_PDE[:, 0])]][:, [0]].detach()
        if self.device.augmented_input:
            n_squared_bd = self.device.refractive_index(bd[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            bd = torch.cat((bd, 1/n_squared_bd), dim=1).to(self.memory_type)

        u_bd = torch.zeros(2, 1, device=self.memory_type, dtype=self.dtype)
        u_bd = u_bd.to(self.memory_type)

        mode_errors_list = []
        for i in range(num_modes):
            mode_errors_list.append([])

        converging = False

        for j in loop:
            # def closure():
            self.optimizer.zero_grad()
            noise = torch.randn_like(x_PDE)*(self.dx)/4
            if self.device.augmented_input:
                noise[:, 1] = 0
            u_pred = self.device.underlying_model(x_PDE + noise)
            u_pred_bd = self.device.underlying_model(bd)

            loss_1, loss_2, loss_3, loss_4, rayleigh_values, new_points = self.interior_loss(x_rayleigh, x_PDE, u_pred, num_modes, sample_new, u_ref)
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

            # return (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6)

            self.optimizer.step()
            self.scheduler.step()

            # for i in range(num_modes):
            #     if abs(rayleigh_history[i][-1] - sum(rayleigh_history[i][-6:-1])/5) < 1e-5 and not converging:
            #         print('Converging')
            #         converging = True
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] *= 0.1
            #
            #
            # for i in range(num_modes):
            #     if abs(rayleigh_history[i][-1] - sum(rayleigh_history[i][-6:-1])/5) < 1e-6:
            #         print('Converged')
            #         return loss_history, rayleigh_history, mode_errors_list


            if verbose > 0:
                if j%(int(1000/verbose))==0:
                    with torch.no_grad():
                        u = self.device.underlying_model(x_rayleigh)
                    if plot_solution:
                        for i in range(num_modes):
                            plt.subplot(num_modes, 1, i+1)
                            plt.plot(x_rayleigh[:, 0].detach().cpu().view(-1), u[:, i].cpu().view(-1))
                        plt.show()
                    if calc_error:
                        mode_errors = utils.compare_against_analytic(self.device, x_rayleigh[:, [0]].clone().detach().cpu().numpy(), u.cpu())
                        for i in range(num_modes):
                            mode_errors_list[i].append(mode_errors[i])
        if verbose == 10:
          loop.close()

        return loss_history, rayleigh_history, mode_errors_list, u_pred.to(self.memory_type).detach().squeeze()