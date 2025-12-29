import torch
import numpy as np
import torch.nn as nn
import tqdm
import utils
import matplotlib.pyplot as plt
import functools

class LMTrainer:
    def __init__(self, dtype, guide, device,samplers, optimzer, scheduler, dx=None, num_samples=None, n_lower=None, n_upper=None):
      self.samplers = samplers
      self.optimizer = optimzer
      self.scheduler = scheduler
      self.guide = guide
      self.dtype = dtype
      self.n_lower = n_lower
      self.n_upper = n_upper

      if self.n_lower is None:
        self.n_lower = max(self.guide.n_cladding, self.guide.n_substrate)
      if self.n_upper is None:
        self.n_upper = self.guide.n_core

      self.dx = dx
      self.num_samples = num_samples
      self.device = device

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
        loss = [torch.zeros(1, 1, device=self.device)]

        discontinuities_tensor = torch.tensor([[-self.guide.core_length/2], [self.guide.core_length/2]], dtype=self.dtype, device=self.device, requires_grad=True).reshape(-1, 1)
        
        refractive_indices_squared = torch.tensor([[self.guide.n_substrate], [self.guide.n_core], [self.guide.n_cladding]], device=self.device, dtype=self.dtype).reshape(-1, 1) ** 2

        for i in range(len(discontinuities_tensor)):
            for j in range(num_modes):
                if self.guide.augmented_input:
                    interface_input_l = torch.cat((discontinuities_tensor[[i]], refractive_indices_squared[[i]]), dim=-1)
                else:
                    interface_input_l = discontinuities_tensor[[i]]-self.dx
                left_val = self.guide.underlying_model(interface_input_l)[:, j]

                if self.guide.augmented_input:
                    interface_input_r = torch.cat((discontinuities_tensor[[i]], refractive_indices_squared[[i+1]]), dim=-1)
                else:
                    interface_input_r = discontinuities_tensor[[i]]+self.dx
                right_val = self.guide.underlying_model(interface_input_r)[:, j]

                grad_left = torch.autograd.grad(left_val, discontinuities_tensor, create_graph=True)[0]
                grad_right = torch.autograd.grad(right_val, discontinuities_tensor, create_graph=True)[0]
                      
                if self.guide.augmented_input:
                    if self.guide.continuous_field:
                        loss.append(torch.abs(left_val - right_val))
                    else:
                        loss.append(torch.abs(refractive_indices_squared[i]*left_val - refractive_indices_squared[i+1]*right_val))
                    if self.guide.continuous_derivative:
                        loss.append(torch.sum(torch.abs(grad_right[i] - grad_left[i])))
                    else:
                        loss.append(torch.sum(torch.abs(refractive_indices_squared[i]*grad_right[i] - refractive_indices_squared[i+1]*grad_left[i])))
                else:
                    if not self.guide.continuous_field:
                        loss.append(torch.abs(refractive_indices_squared[i]*left_val - refractive_indices_squared[i+1]*right_val))
                    if not self.guide.continuous_derivative:
                        loss.append(torch.sum(torch.abs(refractive_indices_squared[i]*grad_right[i] - refractive_indices_squared[i+1]*grad_left[i])))


        return sum(loss)

    def interior_loss(self, x_rayleigh, x_PDE, u, num_modes, sample_new=False):
        cosine_similarity_fn = nn.CosineSimilarity(dim=0, eps=1e-6)
        losses_1 = [torch.zeros(1, 1, device=self.device)]
        losses_2 = [torch.zeros(1, 1, device=self.device)]
        losses_3 = [torch.zeros(1, 1, device=self.device)]
        losses_4 = [torch.zeros(1, 1, device=self.device)]
        rayleigh_values = []
        new_samples = None
        n_squared = self.guide.refractive_index(x_PDE[:, 0], format='torch').detach() ** 2
        
        for i in range(num_modes):
            du_dx = torch.autograd.grad(u[:, i].sum(), x_PDE, create_graph=True, retain_graph=True)[0][:, 0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), x_PDE, create_graph=True)[0][:, 0]

            if len(self.samplers) == 2:
              rayleigh = self.guide.calculate_eigen_value(x_rayleigh, i)
            else:
              rayleigh = self.guide.get_rayleigh_quotient(d2u_dx2, n_squared, u, i)
            rayleigh_values.append(rayleigh.detach().item())

            if not sample_new :
                losses_1.append(torch.mean(torch.abs(d2u_dx2 + (n_squared - rayleigh) * u[:, i])))
            else:
                residuals = torch.abs(d2u_dx2 + (n_squared - rayleigh) * u[:, i])
                new_samples = LMTrainer.adaptive_sampling(x_PDE, residuals, sample_new * len(x_rayleigh), self.guide)
                losses_1.append(torch.mean(residuals))

            losses_2.append(torch.square(torch.sum(u[:, i] ** 2) * self.guide.total_length / len(u[:]) - 1))
            losses_3.append(torch.maximum(self.n_lower ** 2 - rayleigh, torch.zeros_like(rayleigh)) + torch.maximum(rayleigh - self.n_upper ** 2, torch.zeros_like(rayleigh)) - rayleigh/(i+1))

            for j in range(i + 1, num_modes):
                losses_4.append(torch.square(cosine_similarity_fn(u[:, i], u[:, j])))

        return sum(losses_1), sum(losses_2), sum(losses_3), sum(losses_4), rayleigh_values, new_samples

    def construct_training_data(self):
        x_PDE = self.samplers[0](self.guide, self.num_samples if self.num_samples else self.dx, self.dtype, self.device)
        x_PDE.requires_grad = True
        if self.guide.augmented_input:
            n_squared = self.guide.refractive_index(x_PDE[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            x_PDE = torch.cat((x_PDE, n_squared), dim=1).to(self.device)
        if len(self.samplers)==2:
          x_rayleigh = self.samplers[1](self.guide, self.dx, self.dtype, self.device)
          x_rayleigh.requires_grad=True
          if self.guide.augmented_input:
              n_squared_rayleigh = self.guide.refractive_index(x_rayleigh[:, 0], format='torch').detach().reshape(-1, 1) ** 2
              x_rayleigh = torch.cat((x_rayleigh, n_squared_rayleigh), dim=1).to(self.device)
        else:
          x_rayleigh = x_PDE

        print('num PDE samples', len(x_PDE))
        print('num rayleigh samples', len(x_rayleigh))
        return x_PDE, x_rayleigh

    @staticmethod
    def eval_hessian_(loss_grad, model):
        def get_vjp(v):
            grad2rd = torch.autograd.grad(g_vector, model.parameters(), v,create_graph=False, retain_graph=True, allow_unused=True)
            grad2rd = torch.cat(([g.view(-1) for g in grad2rd]))
            return grad2rd.detach()
        g_vector = torch.cat(([g.view(-1) for g in loss_grad]))
        l = g_vector.size(0)
        I_N = torch.eye(l, device=g_vector.device)
        return torch.vmap(get_vjp, chunk_size=256)(I_N), g_vector.unsqueeze(1).detach()

    def train(self, epochs, num_modes, weights=[1, 1, 1, 1, 1, 1], verbose=0, calc_error=False, plot_solution=False, sample_new=False):
        if verbose ==10:
            loop = tqdm.tqdm(range(epochs))
        else:
            loop = range(epochs)

        loss_history = []
        rayleigh_history = [[] for _ in range(num_modes)]

        self.guide.underlying_model.to(self.device)
        self.guide.underlying_model.to(self.dtype)

        x_PDE, x_rayleigh = self.construct_training_data()

        bd = x_PDE[[self.guide.get_boundaries_index(x_PDE[:, 0])]][:, [0]].detach()
        if self.guide.augmented_input:
            n_squared_bd = self.guide.refractive_index(bd[:, 0], format='torch').detach().reshape(-1, 1) ** 2
            bd = torch.cat((bd, n_squared_bd), dim=1).to(self.device)

        u_bd = torch.zeros(2, 1, device=self.device, dtype=self.dtype)
        u_bd = u_bd.to(self.device)

        mode_errors_list = []
        for i in range(num_modes):
            mode_errors_list.append([])

        alpha=1e1

        for j in loop:

            self.optimizer.zero_grad()
            u_pred = self.guide.underlying_model(x_PDE)
            u_pred_bd = self.guide.underlying_model(bd)

            loss_1, loss_2, loss_3, loss_4, rayleigh_values, new_points = self.interior_loss(x_rayleigh, x_PDE, u_pred, num_modes, sample_new)
            loss_5 = self.bd_loss(u_bd, u_pred_bd)
            loss_6 = self.continuity_loss(x_rayleigh, u_pred, rayleigh_values, num_modes)

            loss = (weights[0]*loss_1 + weights[1]*loss_2+ weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6)
            prev_loss = loss.item()

            gradients=torch.autograd.grad(loss, self.guide.underlying_model.parameters(), create_graph=True)
            Hessian, g_vector = LMTrainer.eval_hessian_(gradients, self.guide.underlying_model)

            self.guide.underlying_model.eval()
            dx=-(alpha*torch.eye(Hessian.shape[-1]).to(self.device)+Hessian).inverse().mm(g_vector).detach()

            cnt=0
            self.guide.underlying_model.zero_grad()
            for p in self.guide.underlying_model.parameters():
                mm=torch.Tensor([p.shape]).tolist()[0]
                num=int(functools.reduce(lambda x,y:x*y,mm,1))
                p.requires_grad=False
                p+=dx[cnt:cnt+num,:].reshape(p.shape)
                cnt+=num
                p.requires_grad=True

            u_pred = self.guide.underlying_model(x_PDE)
            u_pred_bd = self.guide.underlying_model(bd)

            loss_1, loss_2, loss_3, loss_4, rayleigh_values, new_points = self.interior_loss(x_rayleigh, x_PDE, u_pred, num_modes, sample_new)
            loss_5 = self.bd_loss(u_bd, u_pred_bd)
            loss_6 = self.continuity_loss(x_rayleigh, u_pred, rayleigh_values, num_modes)

            loss = (weights[0]*loss_1 + weights[1]*loss_2+weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6).item()

            if loss<prev_loss:
                success = 'success'
                loss_history.append(loss)
                alpha/=3
            else:
                success = 'failure'
                alpha*=5
                cnt=0
                for p in self.guide.underlying_model.parameters():
                    mm=torch.Tensor([p.shape]).tolist()[0]
                    num=int(functools.reduce(lambda x,y:x*y,mm,1))
                    p.requires_grad=False
                    p-=dx[cnt:cnt+num,:].reshape(p.shape)
                    cnt+=num
                    p.requires_grad=True

            for i in range(num_modes):
                rayleigh_history[i].append(rayleigh_values[i])

            if verbose == 10:
                loop.set_description(f'loss1 is {round(loss_1.detach().item(), 5)}, loss2 is {round(loss_2.detach().item(), 5)}, loss3 is {round(loss_3.detach().item(), 5)}, loss4 is {round(loss_4.detach().item(), 5)}, loss5 is {round(loss_5.detach().item(), 5)}, loss6 is {round(loss_6.detach().item(), 5)}')
            elif verbose > 0:
                print((f'iteration {j+1} loss1 is {round(loss_1.detach().item(), 5)}, loss2 is {round(loss_2.detach().item(), 5)}, loss3 is {round(loss_3.detach().item(), 5)}, loss4 is {round(loss_4.detach().item(), 5)}, loss5 is {round(loss_5.detach().item(), 5)}, loss6 is {round(loss_6.detach().item(), 5)}'))

            if verbose > 0:
                with torch.no_grad():
                    u = self.guide.underlying_model(x_rayleigh)
                if plot_solution:
                    plt.plot(x_rayleigh[:, 0].detach().cpu().view(-1), u[:, 0].cpu().view(-1))
                    plt.show()
                if calc_error:
                    mode_errors = utils.compare_against_analytic(self.guide, x_rayleigh[:, [0]].clone().detach().cpu().numpy(), u.cpu())
                    for i in range(num_modes):
                        mode_errors_list[i].append(mode_errors[i])
        if verbose == 10:
          loop.close()

        return loss_history, rayleigh_history, mode_errors_list