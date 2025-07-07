import os
import numpy as np
import jax.numpy as jnp
import jax
from scipy import sparse
import time
from jax import grad, hessian, jit, vmap, pmap
from jax.tree_util import tree_leaves
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import flax.linen as nn
import optax
import optax.tree_utils as otu
from NN_class import MLP
from MGNN_class import MLP, MG_NN
from my_helpers import plot_u_theta, plot_kh
from functools import partial
from Pwp_class import PiecewiseLinearPolynomials
import wandb

# Utility function to compute L2 norm of a pytree
def pytree_l2_norm(pytree):
    leaves = tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))

# Optimizer Class
class MinMaxOptProb:
    def __init__(self, N_t=10, N_x=10, lr_theta=0.01, hidden_dims_list=[(64, 64), (32, 32), (16, 16)], seed=0, sigma=0.1, N_perturb=5000, N_bc=100, N_ic=100,
                 t_min=0, t_max=1, x_min=0, x_max=1, k_h_rf_times=1, u0=None, u1=None, u2=None, lambda_bc_static=0.9, project="minmax_opt_mgnn_burgers", 
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[10, 10, 10], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=100, decay_rate_stage2=1e-3,
                 f_adaptive_weight=1000, f_opt_kh=10, nu=0.001, res_visc=None, res=None, use_loss_fn_elem_int=True, total_grade=3, k=2):
        self.pwp = PiecewiseLinearPolynomials(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, times=k_h_rf_times)
        self.lr_theta = lr_theta
        self.lb = jnp.array([t_min, x_min])
        self.ub = jnp.array([t_max, x_max])
        self.h = (x_max - x_min)/N_x
        self.model = MG_NN(input_dim=2, hidden_dims_list=hidden_dims_list, activation=nn.tanh, lb=self.lb, ub=self.ub, seed=seed)
        self.total_grade = total_grade  # Number of grades for MGDL
        self.k = k  # Number of grades to refine in second stage
        self.key = jax.random.PRNGKey(seed)
        self.sigma = sigma
        self.N_perturb = N_perturb
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.t_bc = jnp.linspace(t_min, t_max, self.N_bc)
        self.x_bc = jnp.linspace(x_min, x_max, self.N_ic)
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.lambda_bc_static = lambda_bc_static
        self.lambda_ic = 10.0
        self.lambda_bc = 10.0
        self.lambda_r = 1.0
        self.f_adaptive_weight = f_adaptive_weight
        self.f_opt_kh = f_opt_kh
        self.alpha = 0.9
        self.nu = nu
        self.res_visc = res_visc
        self.res = res
        self.gen_ibc()
        if use_loss_fn_elem_int:
            self.loss_fn_eqn = self.loss_fn_elem_int
        else:
            self.loss_fn_eqn = self.loss_fn_elem_bdry
        self.grade_epochs = grade_epochs
        self.epoch_stage2 = epoch_stage2
        
        self.project = project

        # Initialize optimizers for each grade of the first stage
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.optimizer_list = []
        self.params_list = []
        for i in range(total_grade):
            self.model.add_grade()  # Add a new grade
            self.params_list.append(self.model.mlps[i].params)
            lr = optax.exponential_decay(init_value=init_lrs[i], transition_steps=grade_epochs[i], decay_rate=decay_rates[i])
            self.optimizer_list.append(optax.adam(lr))
        self.opt_state_list = [opt.init(self.model.mlps[i].params) for i, opt in enumerate(self.optimizer_list)]
        # Optimizer for the second stage
        self.init_lr_stage2 = init_lr_stage2
        self.epoch_stage2 = epoch_stage2
        self.decay_rate_stage2 = decay_rate_stage2
        lr_stage2 = optax.exponential_decay(init_value=init_lr_stage2, transition_steps=epoch_stage2, decay_rate=decay_rate_stage2)
        self.optimizer_stage2 = optax.adam(lr_stage2)
        self.opt_state_stage2 = None  # To be initialized later

        self.gen_funcs()

    def gen_ibc(self):
        self.u_t0 = self.u0(self.x_bc)
        self.tx_t0 = jnp.stack([self.pwp.t_min * jnp.ones_like(self.x_bc), self.x_bc], axis=1)
        self.u_x0 = self.u1(self.t_bc)
        self.tx_x0 = jnp.stack([self.t_bc, self.pwp.x_min * jnp.ones_like(self.t_bc)], axis=1)
        self.u_x1 = self.u2(self.t_bc)
        self.tx_x1 = jnp.stack([self.t_bc, self.pwp.x_max * jnp.ones_like(self.t_bc)], axis=1)
        
    def gen_funcs(self, grade_idx=None):
        if grade_idx is None:
            net_u = lambda params_last_k, X: self.model.forward_with_params_last_k(params_last_k, X, self.k).squeeze()
        else:
            net_u = lambda params, X: self.model.forward_with_params(params, X, grade_idx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes = (None, 0)))
        self.grad_u = jit(vmap(grad(net_u, argnums=1), in_axes = (None, 0)))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=1), in_axes = (None, 0)))

    def loss_kh_per_elem(self, tri_idx, p_coeffs_all, params):
        quad_points = self.pwp.quad_points_tri[tri_idx]
        quad_weights = self.pwp.quad_weights_tri[tri_idx].flatten()
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        residual = self.res(u_quad, u_t, u_x).flatten()
        k_h_quad_all = (
            p_coeffs_all[:, 0][:, None] + 
            p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
            p_coeffs_all[:, 2][:, None] * quad_points[:, 1]
        )
        integrand_all = residual[None, :] * jnp.sign(u_quad[None, :] - k_h_quad_all)
        loss = jnp.sum(quad_weights[None, :] * integrand_all, axis=1)
        return loss

    
    def maximize_kh_vec(self, params, current_key):
        Nt = self.pwp.N_elements
        vertices_all = self.pwp.triangles
        u_values_all = self.u_eval(params, vertices_all.reshape(-1, 2)).reshape(Nt, 3, -1)
        A_all = jnp.stack([jnp.ones((Nt, 3)), vertices_all[:, :, 0], vertices_all[:, :, 1]], axis=-1)
        init_coeffs_all = jnp.linalg.solve(A_all, u_values_all.reshape(Nt, 3, 1)).squeeze(-1)
        key, current_key = jax.random.split(current_key)
        keys = jax.random.split(key, Nt)
        perturbations_all = vmap(lambda k: self.sigma * jax.random.normal(k, (self.N_perturb, 3)))(keys)
        perturbed_coeffs_all = init_coeffs_all[:, None, :] + perturbations_all
        losses_all = vmap(self.loss_kh_per_elem, in_axes=(0, 0, None))(jnp.arange(Nt), perturbed_coeffs_all, params)
        best_indices = jnp.argmax(losses_all, axis=1)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        return best_coeffs_all, current_key

    def loss_fn_elem_int(self, params):
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        # hessian_u_tx = self.hessian_u(params, quad_points)
        # u_xx = hessian_u_tx[:, 1, 1].flatten()
        # residual_visc = self.res_visc(u_quad, u_t, u_x, u_xx, self.h**2)
        residual = self.res(u_quad, u_t, u_x).flatten()
        p_coeffs_quad = self.pwp.coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad)
        # integrand = residual * jnp.sign(u_quad - k_h_quad) + jnp.abs(self.h*residual_visc)
        interior_loss = jnp.dot(quad_weights, integrand)
        return interior_loss
    
    def loss_fn_elem_bdry(self, params, p_coeffs):
        def f(u):
            return u**2 / 2
        
        def df_du(u):
            return u

        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        u_quad = self.u_eval(params, quad_points)
        p_coeffs_all = p_coeffs[self.pwp.element_indices]
        k_h_quad = (p_coeffs_all[:, 0] + 
                    p_coeffs_all[:, 1] * quad_points[:, 0] + 
                    p_coeffs_all[:, 2] * quad_points[:, 1])
        b_K = p_coeffs_all[:, 1]
        c_K = p_coeffs_all[:, 2]
        residual_kh = b_K + df_du(k_h_quad) * c_K
        sgn = jnp.sign(u_quad - k_h_quad)
        integrand_int = residual_kh * sgn
        interior_loss = jnp.dot(quad_weights, integrand_int)

        N_points_per_edge = 2**self.pwp.bdry_times + 1
        points_per_edge = self.pwp.bdry_points_tri.reshape(self.pwp.N_elements, 3, N_points_per_edge, 2)
        weights_per_edge = self.pwp.bdry_weights_tri.reshape(self.pwp.N_elements, 3, N_points_per_edge)
        normals_per_edge = self.pwp.normals

        def compute_edge_integral(points, weights, normal, p_coeffs_K):
            u_theta = self.u_eval(points)
            k_h = (p_coeffs_K[0] + 
                   p_coeffs_K[1] * points[:, 0] + 
                   p_coeffs_K[2] * points[:, 1])
            sgn = jnp.sign(u_theta - k_h)
            f_u = f(u_theta)
            f_k = f(k_h)
            flux_diff = ((u_theta - k_h) * normal[0] + 
                        (f_u - f_k) * normal[1])
            integrand = flux_diff * sgn
            return jnp.sum(weights * integrand)

        def compute_triangle_boundary_loss(points_tri, weights_tri, normals_tri, p_coeffs_K):
            edge_losses = vmap(compute_edge_integral, in_axes=(0, 0, 0, None))(
                points_tri, weights_tri, normals_tri, p_coeffs_K)
            return jnp.sum(edge_losses)

        boundary_loss = jnp.sum(vmap(compute_triangle_boundary_loss, in_axes=(0, 0, 0, 0))(
                                points_per_edge, weights_per_edge, normals_per_edge, p_coeffs))
        total_loss = interior_loss + boundary_loss
        return total_loss
        
    def loss_fn_ic(self, params):
        return jnp.mean((self.u_eval(params, self.tx_t0) - self.u_t0)**2)

    def loss_fn_bc(self, params):
        return jnp.mean((self.u_eval(params, self.tx_x0) - self.u_x0)**2) + jnp.mean((self.u_eval(params, self.tx_x1) - self.u_x1)**2)

    def loss_fn_ibc(self, params):
        return self.loss_fn_ic(params) + self.loss_fn_bc(params)

    def loss_fn(self, params):
        loss_ic = self.loss_fn_ic(params)
        loss_bc = self.loss_fn_bc(params)
        loss_eqn = self.loss_fn_elem_int(params)
        total_loss = self.lambda_r * loss_eqn + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc
        return total_loss, (loss_eqn, loss_ic + loss_bc)

    def train_step(self, params, optimizar, opt_state):
        (_, all_losses), grads = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)(params)
        updates, opt_state = optimizar.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return all_losses[0], all_losses[1], params, opt_state

    def train_process(self, save_model=False):
        print("\nStarting Train Stage with MGDL")
        start_time = time.perf_counter()
        self.global_step = 0
        wandb.init(project=self.project, name=f"total_grade_{self.total_grade}_k_{self.k}", config={
            "total_grade": self.total_grade,
            "k": self.k,
            "hidden_dims_list": self.model.hidden_dims_list,
            "lrs": self.init_lrs,
            "epochs": self.grade_epochs,
            "decay_rates": self.decay_rates,
            "lr_2ndstage": self.init_lr_stage2,
            "epoch_2ndstage": self.epoch_stage2,
            "decay_rate": self.decay_rate_stage2,
            "f_adaptive_weight": self.f_adaptive_weight,
            "f_opt_kh" : self.f_opt_kh,
            })

        # Stage 1: Sequential Grade Training
        print("\nStage 1: Sequential Grade Training")
        for grade_idx in range(self.total_grade):
            print(f"\nTraining Grade {grade_idx}")
            self.gen_funcs(grade_idx)
            self.maximize_kh = jit(self.maximize_kh_vec)
            params = self.model.mlps[grade_idx].params
            optimizar = self.optimizer_list[grade_idx]
            opt_state = self.opt_state_list[grade_idx]
            self.pwp.coeffs, self.key = self.maximize_kh(params, self.key)
            for iter in range(self.grade_epochs[grade_idx]):
                if iter % self.f_adaptive_weight == 0:
                    self.update_adaptive_weights(params=params)
                    print(f"Grade {grade_idx}, Iter {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                if iter % self.f_opt_kh == 0:
                    self.pwp.coeffs, self.key = self.maximize_kh(params, self.key)
                loss_eqn, loss_ibc, params, opt_state = self.train_step(params, optimizar, opt_state)
                if iter % 100 == 0:
                    print(f"Grade {grade_idx}, Iter {iter}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
                    wandb.log({f"grade_{grade_idx}_loss_eqn": loss_eqn, f"grade_{grade_idx}_loss_ibc": loss_ibc, "step": self.global_step + iter})
            self.model.mlps[grade_idx].params = params
            self.opt_state_list[grade_idx] = opt_state
            self.global_step += self.grade_epochs[grade_idx]

        # Stage 2: Refinement of Last k Grades
        if self.k > 0 and self.k <= self.total_grade:
            print(f"\nStage 2: Refining Last {self.k} Grades")
            self.gen_funcs()
            self.maximize_kh = jit(self.maximize_kh_vec)
            params_last_k = [mlp.params for mlp in self.model.mlps[-self.k:]]
            self.opt_state_stage2 = self.optimizer_stage2.init(params_last_k)
            optimizar = self.optimizer_stage2
            opt_state = self.opt_state_stage2
            self.pwp.coeffs, self.key = self.maximize_kh(params_last_k, self.key)
            for iter in range(self.epoch_stage2):
                if iter % self.f_adaptive_weight == 0:
                    self.update_adaptive_weights(params=params_last_k)
                    print(f"Stage 2, Iter {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                if iter % 10 == 0:
                    self.pwp.coeffs, self.key = self.maximize_kh(params_last_k, self.key)
                loss_eqn, loss_ibc, params_last_k, opt_state = self.train_step(params_last_k, optimizar, opt_state)
                if iter % 100 == 0:
                    print(f"Stage 2, Iter {iter}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
                    wandb.log({"second_stage_loss": loss_eqn, "second_stage_loss_ibc": loss_ibc, "step": self.global_step + iter})
            for mlp, p in zip(self.model.mlps[-self.k:], params_last_k):
                mlp.params = p
            self.global_step += self.epoch_stage2

        end_time = time.perf_counter()
        train_total_time = end_time - start_time
        print(f"Train total time: {train_total_time:.8f} seconds")
        print(f"Train average time per iteration: {train_total_time / (sum(self.grade_epochs) + self.epoch_stage2):.9e} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        wandb.finish()
        if save_model:
            model_path = os.path.join('data', f'train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Train model saved to {model_path}")
        
    def update_adaptive_weights(self, params):
        grad_loss_ic = grad(self.loss_fn_ic)(params)
        grad_loss_bc = grad(self.loss_fn_bc)(params)
        grad_loss_int = grad(self.loss_fn_elem_int)(params)
        norm_ic = pytree_l2_norm(grad_loss_ic)
        norm_bc = pytree_l2_norm(grad_loss_bc)
        norm_r = pytree_l2_norm(grad_loss_int)
        epsilon = 1e-7
        sum_norm = norm_ic + norm_bc + norm_r + 3*epsilon
        hat_lambda_ic = jnp.clip(sum_norm / (norm_ic + epsilon), a_min=1, a_max=1e7)
        hat_lambda_bc = jnp.clip(sum_norm / (norm_bc + epsilon), a_min=1, a_max=1e7)
        hat_lambda_r = jnp.clip(sum_norm / (norm_r + epsilon), a_min=1, a_max=1e7)
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * hat_lambda_ic
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * hat_lambda_bc
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * hat_lambda_r

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_dict = {
            'params': [mlp.params for mlp in self.model.mlps],
            'hidden_dims_list': self.model.hidden_dims_list,
            'total_grade': self.total_grade,
            'lb': self.lb,
            'ub': self.ub,
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        loaded_params = loaded_data['params']
        loaded_hidden_dims_list = loaded_data['hidden_dims_list']
        lb = loaded_data['lb']
        ub = loaded_data['ub']
        self.model = MG_NN(input_dim=2, hidden_dims_list=loaded_hidden_dims_list, activation=nn.tanh, lb=lb, ub=ub)
        for mlp, params in zip(self.model.mlps, loaded_params):
            mlp.params = params
        print(f"Model loaded from {model_path} with hidden_dims_list: {loaded_hidden_dims_list}")
        self.opt_state_list = [opt.init(mlp.params) for opt, mlp in zip(self.optimizer_list, self.model.mlps)]
        self.pwp.coeffs, self.key = self.maximize_kh_vec(self.key)
        loss_eqn = self.loss_fn_eqn(loaded_params, self.pwp.coeffs)
        loss_ibc = self.loss_fn_ibc(loaded_params)
        print(f"For the loaded model, loss_eqn: {loss_eqn:.8f}, Loss_bc: {loss_ibc:.8f}")