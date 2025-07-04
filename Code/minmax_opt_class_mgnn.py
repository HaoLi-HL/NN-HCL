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
                 t_min=0, t_max=1, x_min=0, x_max=1, k_h_rf_times=1, u0=None, u1=None, u2=None, lambda_bc_static=0.9, alpha_p_coeffs=0, 
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[5000, 5000, 5000], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=10000, decay_rate_stage2=1e-3,
                 f_adaptive=1000, nu=0.001, res_visc=None, res=None, use_loss_elem_int=True, total_grade=3, k=2):
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
        self.f_adaptive = f_adaptive
        self.alpha = 0.9
        self.alpha_p_coeffs = alpha_p_coeffs
        self.nu = nu
        self.res_visc = res_visc
        self.res = res
        self.gen_ibc()
        self.gen_derivative()
        if use_loss_elem_int:
            self.compute_loss_eqn_opt_utheta = self.loss_elem_int
        else:
            self.compute_loss_eqn_opt_utheta = self.loss_elem_bdry
        self.grade_epochs = grade_epochs
        self.epoch_stage2 = epoch_stage2

        # Initialize optimizers for each grade of the first stage
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.optimizer_list = []
        for i in range(total_grade):
            self.model.add_grade()  # Add a new grade
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

    def gen_ibc(self):
        self.u_t0 = self.u0(self.x_bc)
        self.tx_t0 = jnp.stack([self.pwp.t_min * jnp.ones_like(self.x_bc), self.x_bc], axis=1)
        self.u_x0 = self.u1(self.t_bc)
        self.tx_x0 = jnp.stack([self.t_bc, self.pwp.x_min * jnp.ones_like(self.t_bc)], axis=1)
        self.u_x1 = self.u2(self.t_bc)
        self.tx_x1 = jnp.stack([self.t_bc, self.pwp.x_max * jnp.ones_like(self.t_bc)], axis=1)
        
    def gen_derivative(self):
        net_u = lambda tx: self.model.forward(tx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes=0))
        self.grad_u = jit(vmap(grad(net_u, argnums=0), in_axes=0))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=0), in_axes=0))

    def loss_kh_element(self, tri_idx, p_coeffs_all, params_list=None):
        quad_points = self.pwp.quad_points_tri[tri_idx]
        quad_weights = self.pwp.quad_weights_tri[tri_idx].flatten()
        # Use the model's forward method directly if no params_list are provided
        if params_list is None:
            u_eval = lambda tx: self.model.forward(tx).squeeze()
            u_quad = self.u_eval(quad_points)
        else:
            # Define u_eval based on provided params_list (for stage 2)
            u_eval = lambda X: self.model.forward_with_params(params_list, X).squeeze()
            u_quad = jit(vmap(u_eval, in_axes=0))(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        residual = self.res(u_quad, u_t, u_x).flatten()
        k_h_quad_all = (
            p_coeffs_all[:, 0][:, None] + 
            p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
            p_coeffs_all[:, 2][:, None] * quad_points[:, 1]
        )
        integrand_all = residual[None, :] * jnp.sign(u_quad[None, :] - k_h_quad_all)
        losses = jnp.sum(quad_weights[None, :] * integrand_all, axis=1)
        return losses

    def maximize_kh_vec(self, current_key):
        Nt = self.pwp.N_elements
        vertices_all = self.pwp.triangles
        u_values_all = self.u_eval(vertices_all.reshape(-1, 2)).reshape(Nt, 3, -1)
        A_all = jnp.stack([jnp.ones((Nt, 3)), vertices_all[:, :, 0], vertices_all[:, :, 1]], axis=-1)
        initial_coeffs_all = jnp.linalg.solve(A_all, u_values_all.reshape(Nt, 3, 1)).squeeze(-1)
        key, current_key = jax.random.split(current_key)
        keys = jax.random.split(key, Nt)
        perturbations_all = vmap(lambda k: self.sigma * jax.random.normal(k, (self.N_perturb, 3)))(keys)
        perturbed_coeffs_all = initial_coeffs_all[:, None, :] + perturbations_all
        losses_all = vmap(self.loss_kh_element, in_axes=(0, 0, None))(jnp.arange(Nt), perturbed_coeffs_all, None)
        best_indices = jnp.argmax(losses_all, axis=1)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        return best_coeffs_all, current_key

    def loss_elem_int(self, p_coeffs):
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        u_quad = self.u_eval(quad_points)
        grad_u_tx = self.grad_u(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = self.hessian_u(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual_visc = self.res_visc(u_quad, u_t, u_x, u_xx, self.h**2)
        residual = self.res(u_quad, u_t, u_x).flatten()
        p_coeffs_quad = p_coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad) + jnp.abs(self.h*residual_visc)
        interior_loss = jnp.dot(quad_weights, integrand)
        return interior_loss
    
    def loss_elem_bdry(self, p_coeffs):
        def f(u):
            return u**2 / 2
        
        def df_du(u):
            return u

        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        u_quad = self.u_eval(quad_points)
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
        
    def compute_L_ic(self):
        return jnp.mean((self.u_eval(self.tx_t0) - self.u_t0)**2)

    def compute_L_bc(self):
        return jnp.mean((self.u_eval(self.tx_x0) - self.u_x0)**2) + jnp.mean((self.u_eval(self.tx_x1) - self.u_x1)**2)

    def compute_loss_bc_opt_utheta(self):
        return self.compute_L_ic() + self.compute_L_bc()

    def compute_loss_opt_utheta_adaptive(self, p_coeffs, lambda_ic, lambda_bc, lambda_r):
        L_ic = self.compute_L_ic()
        L_bc = self.compute_L_bc()
        L_r = self.compute_loss_eqn_opt_utheta(p_coeffs)
        total_loss = (lambda_ic * L_ic + 
                      lambda_bc * L_bc + 
                      lambda_r * L_r)
        return total_loss, (L_r, L_ic + L_bc)

    def loss_eqn_stage1(self, p_coeffs, grade_idx):
        quad_points = self.pwp.quad_points
        u_quad = self.u_eval(quad_points)
        grad_u_tx = self.grad_u(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = self.hessian_u(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res(u_quad, u_t, u_x).flatten()
        p_coeffs_quad = p_coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad)
        loss_res = jnp.dot(self.pwp.quad_weights.flatten(), integrand)
        return loss_res

    def loss_eqn_stage2(self, p_coeffs, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params(params_list, X).squeeze()
        quad_points = self.pwp.quad_points
        u_quad = jit(vmap(u_eval))(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(u_eval))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res(u_quad, u_t, u_x).flatten()
        p_coeffs_quad = p_coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad)
        loss_res = jnp.dot(self.pwp.quad_weights.flatten(), integrand)
        return loss_res

    def loss_ic_stage2(self, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params(params_list, X).squeeze()
        loss_ic = jnp.mean((jit(vmap(u_eval))(self.tx_t0) - self.u_t0)**2)
        return loss_ic

    def loss_bc_stage2(self, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params(params_list, X).squeeze()
        u_x0_nn = jit(vmap(u_eval))(self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        u_x1_nn = jit(vmap(u_eval))(self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        loss_bc = bc_x0_loss + bc_x1_loss
        return loss_bc

    def train_loss_fn_stage1(self, p_coeffs, grade_idx):
        L_ic = self.compute_L_ic()
        L_bc = self.compute_L_bc()
        L_r = self.loss_eqn_stage1(p_coeffs, grade_idx)
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)

    def train_loss_fn_stage2(self, p_coeffs, params_last_k):
        L_ic = self.loss_ic_stage2(params_last_k)
        L_bc = self.loss_bc_stage2(params_last_k)
        L_r = self.loss_eqn_stage2(p_coeffs, params_last_k)
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)

    def train_step_stage1(self, p_coeffs, grade_idx, opt_state):
        if grade_idx >= len(self.model.mlps):
            raise ValueError(f"Grade index {grade_idx} exceeds number of MLPs.")
        (_, all_losses), grads = jax.value_and_grad(self.train_loss_fn_stage1, argnums=0, has_aux=True)(p_coeffs, grade_idx)
        updates, opt_state = self.optimizer_list[grade_idx].update(grads, opt_state)
        params = optax.apply_updates(self.model.mlps[grade_idx].params, updates)
        return all_losses[0], all_losses[1], params, opt_state

    def train_step_stage2(self, p_coeffs, params_last_k, opt_state):
        (_, all_losses), grads = jax.value_and_grad(self.train_loss_fn_stage2, argnums=1, has_aux=True)(p_coeffs, params_last_k)
        updates, opt_state = self.optimizer_stage2.update(grads, opt_state)
        params_last_k = optax.apply_updates(params_last_k, updates)
        return all_losses[0], all_losses[1], params_last_k, opt_state

    def update_adaptive_weights(self, phase='train', grade_idx=None, params_last_k=None):
        if phase == 'train':
            if params_last_k is not None:
                grad_L_ic = grad(self.loss_ic_stage2)(params_last_k)
                grad_L_bc = grad(self.loss_bc_stage2)(params_last_k)
                grad_L_int = grad(lambda p: self.loss_eqn_stage2(p, params_last_k))(self.pwp.coeffs)
            else:
                grad_L_ic = grad(self.compute_L_ic)()
                grad_L_bc = grad(self.compute_L_bc)()
                grad_L_int = grad(lambda p: self.loss_eqn_stage1(p, grade_idx))(self.pwp.coeffs)
        else:
            raise ValueError("Unknown phase for update_adaptive_weights.")
        norm_ic = pytree_l2_norm(grad_L_ic)
        norm_bc = pytree_l2_norm(grad_L_bc)
        norm_r = pytree_l2_norm(grad_L_int)
        epsilon = 1e-7
        sum_norm = norm_ic + norm_bc + norm_r + 3*epsilon
        hat_lambda_ic = jnp.clip(sum_norm / (norm_ic + epsilon), a_min=1, a_max=1e7)
        hat_lambda_bc = jnp.clip(sum_norm / (norm_bc + epsilon), a_min=1, a_max=1e7)
        hat_lambda_r = jnp.clip(sum_norm / (norm_r + epsilon), a_min=1, a_max=1e7)
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * hat_lambda_ic
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * hat_lambda_bc
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * hat_lambda_r

    def train_process(self, train_num_iterations, save_model=True):
        print("\nStarting Train Stage with MGDL")
        start_time = time.perf_counter()
        self.loss_eqn_history = []
        self.loss_bc_history = []
        self.maximize_kh = jit(self.maximize_kh_vec)
        self.global_step = 0

        # Stage 1: Sequential Grade Training
        print("\nStage 1: Sequential Grade Training")
        for grade_idx in range(self.total_grade):
            print(f"\nTraining Grade {grade_idx}")
            params = self.model.mlps[grade_idx].params
            opt_state = self.opt_state_list[grade_idx]
            self.pwp.coeffs, self.key = self.maximize_kh(self.key)
            for iter in range(self.grade_epochs[grade_idx]):
                if iter % self.f_adaptive == 0:
                    self.update_adaptive_weights(phase='train', grade_idx=grade_idx)
                    print(f"Grade {grade_idx}, Iter {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                if iter % 10 == 0:
                    hat_coeffs, self.key = self.maximize_kh(self.key)
                    self.pwp.coeffs = self.alpha_p_coeffs * self.pwp.coeffs + (1 - self.alpha_p_coeffs) * hat_coeffs
                loss_eqn, loss_ibc, params, opt_state = self.train_step_stage1(self.pwp.coeffs, grade_idx, opt_state)
                self.model.mlps[grade_idx].params = params
                self.opt_state_list[grade_idx] = opt_state
                self.loss_eqn_history.append(float(loss_eqn))
                self.loss_bc_history.append(float(loss_ibc))
                if iter % 100 == 0:
                    print(f"Grade {grade_idx}, Iter {iter}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
            self.global_step += self.grade_epochs[grade_idx]

        # Stage 2: Refinement of Last k Grades
        if self.k > 0 and self.k <= self.total_grade:
            print(f"\nStage 2: Refining Last {self.k} Grades")
            params_last_k = [mlp.params for mlp in self.model.mlps[-self.k:]]
            self.opt_state_stage2 = self.optimizer_stage2.init(params_last_k)
            self.pwp.coeffs, self.key = self.maximize_kh(self.key)
            for iter in range(self.epoch_stage2):
                if iter % self.f_adaptive == 0:
                    self.update_adaptive_weights(phase='train', params_last_k=params_last_k)
                    print(f"Stage 2, Iter {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                if iter % 10 == 0:
                    hat_coeffs, self.key = self.maximize_kh(self.key)
                    self.pwp.coeffs = self.alpha_p_coeffs * self.pwp.coeffs + (1 - self.alpha_p_coeffs) * hat_coeffs
                loss_eqn, loss_ibc, params_last_k, self.opt_state_stage2 = self.train_step_stage2(self.pwp.coeffs, params_last_k, self.opt_state_stage2)
                for mlp, p in zip(self.model.mlps[-self.k:], params_last_k):
                    mlp.params = p
                self.loss_eqn_history.append(float(loss_eqn))
                self.loss_bc_history.append(float(loss_ibc))
                if iter % 100 == 0:
                    print(f"Stage 2, Iter {iter}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
            self.global_step += self.epoch_stage2

        end_time = time.perf_counter()
        train_total_time = end_time - start_time
        print(f"Train total time: {train_total_time:.8f} seconds")
        print(f"Train average time per iteration: {train_total_time / (sum(self.grade_epochs) + self.epoch_stage2):.9e} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_model:
            model_path = os.path.join('..', 'data', f'train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Train model saved to {model_path}")

    def save_model(self, model_path):
        save_dict = {
            'params': [mlp.params for mlp in self.model.mlps],
            'hidden_dims_list': self.model.hidden_dims_list
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        loaded_params = loaded_data['params']
        loaded_hidden_dims_list = loaded_data['hidden_dims_list']
        self.model = MG_NN(input_dim=2, hidden_dims_list=loaded_hidden_dims_list, activation=nn.tanh, lb=self.lb, ub=self.ub)
        for mlp, params in zip(self.model.mlps, loaded_params):
            mlp.params = params
        print(f"Model loaded from {model_path} with hidden_dims_list: {loaded_hidden_dims_list}")
        self.opt_state_list = [opt.init(mlp.params) for opt, mlp in zip(self.optimizer_list, self.model.mlps)]
        self.pwp.coeffs, self.key = self.maximize_kh_vec(self.key)
        loss_eqn = self.compute_loss_eqn_opt_utheta(self.pwp.coeffs)
        loss_bc = self.compute_loss_bc_opt_utheta()
        print(f"For the loaded model, loss_eqn: {loss_eqn:.8f}, Loss_bc: {loss_bc:.8f}")