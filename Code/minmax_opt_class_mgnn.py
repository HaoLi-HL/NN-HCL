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

# Utility function to compute L2 norm of a pytree
def pytree_l2_norm(pytree):
    leaves = tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))

# Optimizer Class
class MinMaxOptProb:
    def __init__(self, N_t=10, N_x=10, lr_theta=0.01, hidden_dims_list=[(64, 64), (32, 32), (16, 16)], seed=0, sigma=0.1, N_perturb=5000, N_bc=100, N_ic=100,
                 t_min=0, t_max=1, x_min=0, x_max=1, k_h_rf_times=1, u0=None, u1=None, u2=None, lambda_bc_static=0.9, alpha_p_coeffs=0, 
                 init_lrs=None, grade_epochs=None, decay_rates=None, init_lr_stage2=1e-3, epoch_stage2=5e4, decay_rate_stage2=1e-4,
                 f_adaptive=1000, nu=0.001, res_visc=None, res=None, use_loss_elem_int=True, k=2):
        self.pwp = PiecewiseLinearPolynomials(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, times=k_h_rf_times)
        self.lr_theta = lr_theta
        self.lb = jnp.array([t_min, x_min])
        self.ub = jnp.array([t_max, x_max])
        self.h = (x_max - x_min)/N_x
        self.model = MG_NN(input_dim=2, hidden_dims_list=hidden_dims_list, activation=nn.tanh, lb=self.lb, ub=self.ub, seed=seed)
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
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
        self.lambda_r = 1.0
        self.f_adaptive = f_adaptive
        self.alpha = 0.9
        self.alpha_p_coeffs = alpha_p_coeffs
        self.nu = nu
        self.res_visc = res_visc
        self.res = res
        self.k = k  # Number of grades to unfreeze in the second stage
        self.gen_ibc()
        self.gen_derivative()
        if use_loss_elem_int:
            self.compute_loss_eqn_opt_utheta = self.compute_loss_elem_int_opt_utheta
        else:
            self.compute_loss_eqn_opt_utheta = self.compute_loss_elem_bdry_opt_utheta
        
        # Initialize optimizers for each grade of the first stage
        total_grade = len(self.model.hidden_dims_list)
        self.opt_theta_list = []
        for i in range(total_grade):
            lr = optax.exponential_decay(init_value=init_lrs[i], transition_steps=grade_epochs[i], decay_rate=decay_rates[i])
            self.opt_theta_list.append(optax.adam(lr))
        self.opt_state_theta_list = [opt.init(self.model.mlps[i].params) for i, opt in enumerate(self.opt_theta_list)]
        # Optimizer for the second stage
        lr_stage2 = optax.exponential_decay(init_value=init_lr_stage2, transition_steps=epoch_stage2, decay_rate=decay_rate_stage2)
        self.opt_theta_second_stage = optax.adam(lr_stage2)
        self.opt_state_theta_second_stage = None  # To be initialized later

    def gen_ibc(self):
        self.u_t0 = self.u0(self.x_bc)
        self.tx_t0 = jnp.stack([self.pwp.t_min * jnp.ones_like(self.x_bc), self.x_bc], axis=1)
        self.u_x0 = self.u1(self.t_bc)
        self.tx_x0 = jnp.stack([self.t_bc, self.pwp.x_min * jnp.ones_like(self.t_bc)], axis=1)
        self.u_x1 = self.u2(self.t_bc)
        self.tx_x1 = jnp.stack([self.t_bc, self.pwp.x_max * jnp.ones_like(self.t_bc)], axis=1)
        
    def gen_derivative(self):
        net_u = lambda params, tx: self.model(params, tx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes = (None, 0)))
        self.grad_u = jit(vmap(grad(net_u, argnums=1), in_axes = (None, 0)))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=1), in_axes = (None, 0)))

    def compute_loss_opt_kh_element(self, tri_idx, p_coeffs_all, params):
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
        losses = jnp.sum(quad_weights[None, :] * integrand_all, axis=1)
        return losses

    def maximize_kh_vec(self, params, current_key):
        Nt = self.pwp.N_elements
        vertices_all = self.pwp.triangles
        u_values_all = self.u_eval(params, vertices_all.reshape(-1, 2)).reshape(Nt, 3, -1)
        A_all = jnp.stack([jnp.ones((Nt, 3)), vertices_all[:, :, 0], vertices_all[:, :, 1]], axis=-1)
        initial_coeffs_all = jnp.linalg.solve(A_all, u_values_all.reshape(Nt, 3, 1)).squeeze(-1)
        key, current_key = jax.random.split(current_key)
        keys = jax.random.split(key, Nt)
        perturbations_all = vmap(lambda k: self.sigma * jax.random.normal(k, (self.N_perturb, 3)))(keys)
        perturbed_coeffs_all = initial_coeffs_all[:, None, :] + perturbations_all
        losses_all = vmap(self.compute_loss_opt_kh_element, in_axes=(0, 0, None))(jnp.arange(Nt), perturbed_coeffs_all, params)
        best_indices = jnp.argmax(losses_all, axis=1)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        return best_coeffs_all, current_key

    def compute_loss_elem_int_opt_utheta(self, params, p_coeffs):
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = self.hessian_u(params, quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual_visc = self.res_visc(u_quad, u_t, u_x, u_xx, self.h**2)
        residual = self.res(u_quad, u_t, u_x).flatten()
        p_coeffs_quad = p_coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad) + jnp.abs(self.h*residual_visc)
        interior_loss = jnp.dot(quad_weights, integrand)
        return interior_loss
    
    def compute_loss_elem_bdry_opt_utheta(self, params, p_coeffs):
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
        integrand_int = residual_kh
        interior_loss = jnp.dot(quad_weights, integrand_int)
        N_points_per_edge = 2**self.pwp.bdry_times + 1
        points_per_edge = self.pwp.bdry_points_tri.reshape(self.pwp.N_elements, 3, N_points_per_edge, 2)
        weights_per_edge = self.pwp.bdry_weights_tri.reshape(self.pwp.N_elements, 3, N_points_per_edge)
        normals_per_edge = self.pwp.normals
        def compute_edge_integral(points, weights, normal, p_coeffs_K, params):
            u_theta = self.u_eval(params, points)
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
        def compute_triangle_boundary_loss(points_tri, weights_tri, normals_tri, p_coeffs_K, params):
            edge_losses = vmap(compute_edge_integral, in_axes=(0, 0, 0, None, None))(
                points_tri, weights_tri, normals_tri, p_coeffs_K, params)
            return jnp.sum(edge_losses)
        boundary_loss = jnp.sum(vmap(compute_triangle_boundary_loss, in_axes=(0, 0, 0, 0, None))(
                                points_per_edge, weights_per_edge, normals_per_edge, p_coeffs, params))
        total_loss = interior_loss + boundary_loss
        return total_loss
        
    def compute_loss_bc_opt_utheta(self, params):
        return self.compute_L_ic(params) + self.compute_L_bc(params)

    def compute_loss_opt_utheta_adaptive(self, params, p_coeffs, lambda_ic, lambda_bc, lambda_r):
        L_ic = self.compute_L_ic(params)
        L_bc = self.compute_L_bc(params)
        L_r = self.compute_loss_eqn_opt_utheta(params, p_coeffs)
        total_loss = (lambda_ic * L_ic + 
                      lambda_bc * L_bc + 
                      lambda_r * L_r)
        return total_loss, (L_r, L_ic+L_bc)
        
    def compute_loss_opt_utheta(self, params, p_coeffs):
        interior_loss = self.compute_loss_eqn_opt_utheta(params, p_coeffs)
        boundary_loss = self.compute_loss_bc_opt_utheta(params)
        total_loss = (1-self.lambda_bc_static) * interior_loss + self.lambda_bc_static * boundary_loss
        return total_loss

    def minimize_u_theta(self, params, p_coeffs, opt_state_theta):
        loss_value, grad_params = jax.value_and_grad(self.compute_loss_opt_utheta, argnums=0)(params, p_coeffs)
        updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
        params = optax.apply_updates(params, updates)
        return loss_value, params, opt_state_theta

    def minimize_u_theta_adaptive(self, params, p_coeffs, opt_state_theta, lambda_ic, lambda_bc, lambda_r):
        (_, L_all), grad_params = jax.value_and_grad(self.compute_loss_opt_utheta_adaptive, 
                                                     argnums=0, has_aux=True)(params, p_coeffs, lambda_ic, lambda_bc, lambda_r)
        updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
        params = optax.apply_updates(params, updates)
        return L_all, params, opt_state_theta

    # MGDL-specific methods for pre-training
    def total_u_up_to_grade(self, params_list, grade_idx, X):
        temp = X
        sum_out = 0
        for j in range(grade_idx):
            out, out_wo_layer = self.model.mlps[j].forward(params_list[j], temp, extra_output=True)
            sum_out += out
            temp = out_wo_layer
        return sum_out

    def pre_train_loss_int(self, params_list, grade_idx):
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights.flatten()
        total_u = lambda X: self.total_u_up_to_grade(params_list, grade_idx, X)
        u_quad = vmap(total_u)(quad_points)
        grad_u_tx = vmap(grad(total_u))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(total_u))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)
        integrand = residual**2
        interior_loss = jnp.dot(quad_weights, integrand)
        return interior_loss

    def loss_ic(self, params_list, grade_idx):
        total_u = lambda X: self.total_u_up_to_grade(params_list, grade_idx, X)
        u_ic_pred = vmap(total_u)(self.tx_t0)
        return jnp.mean((u_ic_pred - self.u_t0)**2)

    def loss_bc(self, params_list, grade_idx):
        total_u = lambda X: self.total_u_up_to_grade(params_list, grade_idx, X)
        u_x0_nn = vmap(total_u)(self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        u_x1_nn = vmap(total_u)(self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        return bc_x0_loss + bc_x1_loss

    def pre_train_loss(self, params_grade_i, grade_idx):
        params_list = [self.model.mlps[j].params for j in range(grade_idx - 1)] + [params_grade_i]
        L_r = self.pre_train_loss_int(params_list, grade_idx)
        L_ic = self.loss_ic(params_list, grade_idx)
        L_bc = self.loss_bc(params_list, grade_idx)
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)

    def pre_train_minimize_u_theta_adaptive(self, params_grade_i, grade_idx, opt_state_theta):
        (_, L_all), grads = jax.value_and_grad(self.pre_train_loss, argnums=0, has_aux=True)(params_grade_i, grade_idx)
        updates, opt_state_theta = self.opt_theta_list[grade_idx-1].update(grads, opt_state_theta)
        params_grade_i = optax.apply_updates(params_grade_i, updates)
        return L_all, params_grade_i, opt_state_theta

    def pre_train_step(self, iter, grade_idx):
        if iter % self.f_adaptive == 0:
            self.update_adaptive_weights_for_grade(grade_idx)
            print(f"Grade {grade_idx}, Iteration {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
        (loss_eqn, loss_ibc), self.model.mlps[grade_idx-1].params, self.opt_state_theta_list[grade_idx-1] = \
        self.pre_train_minimize_u_theta_adaptive(self.model.mlps[grade_idx-1].params, grade_idx, self.opt_state_theta_list[grade_idx-1])
        
        (_, L_all), grads = jax.value_and_grad(self.pre_train_loss, argnums=0, has_aux=True)(params_grade_i, grade_idx)
        updates, opt_state_theta = self.opt_theta_list[grade_idx-1].update(grads, opt_state_theta)
        params_grade_i = optax.apply_updates(params_grade_i, updates)
        return loss_eqn, loss_ibc

    def pre_train_process(self, pre_train_num_iterations, save_loss=True, save_model=True):
        print("\nStarting Pre-Train Stage with MGDL")
        start_time = time.perf_counter()
        total_grade = len(self.model.hidden_dims_list)
        iterations_per_grade = pre_train_num_iterations // (total_grade + 1)  # +1 for second stage
        self.lambda_ic = 10.0
        self.lambda_bc = 10.0
        self.lambda_r = 1.0
        self.pre_loss_eqn_history = []
        self.pre_loss_bc_history = []

        global_iter = 0
        for grade_idx in range(1, total_grade + 1):
            print(f"\nTraining Grade {grade_idx}")
            for iter in range(iterations_per_grade):
                loss_eqn, loss_ibc = self.pre_train_step(global_iter, grade_idx)
                self.pre_loss_eqn_history.append(loss_eqn)
                self.pre_loss_bc_history.append(loss_ibc)
                if iter % 200 == 0:
                    print(f"Grade {grade_idx}, Iteration {iter}, loss_eqn: {loss_eqn:.8f}, Loss_ibc: {loss_ibc:.8f}")
                global_iter += 1

        # Second stage: Unfreeze and train last k grades
        if self.k > 0 and self.k <= total_grade:
            print(f"\nSecond stage: Unfreezing last {self.k} grades")
            params_last_k = [self.model.mlps[-i].params for i in range(1, self.k+1)]
            self.opt_state_theta_second_stage = self.opt_theta_second_stage.init(params_last_k)
            for iter in range(iterations_per_grade):
                (loss_eqn, loss_ibc), params_last_k, self.opt_state_theta_second_stage = self.second_stage_minimize_u_theta_adaptive(params_last_k, self.opt_state_theta_second_stage)
                self.pre_loss_eqn_history.append(loss_eqn)
                self.pre_loss_bc_history.append(loss_ibc)
                if iter % 200 == 0:
                    print(f"Second stage, Iteration {iter}, loss_eqn: {loss_eqn:.8f}, Loss_ibc: {loss_ibc:.8f}")
            # Update the parameters of the last k grades
            for i, params in enumerate(params_last_k):
                self.model.mlps[-i-1].params = params

        end_time = time.perf_counter()
        pre_train_total_time = end_time - start_time
        print(f"Pre-train total time: {pre_train_total_time:.8f} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_loss:
            loss_eqn_path = os.path.join('..', 'data', f'pre_train_loss_eqn_history_{timestamp}.pkl')
            with open(loss_eqn_path, 'wb') as f:
                pickle.dump(self.pre_loss_eqn_history, f)
            loss_bc_path = os.path.join('..', 'data', f'pre_train_loss_bc_history_{timestamp}.pkl')
            with open(loss_bc_path, 'wb') as f:
                pickle.dump(self.pre_loss_bc_history, f)
            print(f"Pre-train loss histories saved to {loss_eqn_path} and {loss_bc_path}")
        
        if save_model:
            model_path = os.path.join('..', 'data', f'pre_train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Pre-train model saved to {model_path}")

    # Second stage methods
    def second_stage_compute_loss_opt_utheta_adaptive(self, params_last_k):
        total_grade = len(self.model.hidden_dims_list)
        params_list = [self.model.mlps[j].params for j in range(total_grade - self.k)] + list(params_last_k)
        L_r = self.pre_train_loss_int(params_list, total_grade)
        L_ic = self.loss_ic(params_list, total_grade)
        L_bc = self.loss_bc(params_list, total_grade)
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)

    def second_stage_minimize_u_theta_adaptive(self, params_last_k, opt_state_theta):
        (_, L_all), grads = jax.value_and_grad(self.second_stage_compute_loss_opt_utheta_adaptive, 
                                               argnums=0, has_aux=True)(params_last_k)
        updates, opt_state_theta = self.opt_theta_second_stage.update(grads, opt_state_theta)
        params_last_k = optax.apply_updates(params_last_k, updates)
        return L_all, params_last_k, opt_state_theta

    def compute_L_ic(self, params):
        return jnp.mean((self.u_eval(params, self.tx_t0) - self.u_t0)**2)

    def compute_L_bc(self, params):
        u_x0_nn = self.u_eval(params, self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        u_x1_nn = self.u_eval(params, self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        return bc_x0_loss + bc_x1_loss

    def update_adaptive_weights(self, params, p_coeffs, phase='pre-train'):
        grad_L_ic = grad(self.compute_L_ic)(params)
        grad_L_bc = grad(self.compute_L_bc)(params)
        if phase == 'pre-train':
            grad_L_int = grad(lambda p: self.pre_train_compute_loss_int_opt_utheta(p))(params)
        elif phase == 'train':
            grad_L_int = grad(lambda p: self.compute_loss_eqn_opt_utheta(p, p_coeffs))(params)
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

    def update_adaptive_weights_for_grade(self, grade_idx):
        params_up_to_grade = [self.model.mlps[j].params for j in range(grade_idx)]
        grad_L_ic = grad(self.loss_ic, argnums=0)(params_up_to_grade, grade_idx)
        grad_L_bc = grad(self.loss_bc, argnums=0)(params_up_to_grade, grade_idx)
        grad_L_r = grad(self.pre_train_loss_int, argnums=0)(params_up_to_grade, grade_idx)
        norm_ic = pytree_l2_norm(grad_L_ic)
        norm_bc = pytree_l2_norm(grad_L_bc)
        norm_r = pytree_l2_norm(grad_L_r)
        epsilon = 1e-7
        sum_norm = norm_ic + norm_bc + norm_r + 3*epsilon
        hat_lambda_ic = jnp.clip(sum_norm / (norm_ic + epsilon), a_min=1, a_max=1e7)
        hat_lambda_bc = jnp.clip(sum_norm / (norm_bc + epsilon), a_min=1, a_max=1e7)
        hat_lambda_r = jnp.clip(sum_norm / (norm_r + epsilon), a_min=1, a_max=1e7)
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * hat_lambda_ic
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * hat_lambda_bc
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * hat_lambda_r

    def train_step(self, iter, moving_avg=True):
        if iter % self.f_adaptive == 0:
            self.update_adaptive_weights(self.params, self.pwp.coeffs, phase='train')
            print(f"Iteration {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
        if iter % 10 == 0:
            print(f"Iteration {iter}, before maximize k_h, loss: {self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_bc_opt_utheta(self.params):.8f}")
        hat_coeffs, self.key = self.maximize_kh(self.params, self.key)
        self.pwp.coeffs = self.alpha_p_coeffs * self.pwp.coeffs + (1 - self.alpha_p_coeffs) * hat_coeffs
        if iter % 10 == 0:
            print(f"Iteration {iter}, after maximize k_h, loss: {self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_bc_opt_utheta(self.params):.8f}")
        (loss_eqn, loss_ibc), self.params, self.opt_state_theta = self.minimize_u(self.params, self.pwp.coeffs, self.opt_state_theta, 
                                                                                 self.lambda_ic, self.lambda_bc, self.lambda_r)
        return loss_eqn, loss_ibc

    def train_process(self, train_num_iterations, save_loss=True, save_model=True):
        print("\nStarting Train Stage")
        start_time = time.perf_counter()
        self.lambda_ic = 10.0
        self.lambda_bc = 10.0
        self.lambda_r = 1.0
        self.loss_eqn_history = []
        self.loss_bc_history = []
        self.maximize_kh = jit(self.maximize_kh_vec)
        self.minimize_u = jit(self.minimize_u_theta_adaptive)
        self.pwp.coeffs, self.key = self.maximize_kh(self.params, self.key)
        for iter in range(train_num_iterations):
            loss_eqn, loss_bc = self.train_step(iter)
            self.loss_eqn_history.append(loss_eqn)
            self.loss_bc_history.append(loss_bc)
            if iter % 100 == 0:
                print(f"Train Iteration {iter}, loss_eqn: {loss_eqn:.8f}, Loss_bc: {loss_bc:.8f}")
        end_time = time.perf_counter()
        train_total_time = (end_time - start_time)
        print(f"Train total time: {train_total_time:.8f} seconds")
        print(f"Train average time per iteration: {train_total_time / train_num_iterations:.9e} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_loss:
            loss_eqn_path = os.path.join('..', 'data', f'loss_eqn_history_{timestamp}.pkl')
            with open(loss_eqn_path, 'wb') as f:
                pickle.dump(self.loss_eqn_history, f)
            loss_bc_path = os.path.join('..', 'data', f'loss_bc_history_{timestamp}.pkl')
            with open(loss_bc_path, 'wb') as f:
                pickle.dump(self.loss_bc_history, f)
            print(f"Loss histories saved to {loss_eqn_path} and {loss_bc_path}")
        if save_model:
            model_path = os.path.join('..', 'data', f'train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Train model saved to {model_path}")

    def save_model(self, model_path):
        save_dict = {
            'params': self.params,
            'hidden_dims': self.model.hidden_dims
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        loaded_params = loaded_data['params']
        loaded_hidden_dims = loaded_data['hidden_dims']
        self.model = MLP(hidden_dims=loaded_hidden_dims, activation=nn.tanh, lb=self.lb, ub=self.ub)
        self.params = loaded_params
        self.model.params = loaded_params
        print(f"Model loaded from {model_path} with hidden_dims: {loaded_hidden_dims}")
        self.opt_theta = optax.adam(self.lr_theta)
        self.opt_state_theta = self.opt_theta.init(self.params)
        self.pwp.coeffs, self.key = self.maximize_kh_vec(self.params, self.key)
        loss_eqn, loss_bc = self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs), self.compute_loss_bc_opt_utheta(self.params)
        print(f"For the loaded model, loss_eqn: {loss_eqn:.8f}, Loss_bc: {loss_bc:.8f}")