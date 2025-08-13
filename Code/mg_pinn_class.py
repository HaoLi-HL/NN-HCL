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
from PwpTri_class import PiecewiseLinearPolynomialsTri
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
        self.pwp = PiecewiseLinearPolynomialsTri(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, times=k_h_rf_times)
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
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
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
            self.compute_loss_eqn_opt_utheta = self.compute_loss_elem_int_opt_utheta
        else:
            self.compute_loss_eqn_opt_utheta = self.compute_loss_elem_bdry_opt_utheta
        self.grade_epochs = grade_epochs
        self.epoch_stage2 = epoch_stage2

        # Initialize optimizers for each grade of the first stage
        total_grade = len(self.model.hidden_dims_list)
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.optimizer_list = []
        for i in range(total_grade):
            self.model.add_grade()
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
        # Redefine derivatives to use MG_NN's forward method without explicit params
        net_u = lambda tx: self.model.forward(tx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes=0))
        self.grad_u = jit(vmap(grad(net_u, argnums=0), in_axes=0))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=0), in_axes=0))

    def compute_loss_opt_kh_element(self, tri_idx, p_coeffs_all, params):
        quad_points = self.pwp.quad_points_tri[tri_idx]
        quad_weights = self.pwp.quad_weights_tri[tri_idx].flatten()
        u_quad = self.u_eval(quad_points)
        grad_u_tx = self.grad_u(quad_points)
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
        u_values_all = self.u_eval(vertices_all.reshape(-1, 2)).reshape(Nt, 3, -1)
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
    
    def compute_loss_elem_bdry_opt_utheta(self, params, p_coeffs):
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

        def compute_edge_integral(points, weights, normal, p_coeffs_K, params):
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

    def minimize_u_theta(self, params, p_coeffs, opt_state):
        loss_value, grad_params = jax.value_and_grad(self.compute_loss_opt_utheta, argnums=0)(params, p_coeffs)
        updates, opt_state = self.optimizer.update(grad_params, opt_state)
        params = optax.apply_updates(params, updates)
        return loss_value, params, opt_state

    def minimize_u_theta_adaptive(self, params, p_coeffs, opt_state, lambda_ic, lambda_bc, lambda_r):
        (_, L_all), grad_params = jax.value_and_grad(self.compute_loss_opt_utheta_adaptive, 
                                                     argnums=0, has_aux=True)(params, p_coeffs, lambda_ic, lambda_bc, lambda_r)
        updates, opt_state = self.optimizer.update(grad_params, opt_state)
        params = optax.apply_updates(params, updates)
        return L_all, params, opt_state

    def pre_train_loss_fn_stage1(self, params, grade_idx):
        """Loss function for first stage of MGDL pre-training."""
        params_list = [self.model.mlps[j].params for j in range(grade_idx)] + [params]
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        # IC loss
        L_ic = jnp.mean((u_eval(self.tx_t0) - self.u_t0)**2)
        # BC loss
        L_bc = (jnp.mean((u_eval(self.tx_x0) - self.u_x0)**2) + 
                jnp.mean((u_eval(self.tx_x1) - self.u_x1)**2))
        # Interior loss
        quad_points = self.pwp.quad_points
        u_quad = u_eval(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(u_eval))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)
        L_r = jnp.dot(self.pwp.quad_weights.flatten(), residual**2)
        # Weighted total loss
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)


    def pre_train_loss_fn_stage2(self, params_last_k):
        """Loss function for second stage of MGDL pre-training."""
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        # IC loss
        L_ic = jnp.mean((u_eval(self.tx_t0) - self.u_t0)**2)
        # BC loss
        L_bc = (jnp.mean((u_eval(self.tx_x0) - self.u_x0)**2) + 
                jnp.mean((u_eval(self.tx_x1) - self.u_x1)**2))
        # Interior loss
        quad_points = self.pwp.quad_points
        u_quad = u_eval(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(u_eval))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1]
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)
        L_r = jnp.dot(self.pwp.quad_weights.flatten(), residual**2)
        # Weighted total loss
        total_loss = self.lambda_r * L_r + self.lambda_ic * L_ic + self.lambda_bc * L_bc
        return total_loss, (L_r, L_ic + L_bc)
        
    def pre_train_process(self, save_model=False):
        """Pre-train the model using MGDL framework."""
        print("\nStarting Pre-Train Stage with MGDL")
        start_time = time.perf_counter()
        self.global_step = 0

        wandb.init(project="burgers_mgdl", name=f"total_grade_{self.total_grade}_k_{self.k}", config={
            "total_grade": self.total_grade,
            "k": self.k,
            "hidden_dims_list": self.model.hidden_dims_list,
            "lrs": self.init_lrs,
            "epochs": self.grade_epochs,
            "lr_2ndstage": self.init_lr_stage2,
            "epoch_2ndstage": self.epoch_stage2,
            "decay_rate": self.decay_rate_stage2,
            })
        # First Stage: Train each grade sequentially
        for grade_idx in range(self.total_grade):
            print(f"\nTraining Grade {grade_idx}")
            params = self.model.mlps[grade_idx].params
            optimizer = self.optimizer_list[grade_idx]
            opt_state = self.opt_state_list[grade_idx]
            for iter in range(self.grade_epochs[grade_idx]):
                if iter % self.f_adaptive == 0:
                    self.update_adaptive_weights(params=params, phase='pre-train', grade_idx=grade_idx)
                    print(f"Grade {grade_idx}, Epoch {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                    
                (_, all_losses), grads = jax.value_and_grad(self.pre_train_loss_fn_stage1, argnums=0, has_aux=True)(params, grade_idx)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                if iter % 100 == 0:
                    loss_eqn, loss_ibc = all_losses
                    loss = loss_eqn + loss_ibc
                    print(f"Grade {grade_idx}, Epoch {iter}, loss: {loss:.8f}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
                    wandb.log({f"grade_{grade_idx}_loss_eqn": loss_eqn, f"grade_{grade_idx}_loss_ibc": loss_ibc, "step": self.global_step + iter})
            self.model.mlps[grade_idx].params = params
            self.global_step += self.grade_epochs[grade_idx]

        # Second Stage: Refine the last k grades
        if self.k > 0 and self.k <= self.total_grade:
            print(f"\nSecond Stage: Refining last {self.k} grades")
            params_last_k = [mlp.params for mlp in self.model.mlps[-self.k:]]
            optimizer = self.optimizer_stage2
            opt_state = optimizer.init(params_last_k)
            for iter in range(self.epoch_stage2):
                if iter % self.f_adaptive == 0:
                    self.update_adaptive_weights(params=None, phase='pre-train', params_last_k=params_last_k)
                    print(f"Second Stage, Epoch {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}")
                (_, all_losses), grads = jax.value_and_grad(self.pre_train_loss_fn_stage2, argnums=0, has_aux=True)(params_last_k)
                updates, opt_state = optimizer.update(grads, opt_state)
                params_last_k = optax.apply_updates(params_last_k, updates)
                if iter % 100 == 0:
                    loss_eqn, loss_ibc = all_losses
                    loss = loss_eqn + loss_ibc
                    print(f"Second Stage, Epoch {iter}, loss: {loss:.8f}, loss_eqn: {loss_eqn:.8f}, loss_ibc: {loss_ibc:.8f}")
                    wandb.log({"second_stage_loss": loss_eqn, "second_stage_loss_ibc": loss_ibc, "step": self.global_step + iter})
            for mlp, params in zip(self.model.mlps[-self.k:], params_last_k):
                mlp.params = params
            self.global_step += self.epoch_stage2

        end_time = time.perf_counter()
        pre_train_total_time = end_time - start_time
        print(f"Pre-train total time: {pre_train_total_time:.8f} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.finish()
        
        if save_model:
            model_path = os.path.join('..', 'data', f'pre_train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Pre-train model saved to {model_path}")
        

    def loss_ic_stage1(self, params, grade_idx):
        params_list = [self.model.mlps[j].params for j in range(grade_idx)] + [params]
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        loss_ic = jnp.mean((u_eval(self.tx_t0) - self.u_t0)**2)
        return loss_ic

    def loss_bc_stage1(self, params, grade_idx):
        params_list = [self.model.mlps[j].params for j in range(grade_idx)] + [params]
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        u_x0_nn = u_eval(self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        u_x1_nn = u_eval(self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        loss_bc = bc_x0_loss + bc_x1_loss
        return loss_bc

    def loss_ic_stage2(self, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        loss_ic = jnp.mean((u_eval(self.tx_t0) - self.u_t0)**2)
        return loss_ic
    
    def loss_bc_stage2(self, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        u_x0_nn = u_eval(self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        u_x1_nn = u_eval(self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        loss_bc = bc_x0_loss + bc_x1_loss
        return loss_bc
    
    def loss_eqn_stage1(self, params, grade_idx):
        params_list = [self.model.mlps[j].params for j in range(grade_idx)] + [params]
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        # Interior loss
        quad_points = self.pwp.quad_points
        u_quad = u_eval(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(u_eval))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)
        loss_res = jnp.dot(self.pwp.quad_weights.flatten(), residual**2)
        return loss_res
    
    def loss_eqn_stage2(self, params_last_k):
        n = len(self.model.mlps)
        fixed_params = [self.model.mlps[i].params for i in range(n - self.k)]
        params_list = fixed_params + list(params_last_k)
        u_eval = lambda X: self.model.forward_with_params_list(params_list, X)
        # Interior loss
        quad_points = self.pwp.quad_points
        u_quad = u_eval(quad_points)
        grad_u_tx = vmap(grad(u_eval))(quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = vmap(hessian(u_eval))(quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)
        loss_res = jnp.dot(self.pwp.quad_weights.flatten(), residual**2)
        return loss_res

    def update_adaptive_weights(self, params=None, p_coeffs=None, phase='pre-train', grade_idx=None, params_last_k=None):
        # For pre-train stage 1: use params and grade_idx
        # For pre-train stage 2: use params_last_k
        # For train: use params and p_coeffs
        if phase == 'pre-train':
            if params_last_k is not None:
                # Stage 2: params_last_k is a list/tuple of params for last k grades
                grad_L_ic = grad(self.loss_ic_stage2)(params_last_k)
                grad_L_bc = grad(self.loss_bc_stage2)(params_last_k)
                grad_L_int = grad(self.loss_eqn_stage2)(params_last_k)
            else:
                # Stage 1: params and grade_idx
                grad_L_ic = grad(self.loss_ic_stage1)(params, grade_idx)
                grad_L_bc = grad(self.loss_bc_stage1)(params, grade_idx)
                grad_L_int = grad(self.loss_eqn_stage1)(params, grade_idx)
        elif phase == 'train':
            grad_L_ic = grad(self.compute_L_ic)(params)
            grad_L_bc = grad(self.compute_L_bc)(params)
            grad_L_int = grad(lambda p: self.compute_loss_eqn_opt_utheta(p, p_coeffs))(params)
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

    def train_step(self, iter, moving_avg=True):
        if iter % self.f_adaptive == 0:
            self.update_adaptive_weights(self.params, self.pwp.coeffs, phase='train')
            print(f"Iteration {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}.")
        if iter % 10 == 0:
            print(f"Iteration {iter}, before maximize k_h, loss: {self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_bc_opt_utheta(self.params):.8f}.")
        hat_coeffs, self.key = self.maximize_kh(self.params, self.key)
        self.pwp.coeffs = self.alpha_p_coeffs * self.pwp.coeffs + (1 - self.alpha_p_coeffs) * hat_coeffs
        if iter % 10 == 0:
            print(f"Iteration {iter}, after maximize k_h, loss: {self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_bc_opt_utheta(self.params):.8f}.")
        (loss_eqn, loss_ibc), self.params, self.opt_state = self.minimize_u(self.params, self.pwp.coeffs, self.opt_state, 
                                                                                 self.lambda_ic, self.lambda_bc, self.lambda_r)
        return loss_eqn, loss_ibc

    def train_process(self, train_num_iterations, save_model=True):
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
        self.optimizer = optax.adam(self.lr_theta)
        self.opt_state = self.optimizer.init(self.params)
        self.pwp.coeffs, self.key = self.maximize_kh_vec(self.params, self.key)
        loss_eqn, loss_bc = self.compute_loss_eqn_opt_utheta(self.params, self.pwp.coeffs), self.compute_loss_bc_opt_utheta(self.params)
        print(f"For the loaded model, loss_eqn: {loss_eqn:.8f}, Loss_bc: {loss_bc:.8f}.")