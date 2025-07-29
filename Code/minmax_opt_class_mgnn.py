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
    def __init__(self, t_min=0, t_max=1, x_min=0, x_max=1, N_t=10, N_x=10, N_bc=128, N_ic=256, hidden_dims_list=[(64, 64, 64, 64)], seed=0, 
                 sigma=0.1, N_perturb=10000, k_h_rf_times=2, f_adaptive_weight=1000, f_opt_kh=1, u0=None, u1=None, u2=None, project="minmax_mgnn_burgers",
                 lambda_ic=10.0, lambda_bc=10.0, lambda_eqn=1.0, gamma=0.0, alpha=0.9, N_test_gamma = 1000, adjust_gamma=True,
                 L1reg=1, sampling_method='normal', notes='', lambda_tv=1.0, N_t_pnts=None, N_x_pnts=None, pre_epochs=1000,
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[10, 10, 10], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=100, decay_rate_stage2=1e-3,
                 res=None, res_visc=None, use_loss_fn_elem_int=True, total_grade=1, k=0, use_a=True,
                 seq_train=False, set_wandb=True):
        self.seq_train = seq_train  # Sequential training flag
        self.set_wandb = set_wandb
        # if self.seq_train is True, then we will not create a new wandb run but just use the existing one.
        # if self.seq_train is False and self.set_wandb is True, then we will create a new wandb run.
        

        self.t_min = t_min
        self.t_max = t_max
        self.pwp = PiecewiseLinearPolynomials(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, times=k_h_rf_times)
        self.lb = jnp.array([t_min, x_min])
        self.ub = jnp.array([t_max, x_max])
        self.h = (x_max - x_min)/N_x
        self.model = MG_NN(input_dim=2, hidden_dims_list=hidden_dims_list, activation=nn.tanh, lb=self.lb, ub=self.ub, seed=seed, use_a=use_a)
        self.total_grade = len(hidden_dims_list)  # Number of grades for MGDL
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
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_eqn = lambda_eqn
        self.gamma = gamma
        self.N_test_gamma = N_test_gamma
        self.adjust_gamma = adjust_gamma
        self.f_adaptive_weight = f_adaptive_weight
        self.f_opt_kh = f_opt_kh
        self.alpha = alpha
        self.res_visc = res_visc
        self.res = res
        self.lambda_tv = lambda_tv
        self.N_t_pnts = 10*N_t if N_t_pnts is None else N_t_pnts
        self.N_x_pnts = 10*N_x if N_x_pnts is None else N_x_pnts
        self.t_points = jnp.linspace(self.pwp.t_min, self.pwp.t_max, self.N_t_pnts)
        self.x_points = jnp.linspace(self.pwp.x_min, self.pwp.x_max, self.N_x_pnts)
        self.pre_epochs = pre_epochs
        self.gen_ibc()
        if use_loss_fn_elem_int:
            self.loss_fn_eqn = self.loss_fn_elem_int
        else:
            self.loss_fn_eqn = self.loss_fn_elem_bdry
        if self.gamma > 0.0:
            self.loss_fn = self.loss_fn_gamma
        else:
            self.loss_fn = self.loss_fn_lambda
        
        self.L1reg = L1reg
        self.sampling_method = sampling_method
        self.project = project
        self.notes = notes
        self.run_name = f"tmin{t_min:.1f}_tmax{t_max:.1f}_Nt{N_t}_Nx{N_x}_lrs{init_lrs[0]}_eps{grade_epochs[0]}_L1reg{L1reg}_sig{sigma}_gam{gamma}_Np{N_perturb}_sd{seed}"
        # Initialize optimizers for each grade of the first stage
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.optimizer_list = []
        self.params_list = []
        self.lr_schedulers = []
        for i in range(self.total_grade):
            self.model.add_grade()  # Add a new grade
            self.params_list.append(self.model.mlps[i].params)
            # lr = optax.schedules.constant_schedule(init_lrs[i])
            # lr = optax.schedules.exponential_decay(init_value=init_lrs[i], transition_steps=grade_epochs[i], decay_rate=decay_rates[i])
            # lr = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6, peak_value=init_lrs[i], warmup_steps=100, transition_steps=grade_epochs[i], decay_rate=decay_rates[i], end_value=1e-6)
            lr = optax.schedules.warmup_cosine_decay_schedule(init_value=1e-6, peak_value=init_lrs[i], warmup_steps=100, decay_steps=grade_epochs[i], end_value=1e-6)
            # lr = optax.schedules.cosine_decay_schedule(init_value=init_lrs[i], decay_steps=grade_epochs[i])
            self.lr_schedulers.append(lr)
            self.optimizer_list.append(optax.adam(self.lr_schedulers[i]))
        self.opt_state_list = [opt.init(self.model.mlps[i].params) for i, opt in enumerate(self.optimizer_list)]
        # Optimizer for the second stage
        self.init_lr_stage2 = init_lr_stage2
        self.epoch_stage2 = epoch_stage2
        self.decay_rate_stage2 = decay_rate_stage2
        lr_stage2 = optax.exponential_decay(init_value=init_lr_stage2, transition_steps=epoch_stage2, decay_rate=decay_rate_stage2, end_value=1e-6)
        self.optimizer_stage2 = optax.adam(lr_stage2)
        self.opt_state_stage2 = None  # To be initialized later

        if not self.seq_train and self.set_wandb:
            wandb.init(project=self.project, name=self.run_name, notes=self.notes,
                       config={
                           "total_grade": self.total_grade,
                           "k": self.k,
                           "N_t": N_t,
                           "N_x": N_x,
                           "init_lrs": init_lrs,
                           "grade_epochs": grade_epochs,
                           "decay_rates": decay_rates,
                           "init_lr_stage2": init_lr_stage2,
                           "epoch_stage2": epoch_stage2,
                           "decay_rate_stage2": decay_rate_stage2,
                           "gamma": gamma,
                           "f_opt_kh" : f_opt_kh,
                           "f_adaptive_weight": f_adaptive_weight,
                           "alpha": alpha,
                           "N_perturb": N_perturb,
                           "sigma": sigma,
                           "L1reg": L1reg
                       })
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

    def maximize_kh_vec(self, params, current_key):
        Nt = self.pwp.N_elements
        vertices_all = self.pwp.triangles
        u_values_all = self.u_eval(params, vertices_all.reshape(-1, 2)).reshape(Nt, 3)
        A_all = jnp.stack([jnp.ones((Nt, 3)), vertices_all[:, :, 0], vertices_all[:, :, 1]], axis=-1)
        init_coeffs_all = jnp.linalg.solve(A_all, u_values_all[..., None]).squeeze(-1)
        key, current_key = jax.random.split(current_key)
        if self.sampling_method == 'normal':
            perturbations_all = self.sigma * jax.random.normal(key, (self.N_perturb, 3))
        elif self.sampling_method == 'uniform':
            perturbations_all = self.sigma * jax.random.uniform(key, (self.N_perturb, 3), minval=-1, maxval=1)
        
        perturbed_coeffs_all = init_coeffs_all[:, None, :] + perturbations_all[None, :, :]
        losses_all, loss_abs = vmap(self.loss_fn_elem_int_per_elem, in_axes=(0, 0, None))(jnp.arange(Nt), perturbed_coeffs_all, params)
        best_indices = jnp.argmax(losses_all, axis=1)
        # print(f"Nt: {Nt}, losses_all shape: {losses_all[:, best_indices].shape}")
        # print(f"Nt: {Nt}, loss_abs shape: {loss_abs.shape}")
        loss_eqn = jnp.sum(losses_all[jnp.arange(Nt), best_indices])
        # Compute absolute loss, which is the L1 norm of the residual. Just for comparison and debugging. We do not use it in the optimization.
        loss_eqn_abs = jnp.sum(loss_abs)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        # print(f"best_coeffs_all shape: {best_coeffs_all.shape}, perturbed_coeffs_all shape: {perturbed_coeffs_all.shape}")
        return best_coeffs_all, current_key, loss_eqn, loss_eqn_abs

    def loss_fn_elem_int_per_elem(self, tri_idx, p_coeffs_all, params):
        quad_points = self.pwp.quad_points_tri[tri_idx]
        quad_weights = self.pwp.quad_weights_tri[tri_idx]
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
        # Compute absolute loss, which is the L1 norm of the residual. Just for comparison and debugging. We do not use it in the optimization.
        loss_abs = jnp.abs(residual)@quad_weights
        return loss, loss_abs
    
    @jit
    def compute_tv_loss(self, params):
        """Compute the Total Variation Diminishing (TVD) loss."""
        # Define time points for TV calculation
        t_points = self.t_points
        x_points = self.x_points
        # Create grid for evaluation at consecutive time steps
        tx_n = jnp.stack(jnp.meshgrid(t_points[:-1], x_points), axis=-1).reshape(-1, 2)
        tx_n_plus_1 = jnp.stack(jnp.meshgrid(t_points[1:], x_points), axis=-1).reshape(-1, 2)
        # Evaluate solution at t_n and t_{n+1}
        u_n = self.u_eval(params, tx_n).reshape(self.N_t_pnts - 1, self.N_x_pnts)
        u_n_plus_1 = self.u_eval(params, tx_n_plus_1).reshape(self.N_t_pnts - 1, self.N_x_pnts)
        # Compute TV_h for each time step
        tv_n = jnp.sum(jnp.abs(u_n[:, 1:] - u_n[:, :-1]), axis=1)
        tv_n_plus_1 = jnp.sum(jnp.abs(u_n_plus_1[:, 1:] - u_n_plus_1[:, :-1]), axis=1)
        # Compute TVD loss: sum of squared differences where TV increases
        tv_diff = tv_n_plus_1 - tv_n
        tvd_loss = jnp.sum(jnp.maximum(0, tv_diff)**2)
        return tvd_loss
    
    # Modified function to compute TV loss with a more efficient approach
    def modified_compute_tv_loss(self, params):
        t_points = self.t_points
        x_points = self.x_points

        def compute_tv(t, params):
            tx = jnp.stack([t * jnp.ones(self.N_x_pnts), x_points], axis=-1)
            u = self.u_eval(params, tx)
            tv = jnp.sum(jnp.abs(u[1:] - u[:-1]))
            return tv

        def body(carry, n):
            tv_prev, loss = carry
            tv_curr = compute_tv(t_points[n+1], params)
            diff = tv_curr - tv_prev
            loss += jnp.maximum(0, diff)**2
            return (tv_curr, loss), None

        initial_tv = compute_tv(t_points[0], params)
        initial_carry = (initial_tv, 0.0)
        carry, _ = jax.lax.scan(body, initial_carry, jnp.arange(self.N_t_pnts - 1))
        tvd_loss = carry[1]
        return tvd_loss
    

    @jit
    def loss_fn_elem_int(self, params):
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]            
        residual = self.res(u_quad, u_t, u_x)
        p_coeffs_quad = self.pwp.coeffs[self.pwp.element_indices]
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
        integrand = residual * jnp.sign(u_quad - k_h_quad)

        if self.L1reg > 0:
            integrand += self.L1reg * jnp.abs(self.h * residual)
            
        interior_loss = jnp.dot(quad_weights, integrand)

        # Add TV penalty term
        tv_loss = self.compute_tv_loss(params)
        total_loss = interior_loss + self.lambda_tv * self.h * tv_loss
        return total_loss
    
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
        
    def loss_fn_ic(self, params):
        return jnp.mean((self.u_eval(params, self.tx_t0) - self.u_t0)**2)

    def loss_fn_bc(self, params):
        return jnp.mean((self.u_eval(params, self.tx_x0) - self.u_x0)**2) + jnp.mean((self.u_eval(params, self.tx_x1) - self.u_x1)**2)

    def loss_fn_ibc(self, params):
        return self.loss_fn_ic(params) + self.loss_fn_bc(params)

    def loss_fn_lambda(self, params):
        loss_ic = self.loss_fn_ic(params)
        loss_bc = self.loss_fn_bc(params)
        loss_eqn = self.loss_fn_eqn(params)
        total_loss = self.lambda_eqn * loss_eqn + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc
        return total_loss, (loss_eqn, loss_ic + loss_bc)
    
    def loss_fn_gamma(self, params):
        loss_ibc = self.loss_fn_ibc(params)
        loss_eqn = self.loss_fn_eqn(params)
        total_loss = self.gamma * loss_eqn + (1 - self.gamma) * loss_ibc
        return total_loss, (loss_eqn, loss_ibc)
    
    def train_step(self, params, optimizer, opt_state):
        (_, all_losses), grads = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return all_losses[0], all_losses[1], params, opt_state
    
    def pre_train_to_ic(self, pre_epochs=1000, pre_lr=1e-3, N_pre=10000):
        print("\nPre-training the network to fit v(t, x) = u0(x)")
        # Generate pre-training points
        self.key, subkey = jax.random.split(self.key)
        tx_pre = jax.random.uniform(subkey, (N_pre, 2), minval=self.lb, maxval=self.ub)
        y_pre = self.u0(tx_pre[:, 1])

        # Collect all parameters
        params = self.model.mlps[0].params
        optimizer = optax.adam(pre_lr)
        opt_state = optimizer.init(params)

        def pre_loss(params):
            u_pred = vmap(lambda tx: self.model.forward_with_params(params, tx))(tx_pre).squeeze()
            return jnp.mean((u_pred - y_pre)**2)

        pre_loss_grad = jax.value_and_grad(pre_loss)

        for iter in range(pre_epochs):
            loss, grads = pre_loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if iter % 100 == 0:
                print(f"Pre-training Iter {iter}, Loss: {loss:.6f}")

        # Assign back the pre-trained parameters
        self.model.mlps[0].params = params

    def train_process_stage1(self, save_model=True):
        self.pre_train_to_ic(self.pre_epochs)
        
        print("\nStarting Train Stage with MGDL")
        start_time = time.perf_counter()
        self.global_step = 0
        # Stage 1: Sequential Grade Training
        print("\nStage 1: Sequential Grade Training")
        self.loss_eqn_history = [[] for _ in range(self.total_grade)]
        self.loss_bc_history = [[] for _ in range(self.total_grade)]
        self.best_params_list = [None] * self.total_grade
        self.best_loss_eqn_list = [float('inf')] * self.total_grade

        for grade_idx in range(self.total_grade):
            print(f"\nTraining Grade {grade_idx}")
            self.gen_funcs(grade_idx)
            self.maximize_kh = jit(self.maximize_kh_vec)
            params = self.model.mlps[grade_idx].params
            optimizer = self.optimizer_list[grade_idx]
            opt_state = self.opt_state_list[grade_idx]
            self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(params, self.key)
            if self.gamma > 0.0 and self.adjust_gamma:
                self.gamma = self.determine_equation_weight(grade_idx, gamma_init=self.gamma)
                print(f"Stage 1, Gamma: {self.gamma:.6f}")
            monitor_start = int(0.9 * self.grade_epochs[grade_idx])
            for iter in range(self.grade_epochs[grade_idx]):
                if (iter % self.f_adaptive_weight == 0) and (self.gamma == 0.0):
                    self.update_adaptive_weights(params)
                    print(f"Grade {grade_idx}, Iter {iter}, lambda_ic: {self.lambda_ic:.6f}, lambda_bc: {self.lambda_bc:.6f}, lambda_eqn: {self.lambda_eqn:.6f}")
                loss_eqn, loss_ibc, params, opt_state = self.train_step(params, optimizer, opt_state)
                if iter % self.f_opt_kh == 0:
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(params, self.key)
                loss, (loss_eqn, loss_ibc) = self.loss_fn(params)

                self.loss_eqn_history[grade_idx].append(loss_eqn)
                self.loss_bc_history[grade_idx].append(loss_ibc)
                if iter >= monitor_start:
                    if loss_eqn < self.best_loss_eqn_list[grade_idx]:
                        self.best_loss_eqn_list[grade_idx] = loss_eqn
                        self.best_params_list[grade_idx] = jax.tree_util.tree_map(jnp.copy, params)

                if iter % 100 == 0:
                    print(f"Grade {grade_idx}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_eqn_from_kh: {loss_eqn_from_kh:.6f}, loss_eqn_abs: {loss_eqn_abs:.6f}, loss_ibc: {loss_ibc:.6f}")
                    if self.set_wandb:
                        wandb.log({f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_eqn_grade{grade_idx}": loss_eqn, 
                                   f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_ibc_grade{grade_idx}": loss_ibc, 
                                   f"step": self.global_step + iter})
                self.model.mlps[grade_idx].params = params
                self.opt_state_list[grade_idx] = opt_state

            # Assign the best params found in the last 10%
            if self.best_params_list[grade_idx] is not None:
                self.model.mlps[grade_idx].params = self.best_params_list[grade_idx]
            self.opt_state_list[grade_idx] = opt_state
            self.global_step += self.grade_epochs[grade_idx]
        end_time = time.perf_counter()
        train_total_time = end_time - start_time
        print(f"Train time for stage1: {train_total_time:.6f} seconds")
        print(f"Train average time per iteration: {train_total_time / sum(self.grade_epochs):.9e} seconds")

    def train_process_stage2(self, save_model=True):
        # Stage 2: Refinement of Last k Grades
        if self.k > 0 and self.k <= self.total_grade:
            start_time = time.perf_counter()
            if self.gamma > 0.0 and self.adjust_gamma:
                self.gamma = self.determine_equation_weight(gamma_init=self.gamma)
            print(f"\nStage 2: Refining Last {self.k} Grades")
            self.gen_funcs()
            self.maximize_kh = jit(self.maximize_kh_vec)
            loss_eqn_fn = jit(self.loss_fn_eqn)
            loss_data_fn = jit(self.loss_fn_ibc)
            params_last_k = [mlp.params for mlp in self.model.mlps[-self.k:]]
            self.opt_state_stage2 = self.optimizer_stage2.init(params_last_k)
            optimizer = self.optimizer_stage2
            opt_state = self.opt_state_stage2
            self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(params_last_k, self.key)
            monitor_start = int(0.9 * self.epoch_stage2)
            best_loss_eqn = float('inf')
            best_params_last_k = None
            for iter in range(self.epoch_stage2):
                if iter % self.f_adaptive_weight == 0:
                    self.update_adaptive_weights(params)
                    print(f"Stage 2, Iter {iter}, lambda_ic: {self.lambda_ic:.6f}, lambda_bc: {self.lambda_bc:.6f}, lambda_eqn: {self.lambda_eqn:.6f}")
                loss_eqn, loss_ibc, params_last_k, opt_state = self.train_step(params_last_k, optimizer, opt_state)
                if iter % self.f_opt_kh == 0:
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(params_last_k, self.key)
                loss_eqn = loss_eqn_fn(params_last_k)
                loss_ibc = loss_data_fn(params_last_k)
                if iter >= monitor_start:
                    if loss_eqn < best_loss_eqn:
                        best_loss_eqn = loss_eqn
                        best_params_last_k = jax.tree_util.tree_map(jnp.copy, params_last_k)
                if iter % 100 == 0:
                    print(f"Stage 2, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_ibc: {loss_ibc:.6f}")
                    if self.set_wandb:
                        wandb.log({f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_eqn_second_stage": loss_eqn, 
                                   f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_ibc_second_stage": loss_ibc, 
                                   f"step": self.global_step + iter})
            # Assign the best params found in the last 10%
            if best_params_last_k is not None:
                params_last_k = best_params_last_k
            for mlp, p in zip(self.model.mlps[-self.k:], params_last_k):
                mlp.params = p
            self.global_step += self.epoch_stage2

            end_time = time.perf_counter()
            train_total_time = end_time - start_time
            print(f"Train time for stage2: {train_total_time:.6f} seconds")
            print(f"Train average time per iteration: {train_total_time / self.epoch_stage2:.9e} seconds")
        
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('data', f'trained_model_{self.run_name}_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Train model saved to {model_path}")
        if not self.seq_train and self.set_wandb:
            wandb.finish()
        
        
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
        self.lambda_eqn = self.alpha * self.lambda_eqn + (1 - self.alpha) * hat_lambda_r
    
    def determine_equation_weight(self, grade_idx=None, N1=None, eta=0.1, gamma_init=0.1, criterion_low=0.1, criterion_high=10.0, max_trials=20, tol=1e-5):
        if N1 is None:
            N1 = int(0.1 * self.N_test_gamma)
        gamma = gamma_init
        if grade_idx is not None:
            optimizer = self.optimizer_list[grade_idx]
            opt_state = self.opt_state_list[grade_idx]
            initial_params = jax.tree_util.tree_map(jnp.copy, self.model.mlps[grade_idx].params)
            initial_opt_state = jax.tree_util.tree_map(jnp.copy, opt_state)
        else:
            optimizer = self.optimizer_stage2
            opt_state = self.opt_state_stage2
            initial_params = jax.tree_util.tree_map(jnp.copy, self.params_stage2)
            initial_opt_state = jax.tree_util.tree_map(jnp.copy, opt_state)
        
        loss_data_fn = self.loss_fn_ibc
        loss_eqn_fn = self.loss_fn_eqn
        
        for trial in range(max_trials):
            print(f"The gamma of {trial + 1}th trials is {gamma}.")
            # Reset params and opt_state for each pre-training
            current_params = jax.tree_util.tree_map(jnp.copy, initial_params)
            current_opt_state = jax.tree_util.tree_map(jnp.copy, initial_opt_state)
        
            # Define total loss for this gamma
            def total_loss(params):
                loss_data = loss_data_fn(params)
                loss_eqn = loss_eqn_fn(params)
                return (1 - gamma) * loss_data + gamma * loss_eqn, (loss_eqn, loss_data)
            
            # Collect loss history
            loss_history_data = []
            loss_history_eqn = []
            self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(current_params, self.key)
            for iter in range(self.N_test_gamma):
                (loss, (loss_eqn, loss_data)), grads = jax.value_and_grad(total_loss, has_aux=True)(current_params)
                updates, current_opt_state = optimizer.update(grads, current_opt_state)
                current_params = optax.apply_updates(current_params, updates)
                if iter % self.f_opt_kh == 0:
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh(current_params, self.key)
                # Store losses
                loss_eqn = loss_eqn_fn(current_params)
                loss_data = loss_data_fn(current_params)
                loss_history_data.append(loss_data)
                loss_history_eqn.append(loss_eqn)
                if iter % 100 == 0:
                    print(f"Gamma {gamma}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_eqn_from_kh: {loss_eqn_from_kh:.6f}, loss_ibc: {loss_data:.6f}")

            # Initial losses (at i=0)
            L0_data = max(loss_history_data[:N1])
            L0_eqn = max(loss_history_eqn[:N1])
            
            # Minimal losses in last N1 iterations
            Lm_data = min(loss_history_data[N1:])
            Lm_eqn = min(loss_history_eqn[N1:])
            
            # Convergence rates
            conv_data = L0_data / Lm_data if Lm_data > 0 else float('inf')
            conv_eqn = L0_eqn / Lm_eqn if Lm_eqn > 0 else float('inf')
            
            # Ratio Rc
            Rc = conv_data / conv_eqn if conv_eqn != 0 else float('inf')
            print(f"Trial {trial + 1}: L0_data={L0_data:.4f}, Lm_data={Lm_data:.4f}, conv_data={conv_data:.4f}, L0_eqn={L0_eqn:.4f}, Lm_eqn={Lm_eqn:.4f}, conv_eqn={conv_eqn:.4f}, Rc={Rc:.4f}")
            
            # Check criterion
            if criterion_low < Rc < criterion_high:
                print(f"Optimal gamma found: {gamma} after {trial + 1} trials.")
                return gamma
            
            # Update gamma
            prev_gamma = gamma
            gamma = gamma * (Rc ** eta)
            print(f"Updating gamma to {gamma:.4f}")
            if gamma < 0:
                print("Gamma became negative, resetting to 1e-3")
                gamma = 1e-3
                return gamma
            elif gamma > 1:
                print("Gamma exceeded 1, resetting to 0.999")
                gamma = 0.999
                return gamma
            
            # Early stop if gamma stabilizes
            if abs(gamma - prev_gamma) < tol:
                print(f"gamma stabilized within tolerance {tol}.")
                break
        print(f"Max trials reached. Returning last gamma: {gamma}")
        return gamma

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
        loaded_total_grade = loaded_data['total_grade']
        lb = loaded_data['lb']
        ub = loaded_data['ub']
        self.model = MG_NN(input_dim=2, hidden_dims_list=loaded_hidden_dims_list, activation=nn.tanh, lb=lb, ub=ub)
        for i in range(self.total_grade):
            self.model.add_grade()  # Add a new grade
        for mlp, params in zip(self.model.mlps, loaded_params):
            mlp.params = params
        print(f"Model loaded from {model_path} with hidden_dims_list: {loaded_hidden_dims_list}, loaded_total_grade = {loaded_data['total_grade']}.")
        self.opt_state_list = [opt.init(mlp.params) for opt, mlp in zip(self.optimizer_list, self.model.mlps)]
        self.gen_funcs(loaded_total_grade-1)
        self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_eqn_abs = self.maximize_kh_vec(loaded_params[-1], self.key)
        loss_eqn = self.loss_fn_eqn(loaded_params[-1])
        loss_ibc = self.loss_fn_ibc(loaded_params[-1])
        print(f"For the loaded model, loss_eqn: {loss_eqn:.6f}, Loss_bc: {loss_ibc:.6f}")