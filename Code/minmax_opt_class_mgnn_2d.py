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
from MGNN_class import MLP, MG_NN
from my_helpers import plot_u_theta, plot_kh, plot_u_theta_mesh
from functools import partial
from PwpTri_class import PiecewiseLinearPolynomialsTri
from PwpRect_class import PiecewiseLinearPolynomialsRect
from PwpHex_class import PiecewiseLinearPolynomialsHex
from PwpTetra_class import PiecewiseLinearPolynomialsTetra
import wandb
from functools import partial

# Utility function to compute L2 norm of a pytree
def pytree_l2_norm(pytree):
    leaves = tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))

# Optimizer Class
class MinMaxOptProb2D:
    def __init__(self, input_dim=2, t_min=0, t_max=1, x_min=0, x_max=1, y_min=None, y_max=None, N_t=10, N_x=10, N_y=None, N_bc=128, N_ic=256, hidden_dims_list=[(64, 64, 64, 64)], seed=0, 
                 sigma=5, N_perturb=10000, k_h_rf_times=2, f_adaptive_weight=1000, f_opt_kh=1, u0=None, u1=None, u2=None, u3=None, u4=None, project="minmax_mgnn_burgers",
                 lambda_ic=20.0, lambda_bc=20.0, lambda_eqn=1.0, alpha=0.9, gamma=0.0, N_test_gamma=1000, adjust_gamma=True,
                 L1reg=0, sampling_method='uniform', notes='', nu=1e-3,
                 init_lrs=[1e-3, 1e-3, 1e-3], grade_epochs=[10, 10, 10], decay_rates=[1e-3, 1e-3, 1e-3], init_lr_stage2=1e-3, epoch_stage2=100, decay_rate_stage2=1e-3,
                 res=None, bc_type='Dirichlet', loss_type='interior', k=0, use_a=True, M_t = 1000, M_x = 1000, f_plot=1000, plt_err_dist=False, plt_mesh=False,
                 partition='tri', p_dof=None, dg_type='P1', use_importance_sampling=False, u_ref=None, pre_epochs=1000, pre_train='',
                 seq_train=False, set_wandb=False):
        self.input_dim = input_dim
        
        self.seq_train = seq_train  # Sequential training flag
        self.set_wandb = set_wandb
        # if self.seq_train is True, then we will not create a new wandb run but just use the existing one.
        # if self.seq_train is False and self.set_wandb is True, then we will create a new wandb run.

        self.partition = partition  # 'tri' or 'rect' or 'hex'
        self.dg_type = dg_type  # 'P1' or 'Q1' only used when partition is 'rect' or 'hex'
        self.p_dof = p_dof  # number of dof per element, if None, will be set to input_dim + 1 for 'P1' and 2**input_dim for 'Q1'
        if self.p_dof is None:
            if self.dg_type == 'P1':
                self.p_dof = self.input_dim + 1
            elif self.dg_type == 'Q1':
                self.p_dof = 2 ** self.input_dim
        
        self.ht = (t_max - t_min) / N_t
        self.hx = (x_max - x_min) / N_x
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.M_t = M_t
        self.M_x = M_x
        self.N_t = N_t
        self.N_x = N_x

        if self.input_dim == 2:
            self.lb = jnp.array([t_min, x_min])
            self.ub = jnp.array([t_max, x_max])
        elif self.input_dim == 3:
            if y_min is None or y_max is None or N_y is None:
                raise ValueError("For input_dim=3, y_min, y_max, and N_y must be provided.")
            self.lb = jnp.array([t_min, x_min, y_min])
            self.ub = jnp.array([t_max, x_max, y_max])
            self.N_y = N_y
            self.y_min = y_min
            self.y_max = y_max
            self.hy = (y_max - y_min) / N_y
        else:
            raise ValueError("input_dim must be 2 or 3.")

        if self.partition == 'tri' and self.input_dim == 2:
            self.pwp = PiecewiseLinearPolynomialsTri(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, rf_times=k_h_rf_times)
        elif self.partition == 'rect' and self.input_dim == 2:
            self.pwp = PiecewiseLinearPolynomialsRect(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, rf_times=k_h_rf_times, 
                                                      dg_type=self.dg_type)
        elif self.partition == 'hex' and self.input_dim == 3:
            self.pwp = PiecewiseLinearPolynomialsHex(N_t, N_x, self.N_y, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, y_min=self.y_min, y_max=self.y_max, rf_times=k_h_rf_times, 
                                                     dg_type=self.dg_type)
        elif self.partition == 'tetra' and self.input_dim == 3:
            self.pwp = PiecewiseLinearPolynomialsTetra(N_t, N_x, self.N_y, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, y_min=self.y_min, y_max=self.y_max, quad_order=k_h_rf_times, 
                                                       dg_type=self.dg_type)
        else:
            raise ValueError("Partition must be 'tri' or 'rect' for 2D, 'hex' for 3D.")

        
        self.use_a = use_a  # Whether to use the adaptive parameter a in MLP
        self.model = MG_NN(input_dim=self.input_dim, hidden_dims_list=hidden_dims_list, activation=nn.tanh, lb=self.lb, ub=self.ub, seed=seed, use_a=self.use_a)
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
        if self.input_dim == 3:
            self.y_bc = jnp.linspace(self.y_min, self.y_max, int(round(self.N_ic * (self.y_max - self.y_min) / (self.x_max - self.x_min))))
            self.u3 = u3
            self.u4 = u4
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_eqn = lambda_eqn
        self.gamma = gamma
        self.N_test_gamma = N_test_gamma
        self.adjust_gamma = adjust_gamma
        self.f_adaptive_weight = f_adaptive_weight
        self.f_opt_kh = f_opt_kh
        self.alpha = alpha
        self.res = res
        self.loss_type = loss_type

        
        self.nu = nu
        self.gen_ibc()
        self.bc_type = bc_type  # 'Dirichlet' or 'inflow'
        if self.bc_type == 'Dirichlet':
            self.loss_fn_bc = self.loss_fn_bc_Dirichlet
        elif self.bc_type == 'inflow':
            self.loss_fn_bc = self.loss_fn_bc_inflow
        else:
            raise ValueError("bc_type must be 'Dirichlet' or 'inflow'.")

        if self.loss_type == 'interior':
            self.loss_fn_eqn = self.loss_fn_elem_int
        elif self.loss_type == 'boundary':
            self.loss_fn_eqn = self.loss_fn_elem_bdry
        elif self.loss_type == 'pinn':
            self.loss_fn_eqn = self.loss_fn_pinn

        if self.gamma > 0.0:
            self.loss_fn = self.loss_fn_gamma
        else:
            self.loss_fn = self.loss_fn_lambda
        
        self.L1reg = L1reg
        # For 'normal' sampling method, we use a normal distribution to generate perturbations.
        # For 'uniform' sampling method, we use a uniform distribution to generate perturbations, which has bounded W^{1,inf} norm.
        self.sampling_method = sampling_method
        # print(f"Using sampling method: {self.sampling_method}")
        self.project = project
        self.notes = notes
        self.run_name = f"tmin{t_min:.1f}_tmax{t_max:.1f}_Nt{N_t}_Nx{N_x}_lrs{init_lrs[0]}_eps{grade_epochs[0]}_L1reg{L1reg}_sig{sigma}_gam{gamma}_Np{N_perturb}_sd{seed}"
        # Initialize optimizers for each grade of the first stage
        self.init_lrs = init_lrs
        self.grade_epochs = grade_epochs
        self.decay_rates = decay_rates
        self.params_list = []
        self.lr_schedulers = []
        self.optimizer_list = []
        self.opt_state_list = []
        for i in range(self.total_grade):
            self.model.add_grade()  # Add a new grade
            self.params_list.append(self.model.mlps[i].params)
            # lr = optax.schedules.constant_schedule(init_lrs[i])
            # lr = optax.schedules.exponential_decay(init_value=init_lrs[i], transition_steps=grade_epochs[i], decay_rate=decay_rates[i])
            # lr = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6, peak_value=init_lrs[i], warmup_steps=100, transition_steps=grade_epochs[i], decay_rate=decay_rates[i], end_value=1e-6)
            # lr = optax.schedules.warmup_cosine_decay_schedule(init_value=1e-6, peak_value=init_lrs[i], warmup_steps=int(grade_epochs[i]/100), decay_steps=grade_epochs[i], end_value=1e-5)
            # lr = optax.schedules.cosine_decay_schedule(init_value=init_lrs[i], decay_steps=grade_epochs[i])
            # self.lr_schedulers.append(lr)
            # self.optimizer_list.append(optax.adam(self.lr_schedulers[i]))
        # self.opt_state_list = [opt.init(self.model.mlps[i].params) for i, opt in enumerate(self.optimizer_list)]
        # Optimizer for the second stage
        self.init_lr_stage2 = init_lr_stage2
        self.epoch_stage2 = epoch_stage2
        self.decay_rate_stage2 = decay_rate_stage2
        lr_stage2 = optax.exponential_decay(init_value=init_lr_stage2, transition_steps=epoch_stage2, decay_rate=decay_rate_stage2, end_value=1e-6)
        self.optimizer_stage2 = optax.adam(lr_stage2)
        self.opt_state_stage2 = None  # To be initialized later

        self.f_plot = f_plot
        self.plt_err_dist = plt_err_dist
        self.plt_mesh = plt_mesh
        
        if self.partition == 'tri':
            self.elem_meas = 0.5 * (self.ht * self.hx)
        elif self.partition == 'rect':
            self.elem_meas = self.ht * self.hx
        elif self.partition == 'hex':
            self.elem_meas = self.ht * self.hx * self.hy
        elif self.partition == 'tetra':
            self.elem_meas = self.ht * self.hx * self.hy / 6.0
        else:
            raise ValueError("Partition must be in ['tri', 'rect', 'hex', 'tetra'].")
        if self.sampling_method == 'normal':
            self.partition = 'tri'  # For normal sampling, we use triangular partitioning
            print(f"Using triangular partitioning for normal sampling method.")
        self.use_importance_sampling = use_importance_sampling
        self.importance_probs = None
        self.u_ref = u_ref  # Reference solution for error calculation, if provided
        self.pre_epochs = pre_epochs
        self.pre_train = pre_train  # 'ic' or 'u_ref' or ''

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
        self.gen_funcs(grade_idx=0)  # Generate functions for the first grade
        self.vmap_loss_fn_elem_int_per_elem = vmap(self.loss_fn_elem_int_per_elem, in_axes=(0, 0, None))
        print(f"Partition type: {self.partition}, dg_type: {self.dg_type}, p_dof: {self.p_dof}")

    def update_importance_probs(self, loss_all_elems):
        print(f"Updating importance sampling probabilities...")
        weights = jnp.maximum(jnp.abs(loss_all_elems), 1e-6)  # Avoid zero
        self.importance_probs = weights / jnp.sum(weights)

    def gen_ibc(self):
        if self.input_dim == 2:
            self.u_t0 = self.u0(self.x_bc)
            self.tx_t0 = jnp.stack([self.t_min * jnp.ones_like(self.x_bc), self.x_bc], axis=1)
            self.u_x0 = self.u1(self.t_bc)
            self.tx_x0 = jnp.stack([self.t_bc, self.x_min * jnp.ones_like(self.t_bc)], axis=1)
            self.u_x1 = self.u2(self.t_bc)
            self.tx_x1 = jnp.stack([self.t_bc, self.x_max * jnp.ones_like(self.t_bc)], axis=1)
        elif self.input_dim == 3:
            # Initial condition at t = t_min
            x_grid, y_grid = jnp.meshgrid(self.x_bc, self.y_bc)
            self.tx_t0 = jnp.stack([self.t_min * jnp.ones_like(x_grid.ravel()), x_grid.ravel(), y_grid.ravel()], axis=1)
            self.u_t0 = self.u0(self.tx_t0[:,1], self.tx_t0[:,2])
            # Boundary at x = x_min
            t_grid, y_grid = jnp.meshgrid(self.t_bc, self.y_bc)
            self.tx_x0 = jnp.stack([t_grid.ravel(), self.x_min * jnp.ones_like(t_grid.ravel()), y_grid.ravel()], axis=1)
            self.u_x0 = self.u1(self.tx_x0[:,0], self.tx_x0[:,2])
            # Boundary at x = x_max
            self.tx_x1 = jnp.stack([t_grid.ravel(), self.x_max * jnp.ones_like(t_grid.ravel()), y_grid.ravel()], axis=1)
            self.u_x1 = self.u2(self.tx_x1[:,0], self.tx_x1[:,2])
            # Boundary at y = y_min
            t_grid, x_grid = jnp.meshgrid(self.t_bc, self.x_bc)
            self.tx_y0 = jnp.stack([t_grid.ravel(), x_grid.ravel(), self.y_min * jnp.ones_like(t_grid.ravel())], axis=1)
            self.u_y0 = self.u3(self.tx_y0[:,0], self.tx_y0[:,1])
            # Boundary at y = y_max
            self.tx_y1 = jnp.stack([t_grid.ravel(), x_grid.ravel(), self.y_max * jnp.ones_like(t_grid.ravel())], axis=1)
            self.u_y1 = self.u4(self.tx_y1[:,0], self.tx_y1[:,1])
        
    def gen_funcs(self, grade_idx=None):
        if grade_idx is None:
            net_u = lambda params_last_k, X: self.model.forward_with_params_last_k(params_last_k, X, self.k).squeeze()
        else:
            net_u = lambda params, X: self.model.forward_with_params(params, X, grade_idx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes = (None, 0)))
        self.grad_u = jit(vmap(grad(net_u, argnums=1), in_axes = (None, 0)))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=1), in_axes = (None, 0)))
        def laplacian_fn(params, X):
            hess = self.hessian_u(params, X)
            if self.input_dim == 2:
                return hess[:, 1]
            elif self.input_dim == 3:
                return jnp.trace(hess[:, 1:, 1:], axis1=1, axis2=2)
        self.laplacian = jit(laplacian_fn)

    def maximize_kh_step(self, params, current_key):
        Nt = self.pwp.N_elements
        # find proper kh by generating samples for each triangle/rectangle
        key, current_key = jax.random.split(current_key)
        if self.sampling_method == 'normal':
            vertices_all = self.pwp.partitions
            num_vertices = vertices_all.shape[1]
            u_values_all = self.u_eval(params, vertices_all.reshape(-1, self.input_dim)).reshape(Nt, num_vertices)
            A_all = jnp.concatenate([jnp.ones((Nt, num_vertices, 1)), vertices_all.reshape(Nt, num_vertices, self.input_dim)], axis=-1)  # (Nt, num_vertices, p_dof)
            init_coeffs_all = vmap(jnp.linalg.lstsq, in_axes=(0, 0))(A_all, u_values_all[..., None])[0].squeeze(-1)  # Least squares for overdetermined
            perturbations_all = self.sigma * jax.random.normal(key, (self.N_perturb, self.p_dof))
        elif self.sampling_method == 'uniform':
            quad_points = self.pwp.quad_points_parti
            u_values_all = self.u_eval(params, quad_points)
            quad_weights = self.pwp.quad_weights_parti
            avg_vals = jnp.einsum('ij,ij->i', u_values_all, quad_weights)/self.elem_meas
            if self.dg_type == 'P1':
                init_coeffs_all = jnp.hstack([avg_vals[..., None] , jnp.zeros((Nt, self.p_dof - 1))])
            elif self.dg_type == 'Q1':
                init_coeffs_all = jnp.tile(avg_vals[..., None], (1, self.p_dof))
            # print(f"init_coeffs_all.shape: {init_coeffs_all.shape}, avg_vals.shape: {avg_vals.shape}")
            perturbations_all = self.sigma * jax.random.uniform(key, (self.N_perturb, self.p_dof), minval=-1, maxval=1)
        
        perturbed_coeffs_all = init_coeffs_all[:, None, :] + perturbations_all[None, :, :]
        loss_all_test, loss_l1_all = self.vmap_loss_fn_elem_int_per_elem(jnp.arange(Nt), perturbed_coeffs_all, params)
        best_indices = jnp.argmax(loss_all_test, axis=1)
        loss_each_elem = loss_all_test[jnp.arange(Nt), best_indices]
        loss_eqn = jnp.sum(loss_all_test[jnp.arange(Nt), best_indices])
        # Compute absolute loss, which is the L1 norm of the residual. Just for comparison and debugging. We do not use it in the optimization.
        loss_l1 = jnp.sum(loss_l1_all)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        return best_coeffs_all, current_key, loss_eqn, loss_l1, loss_each_elem

    def loss_fn_elem_int_per_elem(self, parti_idx, p_coeffs_all, params):
        quad_points = self.pwp.quad_points_parti[parti_idx]
        quad_weights = self.pwp.quad_weights_parti[parti_idx]
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        if self.input_dim == 2:
            u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
            residual = self.res(u_quad, u_t, u_x).flatten()
            if self.dg_type == 'P1':
                k_h_quad_all = (
                    p_coeffs_all[:, 0][:, None] + 
                    p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
                    p_coeffs_all[:, 2][:, None] * quad_points[:, 1]
                    )
            elif self.dg_type == 'Q1':
                # k_h_quad_all = (
                #     p_coeffs_all[:, 0][:, None] + 
                #     p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
                #     p_coeffs_all[:, 2][:, None] * quad_points[:, 1] + 
                #     p_coeffs_all[:, 3][:, None] * quad_points[:, 0] * quad_points[:, 1]
                #     )
                partition = self.pwp.partitions[parti_idx]
                t_low, x_low = partition[0]
                t_high, x_high = partition[3]
                local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
                local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
                k_h_quad_all = (
                    p_coeffs_all[:, 0][:, None] * (1 - local_t) * (1 - local_x)  + 
                    p_coeffs_all[:, 1][:, None] * (1 - local_t) * local_x  + 
                    p_coeffs_all[:, 2][:, None] * local_t * (1 - local_x)  + 
                    p_coeffs_all[:, 3][:, None] * local_t * local_x
                    )

        elif self.input_dim == 3:
            u_t, u_x, u_y = grad_u_tx[:, 0], grad_u_tx[:, 1], grad_u_tx[:, 2]
            residual = self.res(u_quad, u_t, u_x, u_y).flatten()
            if self.dg_type == 'P1':
                k_h_quad_all = (
                    p_coeffs_all[:, 0][:, None] + 
                    p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
                    p_coeffs_all[:, 2][:, None] * quad_points[:, 1] + 
                    p_coeffs_all[:, 3][:, None] * quad_points[:, 2]
                    )
            elif self.dg_type == 'Q1':
                partition = self.pwp.partitions[parti_idx]
                t_low, x_low, y_low = partition[0]
                t_high, x_high, y_high = partition[-1]
                local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
                local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
                local_y = (quad_points[:, 2] - y_low) / (y_high - y_low)
                k_h_quad_all = (
                    p_coeffs_all[:, 0][:, None] * (1 - local_t) * (1 - local_x) * (1 - local_y) + 
                    p_coeffs_all[:, 1][:, None] * (1 - local_t) * (1 - local_x) * local_y +
                    p_coeffs_all[:, 2][:, None] * (1 - local_t) * local_x * (1 - local_y) + 
                    p_coeffs_all[:, 3][:, None] * (1 - local_t) * local_x * local_y + 
                    p_coeffs_all[:, 4][:, None] * local_t * (1 - local_x) * (1 - local_y) + 
                    p_coeffs_all[:, 5][:, None] * local_t * (1 - local_x) * local_y + 
                    p_coeffs_all[:, 6][:, None] * local_t * local_x * (1 - local_y) + 
                    p_coeffs_all[:, 7][:, None] * local_t * local_x * local_y
                    )
        # Compute k_h for the current element
        # k_h_quad_all = p_coeffs_all[:, 1:]@quad_points.T + p_coeffs_all[:, 0][:, None] # This code looks concise but it costs more memory.
        # print(f"k_h_quad_all.shape: {k_h_quad_all.shape}, u_quad.shape: {u_quad.shape}, residual.shape: {residual.shape}")
        loss = jnp.dot(jnp.sign(u_quad[None, :] - k_h_quad_all), (quad_weights*residual))
        # Compute absolute loss, which is the L1 norm of the residual. Just for comparison and debugging. We do not use it in the optimization.
        loss_l1 = jnp.dot(jnp.abs(residual), quad_weights)
        return loss, loss_l1
    
    
    
    # def loss_fn_elem_int(self, params):
    #     p_coeffs = self.pwp.coeffs[:, None, :]  # (Nt, 1, p_dof)
    #     Nt = self.pwp.N_elements
    #     loss_all_elems, loss_l1_elems = self.vmap_loss_fn_elem_int_per_elem(jnp.arange(Nt), p_coeffs, params)
    #     loss_eqn = jnp.sum(loss_all_elems)
    #     # Compute absolute loss, which is the L1 norm of the residual. Just for comparison and debugging. We do not use it in the optimization.
    #     # loss_l1 = jnp.sum(loss_l1_elems)
    #     total_loss = (1 - self.L1reg) * loss_eqn
    #     if self.L1reg > 1e-6: 
    #         loss_l1 = jnp.sum(loss_l1_elems)
    #         total_loss += self.L1reg *  self.hx * loss_l1
    #     return total_loss


    def loss_fn_elem_int(self, params, importance_probs=None, current_key=None):
        if importance_probs is not None:
            key, current_key = jax.random.split(current_key)
            batch_size = int(0.8 * self.pwp.N_elements)
            element_indices_sampled = jax.random.choice(key, jnp.arange(self.pwp.N_elements), shape=(batch_size,), p=importance_probs)
            quad_points_sampled = self.pwp.quad_points_parti[element_indices_sampled].reshape(-1, self.input_dim)
            quad_weights_sampled = self.pwp.quad_weights_parti[element_indices_sampled].reshape(-1)
            quad_element_indices_sampled = jnp.repeat(element_indices_sampled, self.pwp.Nq)
            quad_points = quad_points_sampled
            quad_weights = quad_weights_sampled
        else:
            quad_element_indices_sampled = self.pwp.quad_element_indices
            quad_points = self.pwp.quad_points
            quad_weights = self.pwp.quad_weights
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        # print(f"grad_u_tx.shape: {grad_u_tx.shape}", f"u_quad.shape: {u_quad.shape}")
        p_coeffs_quad = self.pwp.coeffs[quad_element_indices_sampled]
        if self.input_dim == 2:
            u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
            residual = self.res(u_quad, u_t, u_x)
            if self.dg_type == 'P1':
                k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
            elif self.dg_type == 'Q1':
                partitions = self.pwp.partitions[quad_element_indices_sampled]
                t_low = partitions[:, 0, 0]
                x_low = partitions[:, 0, 1]
                t_high = partitions[:, -1, 0]
                x_high = partitions[:, -1, 1]
                local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
                local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
                k_h_quad = (
                    p_coeffs_quad[:, 0] * (1 - local_t) * (1 - local_x)  + 
                    p_coeffs_quad[:, 1] * (1 - local_t) * local_x  + 
                    p_coeffs_quad[:, 2] * local_t * (1 - local_x)  + 
                    p_coeffs_quad[:, 3] * local_t * local_x
                    )

        elif self.input_dim == 3:
            u_t, u_x, u_y = grad_u_tx[:, 0], grad_u_tx[:, 1], grad_u_tx[:, 2]
            residual = self.res(u_quad, u_t, u_x, u_y)
            if self.dg_type == 'P1':
                k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1] + p_coeffs_quad[:, 3] * quad_points[:, 2]
            elif self.dg_type == 'Q1':
                partitions = self.pwp.partitions[quad_element_indices_sampled]
                t_low = partitions[:, 0, 0]
                x_low = partitions[:, 0, 1]
                y_low = partitions[:, 0, 2]
                t_high = partitions[:, -1, 0]
                x_high = partitions[:, -1, 1]
                y_high = partitions[:, -1, 2]
                local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
                local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
                local_y = (quad_points[:, 2] - y_low) / (y_high - y_low)
                k_h_quad = (
                    p_coeffs_quad[:, 0] * (1 - local_t) * (1 - local_x) * (1 - local_y) + 
                    p_coeffs_quad[:, 1] * (1 - local_t) * (1 - local_x) * local_y +
                    p_coeffs_quad[:, 2] * (1 - local_t) * local_x * (1 - local_y) + 
                    p_coeffs_quad[:, 3] * (1 - local_t) * local_x * local_y + 
                    p_coeffs_quad[:, 4] * local_t * (1 - local_x) * (1 - local_y) + 
                    p_coeffs_quad[:, 5] * local_t * (1 - local_x) * local_y + 
                    p_coeffs_quad[:, 6] * local_t * local_x * (1 - local_y) + 
                    p_coeffs_quad[:, 7] * local_t * local_x * local_y
                    )
        
        # k_h_quad = jnp.einsum('ij, ij->i', p_coeffs_quad[:, 1:], quad_points) + p_coeffs_quad[:, 0]
        integrand = (1 - self.L1reg) * residual * jnp.sign(u_quad - k_h_quad)
        if self.L1reg > 1e-6:
            integrand += self.L1reg * jnp.abs(self.hx * residual)
        if importance_probs is not None:
            probs_sampled = importance_probs[quad_element_indices_sampled]
            weighted_integrand = integrand / probs_sampled
            interior_loss = jnp.dot(quad_weights, weighted_integrand) / batch_size
            return interior_loss, current_key
        else:
            interior_loss = jnp.dot(quad_weights, integrand)
            return interior_loss
    
    # def loss_fn_elem_int(self, params):
    #     quad_points = self.pwp.quad_points
    #     quad_weights = self.pwp.quad_weights
    #     u_quad = self.u_eval(params, quad_points)
    #     grad_u_tx = self.grad_u(params, quad_points)
    #     # print(f"grad_u_tx.shape: {grad_u_tx.shape}", f"u_quad.shape: {u_quad.shape}")
    #     p_coeffs_quad = self.pwp.coeffs[self.pwp.quad_element_indices]
    #     if self.input_dim == 2:
    #         u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
    #         residual = self.res(u_quad, u_t, u_x)
    #         if self.dg_type == 'P1':
    #             k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]
    #         elif self.dg_type == 'Q1':
    #             partitions = self.pwp.partitions[self.pwp.quad_element_indices]
    #             t_low = partitions[:, 0, 0]
    #             x_low = partitions[:, 0, 1]
    #             t_high = partitions[:, -1, 0]
    #             x_high = partitions[:, -1, 1]
    #             local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
    #             local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
    #             k_h_quad = (
    #                 p_coeffs_quad[:, 0] * (1 - local_t) * (1 - local_x)  + 
    #                 p_coeffs_quad[:, 1] * (1 - local_t) * local_x  + 
    #                 p_coeffs_quad[:, 2] * local_t * (1 - local_x)  + 
    #                 p_coeffs_quad[:, 3] * local_t * local_x
    #                 )

    #     elif self.input_dim == 3:
    #         u_t, u_x, u_y = grad_u_tx[:, 0], grad_u_tx[:, 1], grad_u_tx[:, 2]
    #         residual = self.res(u_quad, u_t, u_x, u_y)
    #         if self.dg_type == 'P1':
    #             k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1] + p_coeffs_quad[:, 3] * quad_points[:, 2]
    #         elif self.dg_type == 'Q1':
    #             partitions = self.pwp.partitions[self.pwp.quad_element_indices]
    #             t_low = partitions[:, 0, 0]
    #             x_low = partitions[:, 0, 1]
    #             y_low = partitions[:, 0, 2]
    #             t_high = partitions[:, -1, 0]
    #             x_high = partitions[:, -1, 1]
    #             y_high = partitions[:, -1, 2]
    #             local_t = (quad_points[:, 0] - t_low) / (t_high - t_low)
    #             local_x = (quad_points[:, 1] - x_low) / (x_high - x_low)
    #             local_y = (quad_points[:, 2] - y_low) / (y_high - y_low)
    #             k_h_quad = (
    #                 p_coeffs_quad[:, 0] * (1 - local_t) * (1 - local_x) * (1 - local_y) + 
    #                 p_coeffs_quad[:, 1] * (1 - local_t) * (1 - local_x) * local_y +
    #                 p_coeffs_quad[:, 2] * (1 - local_t) * local_x * (1 - local_y) + 
    #                 p_coeffs_quad[:, 3] * (1 - local_t) * local_x * local_y + 
    #                 p_coeffs_quad[:, 4] * local_t * (1 - local_x) * (1 - local_y) + 
    #                 p_coeffs_quad[:, 5] * local_t * (1 - local_x) * local_y + 
    #                 p_coeffs_quad[:, 6] * local_t * local_x * (1 - local_y) + 
    #                 p_coeffs_quad[:, 7] * local_t * local_x * local_y
    #                 )
        
    #     # k_h_quad = jnp.einsum('ij, ij->i', p_coeffs_quad[:, 1:], quad_points) + p_coeffs_quad[:, 0]
    #     integrand = (1 - self.L1reg) * residual * jnp.sign(u_quad - k_h_quad)
    #     if self.L1reg > 1e-6:
    #         integrand += self.L1reg * jnp.abs(self.hx * residual)
    #     interior_loss = jnp.dot(quad_weights, integrand)
    #     total_loss = interior_loss
    #     return total_loss


    def loss_fn_ic(self, params):
        return jnp.mean((self.u_eval(params, self.tx_t0) - self.u_t0)**2)

    def loss_fn_bc_Dirichlet(self, params):
        bc_loss = jnp.mean((self.u_eval(params, self.tx_x0) - self.u_x0)**2) + jnp.mean((self.u_eval(params, self.tx_x1) - self.u_x1)**2)
        if self.input_dim == 3:
            bc_loss += jnp.mean((self.u_eval(params, self.tx_y0) - self.u_y0)**2) + jnp.mean((self.u_eval(params, self.tx_y1) - self.u_y1)**2)
        return bc_loss
    
    def loss_fn_bc_inflow(self, params):
        bc_loss = 0.0

        # x = x_min boundary, n = (-1, 0)
        mask_x0 = (self.u_x0 * (-1) + self.u_x0 * 0) < 0  
        error_x0 = self.u_eval(params, self.tx_x0) - self.u_x0
        if jnp.any(mask_x0):
            bc_loss += jnp.mean(error_x0[mask_x0] ** 2)

        # x = x_max boundary, n = (1, 0)
        mask_x1 = (self.u_x1 * 1 + self.u_x1 * 0) < 0  # u_ref < 0
        error_x1 = self.u_eval(params, self.tx_x1) - self.u_x1
        if jnp.any(mask_x1):
            bc_loss += jnp.mean(error_x1[mask_x1] ** 2)

        if self.input_dim == 3:
            # y = y_min boundary, n = (0, -1)
            mask_y0 = (self.u_y0 * 0 + self.u_y0 * (-1)) < 0  # -u_ref < 0 => u_ref > 0
            error_y0 = self.u_eval(params, self.tx_y0) - self.u_y0
            if jnp.any(mask_y0):
                bc_loss += jnp.mean(error_y0[mask_y0] ** 2)

            # y = y_max boundary, n = (0, 1)
            mask_y1 = (self.u_y1 * 0 + self.u_y1 * 1) < 0  # u_ref < 0
            error_y1 = self.u_eval(params, self.tx_y1) - self.u_y1
            if jnp.any(mask_y1):
                bc_loss += jnp.mean(error_y1[mask_y1] ** 2)
        return bc_loss

    def loss_fn_ibc(self, params):
        return self.loss_fn_ic(params) + self.loss_fn_bc(params)

    def loss_fn_lambda(self, params, importance_probs=None, current_key=None):
        loss_ic = self.loss_fn_ic(params)
        loss_bc = self.loss_fn_bc(params)
        if importance_probs is not None:
            loss_eqn, current_key = self.loss_fn_eqn(params, importance_probs, current_key)
            total_loss = self.lambda_eqn * loss_eqn + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc
            return total_loss, (loss_eqn, loss_ic + loss_bc, current_key)
        else:
            loss_eqn = self.loss_fn_eqn(params)
            total_loss = self.lambda_eqn * loss_eqn + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc
            return total_loss, (loss_eqn, loss_ic + loss_bc)
    
    def loss_fn_gamma(self, params, importance_probs=None, current_key=None):
        loss_ibc = self.loss_fn_ibc(params)
        if importance_probs is not None:
            loss_eqn, current_key = self.loss_fn_eqn(params, importance_probs, current_key)
            total_loss = self.gamma * loss_eqn + (1 - self.gamma) * loss_ibc
            return total_loss, (loss_eqn, loss_ibc, current_key)
        else:
            loss_eqn = self.loss_fn_eqn(params)
            total_loss = self.gamma * loss_eqn + (1 - self.gamma) * loss_ibc
            return total_loss, (loss_eqn, loss_ibc)
    
    # @partial(jax.jit, static_argnames=['self', 'optimizer'])
    def train_step(self, params, optimizer, opt_state, importance_probs=None, current_key=None):
        if self.use_importance_sampling:
            (_, all_losses), grads = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)(params, importance_probs, current_key)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return all_losses[0], all_losses[1], all_losses[2], params, opt_state # In this case, all_losses = (loss_eqn, loss_ibc, current_key)
        else:
            (_, all_losses), grads = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return all_losses[0], all_losses[1], params, opt_state

    def pre_train_to_ic(self, pre_epochs=0, pre_lr=1e-3):
        pre_epochs = pre_epochs if pre_epochs > 0 else self.pre_epochs
        if pre_epochs <= 0:
            print("Skipping pre-training as pre_epochs is set to 0.")
            return
        print("\nPre-training the network to fit v(t, x) = u0(x)")
        
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights
        # Generate pre-training points
        if self.input_dim == 2:
            y_pre = self.u0(quad_points[:, 1])
        elif self.input_dim == 3:
            y_pre = self.u0(quad_points[:, 1], quad_points[:, 2])

        # Collect all parameters
        params = self.model.mlps[0].params
        # lr = optax.schedules.cosine_decay_schedule(init_value=pre_lr, decay_steps=pre_epochs)
        optimizer = optax.adam(pre_lr)
        opt_state = optimizer.init(params)
        net_u = lambda params, quad_points: self.model.forward_with_params(params, quad_points, grade_idx=0)
        u_eval = jit(vmap(net_u, in_axes = (None, 0)))

        def pre_loss(params):
            u_pred = u_eval(params, quad_points)
            return jnp.dot(quad_weights, (u_pred - y_pre)**2)

        pre_loss_grad = jax.value_and_grad(pre_loss)

        for iter in range(pre_epochs):
            loss, grads = pre_loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if iter % 500 == 0:
                print(f"Pre-training Iter {iter}, Loss: {loss:.6f}")

        # Assign back the pre-trained parameters
        self.model.mlps[0].params = params
    
    def pre_train_to_u_ref(self, pre_epochs=0, pre_lr=1e-3):
        pre_epochs = pre_epochs if pre_epochs > 0 else self.pre_epochs
        if self.u_ref is None:
            print("Skipping pre-training to fit u_ref as u_ref is not provided.")
            return
        if pre_epochs <= 0:
            print("Skipping pre-training to fit u_ref as pre_epochs is set to 0.")
            return
        print("\nPre-training the network to fit the reference solution u_ref over the domain")
        quad_points = self.pwp.quad_points
        quad_weights = self.pwp.quad_weights
        # Generate target values from u_ref
        if self.input_dim == 2:
            y_pre = self.u_ref(quad_points[:, 0], quad_points[:, 1])
        elif self.input_dim == 3:
            y_pre = self.u_ref(quad_points[:, 0], quad_points[:, 1], quad_points[:, 2])

        # Collect all parameters (pre-train the first grade)
        params = self.model.mlps[0].params
        optimizer = optax.adam(pre_lr)
        opt_state = optimizer.init(params)
        net_u = lambda params, quad_points: self.model.forward_with_params(params, quad_points, grade_idx=0)
        u_eval = jit(vmap(net_u, in_axes = (None, 0)))

        def pre_loss(params):
            u_pred = u_eval(params, quad_points)
            return jnp.dot(quad_weights, (u_pred - y_pre)**2)

        pre_loss_grad = jax.value_and_grad(pre_loss)

        for iter in range(pre_epochs):
            loss, grads = pre_loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if iter % 500 == 0:
                print(f"Pre-training Iter {iter}, Loss: {loss:.6f}")

        # Assign back the pre-trained parameters
        self.model.mlps[0].params = params
    
    def pre_train_to_pinn(self, pre_epochs=0, pre_lr=1e-3):
        pre_epochs = pre_epochs if pre_epochs > 0 else self.pre_epochs
        if pre_epochs <= 0:
            print("Skipping pre-training to pinn as pre_epochs is set to 0.")
            return
        print("\nPre-training the network using PINN loss over the quadrature points")
        if self.input_dim == 2:
            pwp_pinn = PiecewiseLinearPolynomialsRect(self.N_t, self.N_x, t_min=self.t_min, t_max=self.t_max, x_min=self.x_min, x_max=self.x_max, 
                                                     gauss_order=2, dg_type='Q1')
            print("Using 2D quadrature from Rect meshing for PINN pre-training.")
        elif self.input_dim == 3:
            pwp_pinn = PiecewiseLinearPolynomialsHex(self.N_t, self.N_x, self.N_y, t_min=self.t_min, t_max=self.t_max, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, 
                                                     gauss_order=2, dg_type='Q1')
            print("Using 3D quadrature from Hex meshing for PINN pre-training.")
        # Collect all parameters (pre-train the first grade)
        params = self.model.mlps[0].params
        lr = optax.schedules.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=pre_lr, warmup_steps=int(pre_epochs/100), 
                                                          decay_steps=pre_epochs, end_value=1e-5)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        def pre_loss(params):
            loss_eqn = self.loss_fn_pinn(params, quad_points=pwp_pinn.quad_points, quad_weights=pwp_pinn.quad_weights)
            loss_ic = self.loss_fn_ic(params)
            loss_bc = self.loss_fn_bc(params)
            total_loss = self.lambda_eqn * loss_eqn + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc
            return total_loss, (loss_eqn, loss_ic+loss_bc)

        pre_loss_grad = jax.value_and_grad(pre_loss, argnums=0, has_aux=True)

        for iter in range(pre_epochs):
            (loss, all_losses), grads = pre_loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if iter % 500 == 0:
                # print(f"Pre-training Iter {iter}, Loss: {loss:.6f}")
                print(f"Pre-training Iter {iter}, loss_eqn: {all_losses[0]:.6f}, loss_ibc: {all_losses[1]:.6f}")

        # Assign back the pre-trained parameters
        self.model.mlps[0].params = params

    def loss_fn_pinn(self, params, importance_probs=None, current_key=None, quad_points=None, quad_weights=None):
        """Loss function for PINN training."""
        # Interior loss
        if quad_points is None or quad_weights is None:
            quad_points = self.pwp.quad_points
            quad_weights = self.pwp.quad_weights
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        lap = self.laplacian(params, quad_points)
        if self.input_dim == 2:
            u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
            residual = self.res(u_quad, u_t, u_x) - self.nu*lap
        elif self.input_dim == 3:
            u_t, u_x, u_y = grad_u_tx[:, 0], grad_u_tx[:, 1], grad_u_tx[:, 2]
            residual = self.res(u_quad, u_t, u_x, u_y) - self.nu*lap
        interior_loss = jnp.dot(quad_weights, residual**2)
        return interior_loss
    
    def train_process_stage1(self, grade_epochs=None, save_model=True):
        if self.pre_train == 'ic':
            self.pre_train_to_ic()
        elif self.pre_train == 'u_ref':
            self.pre_train_to_u_ref()
        elif self.pre_train == 'pinn':
            self.pre_train_to_pinn()
        print("\nStarting Train Stage with MGDL")
        if grade_epochs is None:
            grade_epochs = self.grade_epochs
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
            lr = optax.schedules.warmup_cosine_decay_schedule(init_value=1e-6, peak_value=self.init_lrs[grade_idx], warmup_steps=int(grade_epochs[grade_idx]/100), 
                                                              decay_steps=grade_epochs[grade_idx], end_value=1e-5)
            optimizer = optax.adam(lr)
            params = self.model.mlps[grade_idx].params
            opt_state = optimizer.init(params)
            self.optimizer_list.append(optimizer)
            self.opt_state_list.append(opt_state)

            self.gen_funcs(grade_idx)
            self.maximize_kh = jit(self.maximize_kh_step)
            if self.loss_type != 'pinn':
                self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1, loss_each_elem = self.maximize_kh(params, self.key)
                if self.use_importance_sampling:
                    self.update_importance_probs(loss_each_elem)
            if self.pre_train == 'u_ref':
                plot_u_theta_mesh(   
                    model=self.model,
                    M_t=self.M_t,
                    M_x=self.M_x,
                    t_min=self.t_min,
                    t_max=self.t_max,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    y_min=self.y_min if self.input_dim == 3 else None,
                    y_max=self.y_max if self.input_dim == 3 else None,
                    upload_fig=self.set_wandb,
                    pwp=self.pwp,
                    u_ref=self.u_ref if self.input_dim == 3 else None,
                    vals=loss_each_elem if self.plt_err_dist else None,
                    )
            if self.gamma > 0.0 and self.adjust_gamma:
                self.gamma = self.determine_equation_weight(grade_idx, gamma_init=self.gamma)
                print(f"Stage 1, Gamma: {self.gamma:.6f}")
            monitor_start = int(0.9 * grade_epochs[grade_idx])
            for iter in range(grade_epochs[grade_idx]):
                if (iter % self.f_adaptive_weight == 0) and (self.gamma == 0.0):
                    self.update_adaptive_weights(params)
                    print(f"Grade {grade_idx}, Iter {iter}, lambda_ic: {self.lambda_ic:.6f}, lambda_bc: {self.lambda_bc:.6f}, lambda_eqn: {self.lambda_eqn:.6f}")
                if self.use_importance_sampling:
                    loss_eqn, loss_ibc, self.key, params, opt_state = self.train_step(params, optimizer, opt_state, self.importance_probs, self.key)
                else:
                    loss_eqn, loss_ibc, params, opt_state = self.train_step(params, optimizer, opt_state)
                if iter % self.f_opt_kh == 0 and self.loss_type != 'pinn':
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1, loss_each_elem = self.maximize_kh(params, self.key)
                loss_eqn = self.loss_fn_eqn(params)
                self.loss_eqn_history[grade_idx].append(loss_eqn)
                self.loss_bc_history[grade_idx].append(loss_ibc)
                if iter >= monitor_start:
                    if loss_eqn < self.best_loss_eqn_list[grade_idx]:
                        self.best_loss_eqn_list[grade_idx] = loss_eqn
                        self.best_params_list[grade_idx] = jax.tree_util.tree_map(jnp.copy, params)

                if iter % 100 == 0:
                    if self.loss_type == 'pinn':
                        print(f"Grade {grade_idx}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_ibc: {loss_ibc:.6f}")
                    else:
                        if self.use_importance_sampling:
                            self.update_importance_probs(loss_each_elem)
                        print(f"Grade {grade_idx}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_eqn_from_kh: {loss_eqn_from_kh:.6f}", 
                              f"loss_l1: {loss_l1:.6f}, loss_ibc: {loss_ibc:.6f}")
                    if self.set_wandb:
                        wandb.log({f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_eqn_grade{grade_idx}": loss_eqn, 
                                   f"tmin{self.t_min:.1f}_tmax{self.t_max:.1f}_loss_ibc_grade{grade_idx}": loss_ibc, 
                                   f"step": self.global_step + iter})
                if self.f_plot > 0 and iter % self.f_plot == 0:
                    plot_u_theta(
                        prob=self,
                        M_t=self.M_t,
                        M_x=self.M_x,
                        t_min=self.t_min,
                        t_max=self.t_max,
                        x_min=self.x_min,
                        x_max=self.x_max,
                        y_min=self.y_min if self.input_dim == 3 else None,
                        y_max=self.y_max if self.input_dim == 3 else None,
                        ts=jnp.linspace(self.t_min, self.t_max, 2) if self.input_dim == 3 else None,
                        u_ref=self.u_ref if self.input_dim == 3 else None,
                        upload_fig=self.set_wandb
                        )
                    if self.plt_err_dist or self.plt_mesh:
                        plot_u_theta_mesh(   
                            model=self.model,
                            M_t=self.M_t,
                            M_x=self.M_x,
                            t_min=self.t_min,
                            t_max=self.t_max,
                            x_min=self.x_min,
                            x_max=self.x_max,
                            y_min=self.y_min if self.input_dim == 3 else None,
                            y_max=self.y_max if self.input_dim == 3 else None,
                            upload_fig=self.set_wandb,
                            pwp=self.pwp,
                            u_ref=self.u_ref if self.input_dim == 3 else None,
                            vals=loss_each_elem if self.plt_err_dist else None,
                            )
                self.model.mlps[grade_idx].params = params

            # Assign the best params found in the last 10%
            if self.best_params_list[grade_idx] is not None:
                self.model.mlps[grade_idx].params = self.best_params_list[grade_idx]
            self.global_step += grade_epochs[grade_idx]
        end_time = time.perf_counter()
        train_total_time = end_time - start_time
        print(f"Train time for stage1: {train_total_time:.6f} seconds")

    def train_process_stage2(self, save_model=False):
        # Stage 2: Refinement of Last k Grades
        if self.k > 0 and self.k <= self.total_grade:
            start_time = time.perf_counter()
            if self.gamma > 0.0 and self.adjust_gamma:
                self.gamma = self.determine_equation_weight(gamma_init=self.gamma)
            print(f"\nStage 2: Refining Last {self.k} Grades")
            self.gen_funcs()
            self.maximize_kh = jit(self.maximize_kh_step)
            loss_eqn_fn = jit(self.loss_fn_eqn)
            loss_data_fn = jit(self.loss_fn_ibc)
            params_last_k = [mlp.params for mlp in self.model.mlps[-self.k:]]
            self.opt_state_stage2 = self.optimizer_stage2.init(params_last_k)
            optimizer = self.optimizer_stage2
            opt_state = self.opt_state_stage2
            if self.loss_type != 'pinn':
                self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1 = self.maximize_kh(params_last_k, self.key)
            monitor_start = int(0.9 * self.epoch_stage2)
            best_loss_eqn = float('inf')
            best_params_last_k = None
            for iter in range(self.epoch_stage2):
                if iter % self.f_adaptive_weight == 0:
                    self.update_adaptive_weights(params_last_k)
                    print(f"Stage 2, Iter {iter}, lambda_ic: {self.lambda_ic:.6f}, lambda_bc: {self.lambda_bc:.6f}, lambda_eqn: {self.lambda_eqn:.6f}")
                loss_eqn, loss_ibc, params_last_k, opt_state = self.train_step(params_last_k, optimizer, opt_state)
                if iter % self.f_opt_kh == 0 and self.loss_type != 'pinn':
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1 = self.maximize_kh(params_last_k, self.key)
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
            if self.loss_type != 'pinn':
                self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1 = self.maximize_kh(current_params, self.key)
            for iter in range(self.N_test_gamma):
                (loss, (loss_eqn, loss_data)), grads = jax.value_and_grad(total_loss, has_aux=True)(current_params)
                updates, current_opt_state = optimizer.update(grads, current_opt_state)
                current_params = optax.apply_updates(current_params, updates)
                if iter % self.f_opt_kh == 0 and self.loss_type != 'pinn':
                    self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1 = self.maximize_kh(current_params, self.key)
                # Store losses
                loss_eqn = loss_eqn_fn(current_params)
                loss_data = loss_data_fn(current_params)
                loss_history_data.append(loss_data)
                loss_history_eqn.append(loss_eqn)
                if iter % 100 == 0:
                    if self.loss_type == 'pinn':
                        print(f"Gamma {gamma}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_ibc: {loss_data:.6f}")
                    else:
                        print(f"Gamma {gamma}, Iter {iter}, loss_eqn: {loss_eqn:.6f}, loss_eqn_from_kh: {loss_eqn_from_kh:.6f}, loss_l1: {loss_l1:.6f}, loss_ibc: {loss_data:.6f}")

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
            'use_a': self.use_a,
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
        use_a = loaded_data['use_a']
        self.model = MG_NN(input_dim=2, hidden_dims_list=loaded_hidden_dims_list, activation=nn.tanh, lb=lb, ub=ub, use_a=use_a)
        for i in range(loaded_total_grade):
            self.model.add_grade()  # Add a new grade
        for mlp, params in zip(self.model.mlps, loaded_params):
            mlp.params = params
        print(f"Model loaded from {model_path} with hidden_dims_list: {loaded_hidden_dims_list}, loaded_total_grade = {loaded_data['total_grade']}.")
        self.opt_state_list = [opt.init(mlp.params) for opt, mlp in zip(self.optimizer_list, self.model.mlps)]
        self.gen_funcs(loaded_total_grade-1)
        self.pwp.coeffs, self.key, loss_eqn_from_kh, loss_l1 = self.maximize_kh_step(loaded_params[-1], self.key)
        loss_eqn = self.loss_fn_eqn(loaded_params[-1])
        loss_ibc = self.loss_fn_ibc(loaded_params[-1])
        print(f"For the loaded model, loss_eqn: {loss_eqn:.6f}, Loss_bc: {loss_ibc:.6f}")