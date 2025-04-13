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
from my_helpers import plot_u_theta, plot_kh
from functools import partial
from Pwp_class import PiecewiseLinearPolynomials

# Utility function to compute L2 norm of a pytree
def pytree_l2_norm(pytree):
    leaves = tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))


# Optimizer Class
class MinMaxOptProb:
    def __init__(self, N_t=10, N_x=10, lr_theta=0.01, hidden_dims=(64, 64, 64), seed=0, sigma=0.1, N_perturb=5000, N_bc=100, N_ic=100,
                 t_min=0, t_max=1, x_min=0, x_max=1, k_h_rf_times=1, u0=None, u1=None, u2=None, lambda_bc_static=0.99, alpha_p_coeffs=0, 
                 f_adaptive=1000, nu=0.001, res_visc=None, res=None,):
        self.pwp = PiecewiseLinearPolynomials(N_t, N_x, t_min=t_min, t_max=t_max, x_min=x_min, x_max=x_max, times=k_h_rf_times)
        self.lr_theta = lr_theta
        self.lb = jnp.array([t_min, x_min])
        self.ub = jnp.array([t_max, x_max])
        self.model = MLP(hidden_dims=hidden_dims, activation=nn.tanh, lb=self.lb, ub=self.ub)
        key = jax.random.PRNGKey(seed)
        # self.params = self.model.init(key, jnp.ones((1, 2)))
        self.params = self.model.params
        # Define optimizer
        self.opt_theta = optax.adam(self.lr_theta)
        self.opt_state_theta = self.opt_theta.init(self.params)
        self.key = jax.random.PRNGKey(seed)
        self.sigma = sigma
        self.N_perturb = N_perturb
        self.N_bc = N_bc
        self.N_ic = N_ic
        # Sample points on each boundary
        self.t_bc = jnp.linspace(t_min, t_max, self.N_bc)  # t values for x=0 and x=1 boundaries
        self.x_bc = jnp.linspace(x_min, x_max, self.N_ic)  # x values for t=0 boundary
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.lambda_bc_static = lambda_bc_static
        # Adaptive weights for main training
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
        self.lambda_r = 1.0
        self.f_adaptive = f_adaptive
        self.alpha = 0.9  # Moving average parameter
        self.alpha_p_coeffs = alpha_p_coeffs # Moving average parameter for k_h
        self.nu = nu
        self.res_visc = res_visc
        self.res = res
        self.gen_ibc()
        self.gen_derivative()

    def gen_ibc(self):
         # Initial condition at t_min
        self.u_t0 = self.u0(self.x_bc)
        self.tx_t0 = jnp.stack([self.pwp.t_min * jnp.ones_like(self.x_bc), self.x_bc], axis=1)

        # Left Boundary at x_min
        self.u_x0 = self.u1(self.t_bc)
        self.tx_x0 = jnp.stack([self.t_bc, self.pwp.x_min * jnp.ones_like(self.t_bc)], axis=1)

        # Right Boundary at x_max
        self.u_x1 = self.u2(self.t_bc)
        self.tx_x1 = jnp.stack([self.t_bc, self.pwp.x_max * jnp.ones_like(self.t_bc)], axis=1)
        
    def gen_derivative(self):
        # net_u = lambda params, tx: self.model.apply(params, tx).squeeze()
        net_u = lambda params, tx: self.model(params, tx).squeeze()
        self.u_eval = jit(vmap(net_u, in_axes = (None, 0)))
        self.grad_u = jit(vmap(grad(net_u, argnums=1), in_axes = (None, 0)))
        self.hessian_u = jit(vmap(hessian(net_u, argnums=1), in_axes = (None, 0)))     

    def compute_loss_opt_kh_element(self, tri_idx, p_coeffs_all, params):
        """
        Compute losses for all perturbations on a single triangle, compatible with vmap.
    
        Args:
            tri_idx: Index of the triangle.
            p_coeffs_all: Array of shape (N_perturb, 3) of perturbed coefficients.
        Returns:
            losses: Array of shape (N_perturb,) containing loss values for each perturbation.
        """
        # Retrieve precomputed quadrature points and weights
        quad_points = self.pwp.quad_points_tri[tri_idx]  # Shape: (Nq, 2)
        quad_weights = self.pwp.quad_weights_tri[tri_idx].flatten()  # Shape: (Nq,)
    
        # Evaluate the neural network and its gradients once for the triangle
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
    
        # Compute residual (e.g., u_t + u * u_x), fixed for all perturbations
        residual = self.res(u_quad, u_t, u_x).flatten()  # Shape: (Nq,)
    
        # Compute k_h for all perturbations at once
        k_h_quad_all = (
            p_coeffs_all[:, 0][:, None] + 
            p_coeffs_all[:, 1][:, None] * quad_points[:, 0] + 
            p_coeffs_all[:, 2][:, None] * quad_points[:, 1]
        )  # Shape: (N_perturb, Nq)
    
        # Compute integrand for all perturbations using broadcasting
        integrand_all = residual[None, :] * jnp.sign(u_quad[None, :] - k_h_quad_all)  # Shape: (N_perturb, Nq)
    
        # Compute losses by summing over quadrature points for each perturbation
        losses = jnp.sum(quad_weights[None, :] * integrand_all, axis=1)  # Shape: (N_perturb,)
        return losses

    def maximize_kh_vec(self, params, current_key):
        Nt = self.pwp.N_elements
        vertices_all = self.pwp.triangles  # Shape: (Nt, 3, 2)
        # Evaluate model on all vertices
        u_values_all = self.u_eval(params, vertices_all.reshape(-1, 2)).reshape(Nt, 3, -1)
        
        # Compute initial coefficients
        A_all = jnp.stack([jnp.ones((Nt, 3)), vertices_all[:, :, 0], vertices_all[:, :, 1]], axis=-1)
        initial_coeffs_all = jnp.linalg.solve(A_all, u_values_all.reshape(Nt, 3, 1)).squeeze(-1)
        # print(jnp.einsum('ijk,ik->ij', A_all, initial_coeffs_all)-u_values_all)
    
        # Generate perturbations
        key, current_key = jax.random.split(current_key)
        keys = jax.random.split(key, Nt)
        perturbations_all = vmap(lambda k: self.sigma * jax.random.normal(k, (self.N_perturb, 3)))(keys)
        perturbed_coeffs_all = initial_coeffs_all[:, None, :] + perturbations_all
        # print(perturbed_coeffs_all.shape)
        # Compute losses (assuming compute_loss_tri is defined)
        losses_all = vmap(self.compute_loss_opt_kh_element, in_axes=(0, 0, None))(jnp.arange(Nt), perturbed_coeffs_all, params)
        
        # print(losses_all.shape)
        # Select best coefficients
        best_indices = jnp.argmax(losses_all, axis=1)
        best_coeffs_all = perturbed_coeffs_all[jnp.arange(Nt), best_indices]
        # print(f"The max. of optimized coef. for k_h: {jnp.max(jnp.linalg.norm(best_coeffs_all[:,1:]))}")
        return best_coeffs_all, current_key

    def compute_loss_opt_utheta_int(self, params, p_coeffs):
        # Get quadrature points and weights for this triangle
        quad_points = self.pwp.quad_points  # Shape: (Nt*Nq, 2)
        quad_weights = self.pwp.quad_weights.flatten()  # Shape: (Nt*Nq,)
    
        # Evaluate neural network and its gradients at quadrature points
        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
    
        # Compute the residual (example: u_t + u * u_x for a simple PDE)
        residual = self.res(u_quad, u_t, u_x).flatten()  # Shape: (Nt*Nq,)
    
        # Compute k_h at quadrature points using the element's coefficients
        p_coeffs_quad = p_coeffs[self.pwp.element_indices]  # Shape: (Nt*Nq,)
        # print(f"p_coeffs_quad shape: {p_coeffs_quad.shape}")
        k_h_quad = p_coeffs_quad[:, 0] + p_coeffs_quad[:, 1] * quad_points[:, 0] + p_coeffs_quad[:, 2] * quad_points[:, 1]  # Shape: (Nt*Nq,)

        # Define the integrand for the loss (residual * sign(u - k_h))
        integrand = residual * jnp.sign(u_quad - k_h_quad)  # Shape: (Nt*Nq,)
        interior_loss = jnp.dot(quad_weights, integrand)  # Scalar
        return interior_loss
        
    def compute_loss_opt_utheta_bc(self, params):
        return self.compute_L_ic(params) + self.compute_L_bc(params)

    def compute_loss_opt_utheta_adaptive(self, params, p_coeffs, lambda_ic, lambda_bc, lambda_r):
        """Compute total loss with adaptive weights."""
        L_ic = self.compute_L_ic(params)
        L_bc = self.compute_L_bc(params)
        L_r = self.compute_loss_opt_utheta_int(params, p_coeffs)
        total_loss = (lambda_ic * L_ic + 
                      lambda_bc * L_bc + 
                      lambda_r * L_r)
        return total_loss, (L_r, L_ic+L_bc)
        
    def compute_loss_opt_utheta(self, params, p_coeffs):
        # --- Total Loss ---
        interior_loss = self.compute_loss_opt_utheta_int(params, p_coeffs)
        boundary_loss = self.compute_loss_opt_utheta_bc(params)
        total_loss = (1-self.lambda_bc_static) * interior_loss + self.lambda_bc_static * boundary_loss
        return total_loss

    # def test_u_theta_loss(self, params):
    #     u_quad = self.u_eval(params, self.pwp.quad_points)
    #     residual = u_quad - jnp.sin(2*(2*jnp.pi*self.pwp.quad_points[:, 0] + self.pwp.quad_points[:, 1]))
    #     loss = jnp.sum(residual**2 * self.pwp.quad_weights)
    #     return loss

    def minimize_u_theta(self, params, p_coeffs, opt_state_theta):
        loss_value, grad_params = jax.value_and_grad(self.compute_loss_opt_utheta, argnums=0)(params, p_coeffs)
        # loss_value, grad_params = jax.value_and_grad(self.test_u_theta_loss)(params)
        updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
        params = optax.apply_updates(params, updates)
        return loss_value, params, opt_state_theta

    def minimize_u_theta_adaptive(self, params, p_coeffs, opt_state_theta, lambda_ic, lambda_bc, lambda_r):
        """Minimize u_theta with adaptive weights."""
        (_, L_all), grad_params = jax.value_and_grad(self.compute_loss_opt_utheta_adaptive, 
                                                     argnums=0, has_aux=True)(params, p_coeffs, lambda_ic, lambda_bc, lambda_r)
        updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
        params = optax.apply_updates(params, updates)
        return L_all, params, opt_state_theta


    def pre_train_compute_loss_opt_utheta_int(self, params):
        # Get quadrature points and weights for this triangle
        quad_points = self.pwp.quad_points  # Shape: (Nt*Nq, 2)
        quad_weights = self.pwp.quad_weights.flatten()  # Shape: (Nt*Nq,)

        u_quad = self.u_eval(params, quad_points)
        grad_u_tx = self.grad_u(params, quad_points)
        u_t, u_x = grad_u_tx[:, 0], grad_u_tx[:, 1]
        hessian_u_tx = self.hessian_u(params, quad_points)
        u_xx = hessian_u_tx[:, 1, 1].flatten()  # Shape: (Nt*Nq,), second derivative w.r.t. x
        # Compute the residual (example: u_t + u * u_x for a simple PDE)
        residual = self.res_visc(u_quad, u_t, u_x, u_xx, self.nu)  # Shape: (Nt*Nq,)
        integrand = residual**2
        interior_loss = jnp.dot(quad_weights, integrand)  # Scalar
        return interior_loss

    # def pre_train_compute_loss_opt_utheta(self, params):
    #     interior_loss = self.pre_train_compute_loss_opt_utheta_int(params)
    #     boundary_loss = self.compute_loss_opt_utheta_bc(params)
    #     total_loss = (1-self.lambda_bc_static) * interior_loss + self.lambda_bc_static * boundary_loss
    #     return total_loss

    def pre_train_compute_loss_opt_utheta_adaptive(self, params, lambda_ic=0, lambda_bc=0, lambda_r=1):
        L_ic = self.compute_L_ic(params)
        L_bc = self.compute_L_bc(params)
        L_r = self.pre_train_compute_loss_opt_utheta_int(params)
        total_loss = (lambda_ic * L_ic + 
                      lambda_bc * L_bc + 
                      lambda_r * L_r)
        return total_loss, (L_r, L_ic+L_bc)

    # def pre_train_minimize_u_theta(self, params, opt_state_theta):
    #     loss_value, grad_params = jax.value_and_grad(self.pre_train_compute_loss_opt_utheta)(params)
    #     # loss_value, grad_params = jax.value_and_grad(self.test_u_theta_loss)(params)
    #     updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
    #     params = optax.apply_updates(params, updates)
    #     return loss_value, params, opt_state_theta

    def pre_train_minimize_u_theta_adaptive(self, params, opt_state_theta, lambda_ic=0, lambda_bc=0, lambda_r=1):
        (_, L_all), grad_params = jax.value_and_grad(self.pre_train_compute_loss_opt_utheta_adaptive, 
                                                     argnums=0, has_aux=True)(params, lambda_ic, lambda_bc, lambda_r)
        # loss_val, grad_params = jax.value_and_grad(self.test_u_theta_loss)(params)
        updates, opt_state_theta = self.opt_theta.update(grad_params, opt_state_theta)
        params = optax.apply_updates(params, updates)
        return L_all, params, opt_state_theta

    def pre_train_step(self, iter):
        if iter % self.f_adaptive == 0:
            self.update_adaptive_weights(self.params, self.pwp.coeffs, phase='pre-train')
            print(f"Iteration {iter}, lambda_ic: {self.lambda_ic}, lambda_bc: {self.lambda_bc}, lambda_r: {self.lambda_r}.")
        (loss_int, loss_ibc), self.params, self.opt_state_theta = self.pre_train_minimize_u(self.params, self.opt_state_theta,
                                                                         self.lambda_ic, self.lambda_bc, self.lambda_r)
        # return self.compute_loss_opt_utheta(self.params)
        return loss_int, loss_ibc
        # return self.pre_train_compute_loss_opt_utheta_int(self.params), self.compute_loss_opt_utheta_bc(self.params)
        
    def pre_train_process(self, pre_train_num_iterations, save_loss=True, save_model=True):
        print("\nStarting Pre-Train Stage")
        start_time = time.perf_counter()
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
        self.lambda_r = 1.0
        self.pre_loss_int_history = []
        self.pre_loss_bc_history = []
        self.pre_train_minimize_u = jit(self.pre_train_minimize_u_theta_adaptive)
        # self.pre_train_minimize_u = self.pre_train_minimize_u_theta
        for iter in range(pre_train_num_iterations):
            loss_int, loss_ibc = self.pre_train_step(iter)
            self.pre_loss_int_history.append(loss_int)
            self.pre_loss_bc_history.append(loss_ibc)
            if iter % 200 == 0:
                print(f"Pre-train Iteration {iter}, Loss_int: {loss_int:.8f}, Loss_ibc: {loss_ibc:.8f}")
        end_time = time.perf_counter()
        pre_train_total_time = end_time - start_time
        print(f"Pre-train total time: {pre_train_total_time:.8f} seconds")
        # print(f"Pre-train average time per iteration: {pre_train_total_time / pre_train_num_iterations:.9e} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_loss:
            # Save training loss histories with consistent naming
            loss_int_path = os.path.join('..', 'data', f'pre_train_loss_int_history_{timestamp}.pkl')
            with open(loss_int_path, 'wb') as f:
                pickle.dump(self.pre_loss_int_history, f)
            loss_bc_path = os.path.join('..', 'data', f'pre_train_loss_bc_history_{timestamp}.pkl')
            with open(loss_bc_path, 'wb') as f:
                pickle.dump(self.pre_loss_bc_history, f)
            print(f"Pre-train loss histories saved to {loss_int_path} and {loss_bc_path}")
        
        if save_model:
            # Save both params and architecture info
            model_path = os.path.join('..', 'data', f'pre_train_model_{timestamp}.pkl')
            self.save_model(model_path)
            print(f"Pre-train model saved to {model_path}")
                

    # --- Define Individual Loss Components ---
    def compute_L_ic(self, params):
        """Compute initial condition loss L_ic."""
        # Intial condition at t_min
        return jnp.mean((self.u_eval(params, self.tx_t0) - self.u_t0)**2)

    def compute_L_bc(self, params):
        """Compute boundary condition loss L_bc."""
        # Left Boundary at x_min
        u_x0_nn = self.u_eval(params, self.tx_x0)
        bc_x0_loss = jnp.mean((u_x0_nn - self.u_x0)**2)
        # Right Boundary at x_max
        u_x1_nn = self.u_eval(params, self.tx_x1)
        bc_x1_loss = jnp.mean((u_x1_nn - self.u_x1)**2)
        return bc_x0_loss + bc_x1_loss

    # --- Adaptive Weight Update ---
    def update_adaptive_weights(self, params, p_coeffs, phase='pre-train'):
        """Update lambda_ic, lambda_bc, and lambda_r every 10 iterations."""
        # Compute gradients
        grad_L_ic = grad(self.compute_L_ic)(params)
        grad_L_bc = grad(self.compute_L_bc)(params)
        if phase=='pre-train':
            grad_L_int = grad(lambda p: self.pre_train_compute_loss_opt_utheta_int(p))(params)
        elif phase=='train':
            grad_L_int = grad(lambda p: self.compute_loss_opt_utheta_int(p, p_coeffs))(params)

        # Compute L2 norms
        norm_ic = pytree_l2_norm(grad_L_ic)
        norm_bc = pytree_l2_norm(grad_L_bc)
        norm_r = pytree_l2_norm(grad_L_int)

        # Sum of norms with epsilon to avoid division by zero
        epsilon = 1e-7
        sum_norm = norm_ic + norm_bc + norm_r + 3*epsilon

        # Compute hat_lambda values
        hat_lambda_ic = jnp.clip(sum_norm / (norm_ic + epsilon), a_min=1, a_max=1e7)
        hat_lambda_bc = jnp.clip(sum_norm / (norm_bc + epsilon), a_min=1, a_max=1e7)
        hat_lambda_r = jnp.clip(sum_norm / (norm_r + epsilon), a_min=1, a_max=1e7)

        # Update weights using moving average
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * hat_lambda_ic
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * hat_lambda_bc
        self.lambda_r = self.alpha * self.lambda_r + (1 - self.alpha) * hat_lambda_r


    def train_step(self, iter, moving_avg=True):
        if iter % self.f_adaptive == 0:
            self.update_adaptive_weights(self.params, self.pwp.coeffs, phase='train')
            print(f"Iteration {iter}, lambda_ic: {self.lambda_ic:.8f}, lambda_bc: {self.lambda_bc:.8f}, lambda_r: {self.lambda_r:.8f}.")
        
        # First, we maximize k_h.
        if iter % 10 == 0:
            print(f"Iteration {iter}, before maximize k_h, loss: {self.compute_loss_opt_utheta_int(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_opt_utheta_bc(self.params):.8f}.")
        hat_coeffs, self.key = self.maximize_kh(self.params, self.key)
        # Moving average for self.pwp.coeffs
        self.pwp.coeffs = self.alpha_p_coeffs * self.pwp.coeffs + (1 - self.alpha_p_coeffs) * hat_coeffs
        if iter % 10 == 0:
            print(f"Iteration {iter}, after maximize k_h, loss: {self.compute_loss_opt_utheta_int(self.params, self.pwp.coeffs):.8f}, {self.compute_loss_opt_utheta_bc(self.params):.8f}.")

    
        # Then we minimize u_theta.
        (loss_int, loss_ibc), self.params, self.opt_state_theta = self.minimize_u(self.params, self.pwp.coeffs, self.opt_state_theta, 
                                                                                 self.lambda_ic, self.lambda_bc, self.lambda_r)
        # return self.compute_loss_opt_utheta(self.params)
        return loss_int, loss_ibc

    def train_process(self, train_num_iterations, save_loss=True, save_model=True):
        print("\nStarting Train Stage")
        start_time = time.perf_counter()
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0
        self.lambda_r = 1.0
        self.loss_int_history = []
        self.loss_bc_history = []
        # Then we maximize k_h
        self.maximize_kh = jit(self.maximize_kh_vec)
        self.minimize_u = jit(self.minimize_u_theta_adaptive)
        # self.maximize_kh = self.maximize_kh_vec
        self.pwp.coeffs, self.key = self.maximize_kh(self.params, self.key)
        for iter in range(train_num_iterations):
            loss_int, loss_bc = self.train_step(iter)
            self.loss_int_history.append(loss_int)
            self.loss_bc_history.append(loss_bc)
            if iter % 100 == 0:
                print(f"Train Iteration {iter}, Loss_int: {loss_int:.8f}, Loss_bc: {loss_bc:.8f}")
        end_time = time.perf_counter()
        train_total_time = (end_time - start_time)
        print(f"Train total time: {train_total_time:.8f} seconds")
        print(f"Train average time per iteration: {train_total_time / train_num_iterations:.9e} seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_loss:
            # Save training loss histories with consistent naming
            loss_int_path = os.path.join('..', 'data', f'loss_int_history_{timestamp}.pkl')
            with open(loss_int_path, 'wb') as f:
                pickle.dump(self.loss_int_history, f)
            loss_bc_path = os.path.join('..', 'data', f'loss_bc_history_{timestamp}.pkl')
            with open(loss_bc_path, 'wb') as f:
                pickle.dump(self.loss_bc_history, f)
            print(f"Loss histories saved to {loss_int_path} and {loss_bc_path}")
        
        if save_model:
            # Save both params and architecture info
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
        """
        Load the saved model parameters and recreate the architecture.

        Args:
            model_path: Path to the saved model file (e.g., '../data/train_model_20231015_123456.pkl').

        Returns:
            None (updates self.params and self.model)
        """
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Extract parameters and architecture info
        loaded_params = loaded_data['params']
        loaded_hidden_dims = loaded_data['hidden_dims']

        # Recreate the model with the same architecture
        self.model = MLP(hidden_dims=hidden_dims, activation=nn.tanh, lb=self.lb, ub=self.ub)
        
        # # Initialize with a dummy key to set up the structure (params will be overwritten)
        # key = jax.random.PRNGKey(0)  # Any seed; params will be replaced
        # dummy_input = jnp.ones((1, 2))
        # self.params = self.model.init(key, dummy_input)

        # Assign the loaded parameters
        self.params = loaded_params
        self.model.params = loaded_params
        print(f"Model loaded from {model_path} with hidden_dims: {loaded_hidden_dims}")

        # Re-initialize optimizer state with loaded params
        self.opt_theta = optax.adam(self.lr_theta)
        self.opt_state_theta = self.opt_theta.init(self.params)
        
        self.pwp.coeffs, self.key = self.maximize_kh_vec(self.params, self.key)
        loss_int, loss_bc = self.compute_loss_opt_utheta_int(self.params, self.pwp.coeffs), self.compute_loss_opt_utheta_bc(self.params)
        print(f"For the loaded model, Loss_int: {loss_int:.8f}, Loss_bc: {loss_bc:.8f}.")
