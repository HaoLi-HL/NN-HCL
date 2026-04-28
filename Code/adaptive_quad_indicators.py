import jax.numpy as jnp


def compute_oscillation_indicator(pwp, u_eval, params, max_points=65536):
    """
    Compute eta_osc[e] = max_q u_theta - min_q u_theta on level-1 quadrature.

    The scan is chunked by physical quadrature-point count to limit peak
    memory during adaptive-quadrature updates.
    """
    Nq_level1 = pwp.Nq_by_level[1]
    input_dim = pwp.partitions.shape[-1]
    chunk_size = max(1, int(max_points) // Nq_level1)
    eta_chunks = []

    for start in range(0, pwp.N_elements, chunk_size):
        stop = min(start + chunk_size, pwp.N_elements)
        element_indices = jnp.arange(start, stop, dtype=jnp.int32)
        _, quad_points, _ = pwp.get_quadrature_by_level(1, element_indices)
        u_values = u_eval(params, quad_points.reshape(-1, input_dim)).reshape(stop - start, Nq_level1)
        eta_chunks.append(jnp.max(u_values, axis=1) - jnp.min(u_values, axis=1))

    return jnp.concatenate(eta_chunks, axis=0)


def compute_quadrature_difference_indicator(
    pwp,
    params,
    loss_fn_elem_int_per_elem_quad_vmap,
    max_points=65536,
):
    """
    Compute eta_quad[e] = |Q_high[e] - Q_low[e]|.

    Q_low uses level-0 quadrature and Q_high uses level-1 quadrature for the
    same current k_h coefficients. The computation is chunked to limit peak
    memory during adaptive-quadrature updates.
    """
    Nq_max = max(pwp.Nq_by_level)
    chunk_size = max(1, int(max_points) // Nq_max)
    eta_chunks = []

    for start in range(0, pwp.N_elements, chunk_size):
        stop = min(start + chunk_size, pwp.N_elements)
        element_indices = jnp.arange(start, stop, dtype=jnp.int32)
        p_coeffs_all = pwp.coeffs[element_indices][:, None, :]

        _, quad_points_low, quad_weights_low = pwp.get_quadrature_by_level(0, element_indices)
        q_low, _ = loss_fn_elem_int_per_elem_quad_vmap(
            element_indices,
            p_coeffs_all,
            params,
            quad_points_low,
            quad_weights_low,
        )

        _, quad_points_high, quad_weights_high = pwp.get_quadrature_by_level(1, element_indices)
        q_high, _ = loss_fn_elem_int_per_elem_quad_vmap(
            element_indices,
            p_coeffs_all,
            params,
            quad_points_high,
            quad_weights_high,
        )

        eta_chunks.append(jnp.abs(jnp.squeeze(q_high, axis=-1) - jnp.squeeze(q_low, axis=-1)))

    return jnp.concatenate(eta_chunks, axis=0)
