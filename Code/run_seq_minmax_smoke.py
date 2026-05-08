#!/usr/bin/env python3
"""Small smoke test for SeqMinMaxOptProb on a 3D hex mesh."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "seq_minmax_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import jax.numpy as jnp

import seq_minmax_opt_class as seq_module


def build_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight SeqMinMaxOptProb smoke test."
    )
    parser.add_argument("--n-t", type=int, default=1)
    parser.add_argument("--n-x", type=int, default=4)
    parser.add_argument("--n-y", type=int, default=2)
    parser.add_argument("--n-bc", type=int, default=4)
    parser.add_argument("--n-ic", type=int, default=4)
    parser.add_argument("--n-perturb", type=int, default=2)
    parser.add_argument("--kh-batch-size", type=int, default=10000)
    parser.add_argument("--hidden-width", type=int, default=4)
    parser.add_argument("--grade-epochs", type=int, default=1)
    parser.add_argument("--k-h-rf-times", type=int, default=1)
    parser.add_argument("--stage2-k", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--adaptive-quad-top-fraction", type=float, default=None)
    parser.add_argument("--adaptive-quad-top-count", type=int, default=None)
    parser.add_argument("--adapt-start", type=float, default=None)
    parser.add_argument(
        "--adaptive-quad-indicator",
        choices=("osc", "grad", "gradient", "quad", "quad_diff"),
        default="grad",
    )
    parser.add_argument("--enable-adaptive-quadrature", action="store_true")
    parser.add_argument("--disable-adaptive-quadrature", action="store_true")
    parser.add_argument("--importance-sampling", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def make_problem(args):
    u0 = lambda x, y: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    zero_tx = lambda t, z: jnp.zeros_like(t + z)
    flux = lambda u: (0.5 * u**2, 0.5 * u**2)
    res = lambda u, u_t, u_x, u_y: u_t + u * u_x + u * u_y

    if not args.plot:
        seq_module.plot_u_theta = lambda *unused_args, **unused_kwargs: None

    return seq_module.SeqMinMaxOptProb(
        input_dim=3,
        num_segments=1,
        total_t_min=0.0,
        total_t_max=0.1,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        N_t=args.n_t,
        N_x=args.n_x,
        N_y=args.n_y,
        N_bc=args.n_bc,
        N_ic=args.n_ic,
        hidden_dims_list=[(args.hidden_width,)],
        seed=0,
        sigma=0.1,
        N_perturb=args.n_perturb,
        kh_batch_size=args.kh_batch_size,
        k_h_rf_times=args.k_h_rf_times,
        f_opt_kh=1,
        u0=u0,
        u1=zero_tx,
        u2=zero_tx,
        u3=zero_tx,
        u4=zero_tx,
        flux=flux,
        res=res,
        gamma=1e-4,
        adjust_gamma=False,
        init_lrs=[1e-3],
        grade_epochs=[args.grade_epochs],
        decay_rates=[1e-3],
        init_lr_stage2=1e-3,
        epoch_stage2=args.stage2_epochs,
        loss_type="interior",
        partition="hex",
        dg_type="Q1",
        k=args.stage2_k,
        f_plot=0,
        M_t=4,
        M_x=4,
        set_wandb=False,
        use_importance_sampling=args.importance_sampling,
        use_adaptive_quadrature=args.enable_adaptive_quadrature and not args.disable_adaptive_quadrature,
        adaptive_quad_top_fraction=args.adaptive_quad_top_fraction,
        adaptive_quad_top_count=args.adaptive_quad_top_count,
        adaptive_quad_indicator=args.adaptive_quad_indicator,
        adapt_start=args.adapt_start,
    )


def assert_smoke_result(seq):
    if len(seq.segments) != 1:
        raise RuntimeError(f"Expected one trained segment, got {len(seq.segments)}.")

    segment = seq.segments[0]
    if not segment.loss_eqn_history or not segment.loss_eqn_history[0]:
        raise RuntimeError("No equation-loss history was recorded.")

    final_loss = segment.loss_eqn_history[0][-1]
    if not bool(jnp.isfinite(final_loss)):
        raise RuntimeError(f"Final equation loss is not finite: {final_loss}.")

    print(f"final equation loss: {float(final_loss):.6e}")
    print(f"adaptive_quad_top_fraction stored on seq: {seq.adaptive_quad_top_fraction}")
    print(f"adaptive_quad_top_fraction stored on segment: {segment.adaptive_quad_top_fraction}")
    print(f"adaptive_quad_top_count stored on seq: {seq.adaptive_quad_top_count}")
    print(f"adaptive_quad_top_count stored on segment: {segment.adaptive_quad_top_count}")
    print(f"kh_batch_size stored on seq: {seq.kh_batch_size}")
    print(f"kh_batch_size stored on segment: {segment.kh_batch_size}")
    print(f"adapt_start stored on seq: {seq.adapt_start}")
    print(f"adapt_start stored on segment: {segment.adapt_start}")
    print(f"adaptive_quadrature_requested on seq: {seq.adaptive_quadrature_requested}")
    print(f"adaptive_quadrature_requested on segment: {segment.adaptive_quadrature_requested}")
    print(f"adaptive_quadrature_supported: {segment.adaptive_quadrature_supported}")
    print(f"adaptive_quadrature_enabled: {segment.adaptive_quadrature_enabled}")

    if segment.adaptive_quadrature_enabled:
        if segment.adaptive_quad_top_count is None:
            fraction = segment.adaptive_quad_top_fraction
            if fraction is None:
                fraction = 2.0 / segment.N_x
            expected = int(jnp.ceil(fraction * segment.pwp.N_elements))
        else:
            expected = max(0, min(segment.pwp.N_elements, int(segment.adaptive_quad_top_count)))
        actual = int(segment.pwp.level1_element_indices.shape[0])
        print(f"level-1 quadrature elements: actual={actual}, expected={expected}")
        if actual != expected:
            raise RuntimeError(
                f"Unexpected number of refined elements: actual={actual}, expected={expected}."
            )


def main():
    args = build_args()
    seq = make_problem(args)
    seq.train_seq(
        sqgment_start=0,
        sqgment_end=1,
        grade_epochs=[args.grade_epochs],
        epoch_stage2=args.stage2_epochs,
        save_model=False,
    )
    assert_smoke_result(seq)
    print("SeqMinMaxOptProb smoke test passed.")


if __name__ == "__main__":
    main()
