# JAX Differentiable MPC Example

This example shows a simple differentiable MPC setup for a 1D linear system:

`x_{k+1} = a x_k + b u_k`

The MPC problem is solved with unrolled projected gradient descent in JAX, so the
solution is differentiable with respect to model and cost parameters.

## What it demonstrates

- Solving finite-horizon MPC with JAX
- Computing `d u_0^* / d theta` using `jax.grad`
- Computing `d u^* / d theta` (full Jacobian) using `jax.jacrev`

`theta = [a, b, q_raw, r_raw, qf_raw]`

where `q`, `r`, and `qf` are obtained via softplus to enforce positivity.

## Run

```bash
cd JAX-example-DiffMPC
python diff_mpc_jax.py
```

## Output

After running, the script writes:

- `outputs/closed_loop_log.csv`
- `outputs/closed_loop_results.png` (if `matplotlib` is installed)

The CSV contains per-step closed-loop values:

- state `x`
- applied MPC action `u0`
- MPC objective value at each step
- sensitivity `du0/dtheta` for each parameter

## Notes

- This example differentiates through the *unrolled optimizer*.
- For larger MPCs, you can replace the inner optimizer with more advanced methods
  (e.g., custom solvers or implicit differentiation).
