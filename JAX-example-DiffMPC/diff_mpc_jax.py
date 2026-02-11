import jax
import jax.numpy as jnp
import os
from functools import partial
from pathlib import Path
from typing import NamedTuple


class MPCParams(NamedTuple):
    a: jax.Array
    b: jax.Array
    q: jax.Array
    r: jax.Array
    qf: jax.Array


def theta_to_params(theta: jax.Array) -> MPCParams:
    """
    theta = [a, b, q_raw, r_raw, qf_raw]
    q, r, qf are mapped through softplus to keep them positive.
    """
    a, b, q_raw, r_raw, qf_raw = theta
    q = jax.nn.softplus(q_raw) + 1e-6
    r = jax.nn.softplus(r_raw) + 1e-6
    qf = jax.nn.softplus(qf_raw) + 1e-6
    return MPCParams(a=a, b=b, q=q, r=r, qf=qf)


def rollout_dynamics(x0: jax.Array, u: jax.Array, params: MPCParams) -> jax.Array:
    """Rollout for x_{k+1} = a*x_k + b*u_k. Returns x[0..T]."""

    def step(xk, uk):
        x_next = params.a * xk + params.b * uk
        return x_next, x_next

    _, x_future = jax.lax.scan(step, x0, u)  # length T (x1..xT)
    return jnp.concatenate([jnp.array([x0]), x_future], axis=0)


def mpc_cost(u: jax.Array, x0: jax.Array, params: MPCParams) -> jax.Array:
    """Quadratic finite-horizon MPC objective."""
    x = rollout_dynamics(x0, u, params)
    stage_state = params.q * jnp.sum(x[:-1] ** 2)
    stage_control = params.r * jnp.sum(u ** 2)
    terminal = params.qf * (x[-1] ** 2)
    return stage_state + stage_control + terminal


@partial(jax.jit, static_argnames=("horizon", "opt_iters"))
def solve_mpc(
    x0: jax.Array,
    params: MPCParams,
    horizon: int = 20,
    opt_iters: int = 120,
    lr: float = 0.08,
    u_max: float = 1.0,
):
    """
    Solve MPC by unrolling projected gradient descent over the control sequence.
    This unrolled solver is fully differentiable in JAX.
    """
    u0 = jnp.zeros((horizon,))
    cost_grad = jax.grad(mpc_cost)

    def body(_, u):
        g = cost_grad(u, x0, params)
        u_new = jnp.clip(u - lr * g, -u_max, u_max)
        return u_new

    u_star = jax.lax.fori_loop(0, opt_iters, body, u0)
    x_star = rollout_dynamics(x0, u_star, params)
    j_star = mpc_cost(u_star, x0, params)
    return u_star, x_star, j_star


def optimal_u_from_theta(
    theta: jax.Array,
    x0: jax.Array,
    horizon: int = 20,
    opt_iters: int = 120,
    lr: float = 0.08,
    u_max: float = 1.0,
) -> jax.Array:
    params = theta_to_params(theta)
    u_star, _, _ = solve_mpc(
        x0=x0,
        params=params,
        horizon=horizon,
        opt_iters=opt_iters,
        lr=lr,
        u_max=u_max,
    )
    return u_star


def first_action_from_theta(
    theta: jax.Array,
    x0: jax.Array,
    horizon: int = 20,
    opt_iters: int = 120,
    lr: float = 0.08,
    u_max: float = 1.0,
) -> jax.Array:
    return optimal_u_from_theta(theta, x0, horizon, opt_iters, lr, u_max)[0]


def run_closed_loop(
    theta: jax.Array,
    x0: jax.Array,
    sim_steps: int = 40,
    horizon: int = 20,
    opt_iters: int = 120,
    lr: float = 0.05,
    u_max: float = 1.5,
):
    """
    Run closed-loop MPC:
    - solve MPC at each step,
    - apply first action to the true system,
    - compute sensitivity du0*/dtheta at each step.
    """
    params = theta_to_params(theta)
    x = x0

    x_hist = [x0]
    u_hist = []
    cost_hist = []
    du0_dtheta_hist = []

    for _ in range(sim_steps):
        u_star, _, j_star = solve_mpc(
            x0=x,
            params=params,
            horizon=horizon,
            opt_iters=opt_iters,
            lr=lr,
            u_max=u_max,
        )
        u0 = u_star[0]
        du0_dtheta = jax.grad(first_action_from_theta)(
            theta, x, horizon, opt_iters, lr, u_max
        )

        x_next = params.a * x + params.b * u0

        u_hist.append(float(u0))
        cost_hist.append(float(j_star))
        du0_dtheta_hist.append([float(v) for v in du0_dtheta])
        x_hist.append(float(x_next))

        x = x_next

    return x_hist, u_hist, cost_hist, du0_dtheta_hist


def write_csv_log(
    output_dir: Path,
    x_hist,
    u_hist,
    cost_hist,
    du0_dtheta_hist,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "closed_loop_log.csv"
    header = (
        "step,x,u0,cost,du0_da,du0_db,du0_dq_raw,du0_dr_raw,du0_dqf_raw\n"
    )
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for k in range(len(u_hist)):
            grads = du0_dtheta_hist[k]
            row = (
                f"{k},{x_hist[k]:.8f},{u_hist[k]:.8f},{cost_hist[k]:.8f},"
                f"{grads[0]:.8f},{grads[1]:.8f},{grads[2]:.8f},{grads[3]:.8f},{grads[4]:.8f}\n"
            )
            f.write(row)
    return csv_path


def save_plots(output_dir: Path, x_hist, u_hist, du0_dtheta_hist):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "closed_loop_results.png"
    mpl_config_dir = output_dir / ".mplconfig"
    xdg_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps_u = list(range(len(u_hist)))
    steps_x = list(range(len(x_hist)))

    sens_a = [v[0] for v in du0_dtheta_hist]
    sens_b = [v[1] for v in du0_dtheta_hist]
    sens_q = [v[2] for v in du0_dtheta_hist]
    sens_r = [v[3] for v in du0_dtheta_hist]
    sens_qf = [v[4] for v in du0_dtheta_hist]

    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

    axs[0].plot(steps_x, x_hist, color="tab:blue", linewidth=2.0)
    axs[0].set_title("Closed-Loop State Trajectory")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("x")
    axs[0].grid(True, alpha=0.3)

    axs[1].step(steps_u, u_hist, where="post", color="tab:orange", linewidth=2.0)
    axs[1].set_title("Applied Control u0*")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("u")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(steps_u, sens_a, label="du0/da", linewidth=1.8)
    axs[2].plot(steps_u, sens_b, label="du0/db", linewidth=1.8)
    axs[2].plot(steps_u, sens_q, label="du0/dq_raw", linewidth=1.8)
    axs[2].plot(steps_u, sens_r, label="du0/dr_raw", linewidth=1.8)
    axs[2].plot(steps_u, sens_qf, label="du0/dqf_raw", linewidth=1.8)
    axs[2].set_title("Sensitivity of First Action to Parameters")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Sensitivity")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


def main():
    # theta = [a, b, q_raw, r_raw, qf_raw]
    theta0 = jnp.array([0.95, 0.35, 0.5, 0.7, 0.5])
    x0 = jnp.array(2.0)

    horizon = 20
    opt_iters = 120
    lr = 0.05
    u_max = 1.5

    sim_steps = 40
    output_dir = Path(__file__).resolve().parent / "outputs"

    params0 = theta_to_params(theta0)
    u_star0, x_star0, j_star0 = solve_mpc(
        x0=x0, params=params0, horizon=horizon, opt_iters=opt_iters, lr=lr, u_max=u_max
    )
    du0_dtheta0 = jax.grad(first_action_from_theta)(theta0, x0, horizon, opt_iters, lr, u_max)
    du_dtheta0 = jax.jacrev(optimal_u_from_theta)(theta0, x0, horizon, opt_iters, lr, u_max)

    x_hist, u_hist, cost_hist, du0_hist = run_closed_loop(
        theta=theta0,
        x0=x0,
        sim_steps=sim_steps,
        horizon=horizon,
        opt_iters=opt_iters,
        lr=lr,
        u_max=u_max,
    )
    csv_path = write_csv_log(output_dir, x_hist, u_hist, cost_hist, du0_hist)
    png_path = save_plots(output_dir, x_hist, u_hist, du0_hist)

    print("=== Differentiable MPC with JAX ===")
    print(f"Initial state x0: {float(x0):.3f}")
    print(f"Single-shot optimal cost J*: {float(j_star0):.6f}")
    print(f"Single-shot first optimal action u0*: {float(u_star0[0]):.6f}")
    print(f"Single-shot final predicted state x_T*: {float(x_star0[-1]):.6f}")
    print()
    print("Gradient of first action wrt theta [a, b, q_raw, r_raw, qf_raw]:")
    print(du0_dtheta0)
    print()
    print("Jacobian shape of full control sequence wrt theta:")
    print(du_dtheta0.shape)  # (horizon, 5)
    print()
    print(f"Closed-loop steps: {sim_steps}")
    print(f"Final closed-loop state: {x_hist[-1]:.6f}")
    print(f"CSV log: {csv_path}")
    if png_path is None:
        print("Plot not generated: matplotlib is not installed.")
    else:
        print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
