import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp


class DynParams(NamedTuple):
    lf: jax.Array
    lr: jax.Array
    drag: jax.Array


class CostParams(NamedTuple):
    q_pos: jax.Array
    q_yaw: jax.Array
    q_v: jax.Array
    r_a: jax.Array
    r_delta: jax.Array
    qf_pos: jax.Array
    qf_yaw: jax.Array
    qf_v: jax.Array


class FullParams(NamedTuple):
    dyn: DynParams
    cost: CostParams

# =======================================#
# Nonlinear kinematic bicycle with simple drag.
# State: [x, y, yaw, v], Control: [a, delta]
# lf: distance from vehicle center of gravity (CoG) to the front axle.
# lr: distance from CoG to the rear axle.
# Wheelbase is lf + lr.
# =======================================#
def bicycle_step(
    state: jax.Array,
    control: jax.Array,
    dyn: DynParams,
    dt: float,
    a_max: float,
    steer_max: float,
    v_max: float,
) -> jax.Array:
    x, y, yaw, v = state
    a_cmd, delta_cmd = control

    a     = jnp.clip(a_cmd, -a_max, a_max)
    delta = jnp.clip(delta_cmd, -steer_max, steer_max)

    wb   = dyn.lf + dyn.lr
    beta = jnp.arctan((dyn.lr / wb) * jnp.tan(delta))

    x_next   = x + dt * v * jnp.cos(yaw + beta)
    y_next   = y + dt * v * jnp.sin(yaw + beta)
    yaw_next = yaw + dt * (v / dyn.lr) * jnp.sin(beta)
    v_next   = jnp.clip(v + dt * (a - dyn.drag * v * jnp.abs(v)), 0.0, v_max)

    return jnp.array([x_next, y_next, wrap_angle(yaw_next), v_next])


def rollout_dynamics(
    state0:   jax.Array,
    controls: jax.Array,
    dyn:      DynParams,
    dt:       float,
    a_max:    float,
    steer_max: float,
    v_max:    float,
) -> jax.Array:
    def step_fn(st, u):
        st_next = bicycle_step(st, u, dyn, dt, a_max, steer_max, v_max)
        return st_next, st_next

    _, states_future = jax.lax.scan(step_fn, state0, controls)
    return jnp.concatenate([state0[None, :], states_future], axis=0)

def _positive(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return jax.nn.softplus(x) + eps

def theta_to_params(theta: jax.Array) -> FullParams:
    """
    theta_raw = [lf_raw, lr_raw, drag_raw,
                 q_pos_raw, q_yaw_raw, q_v_raw,
                 r_a_raw, r_delta_raw,
                 qf_pos_raw, qf_yaw_raw, qf_v_raw]
    """
    lf   = _positive(theta[0]) + 0.20
    lr   = _positive(theta[1]) + 0.20
    drag = _positive(theta[2], eps=1e-5)

    cost = CostParams(
        q_pos   =_positive(theta[3]),
        q_yaw   =_positive(theta[4]),
        q_v     =_positive(theta[5]),
        r_a     =_positive(theta[6]),
        r_delta =_positive(theta[7]),
        qf_pos  =_positive(theta[8]),
        qf_yaw  =_positive(theta[9]),
        qf_v    =_positive(theta[10]),
    )
    return FullParams(dyn=DynParams(lf=lf, lr=lr, drag=drag), cost=cost)


def wrap_angle(a: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def make_reference_line(state0: jax.Array, horizon: int, dt: float, v_ref: float) -> jax.Array:
    """
    Local reference with sinusoidal lane centerline:
    x_ref[k] grows forward, y_ref = A sin(freq * x_ref).
    """
    x0 = state0[0]
    ks = jnp.arange(horizon + 1)
    x_ref = x0 + ks * dt * v_ref

    amp      = 1.2
    freq     = 0.25
    y_ref    = amp * jnp.sin(freq * x_ref)
    dy_dx    = amp * freq * jnp.cos(freq * x_ref)
    yaw_ref  = jnp.arctan(dy_dx)

    v_ref_vec = jnp.ones_like(x_ref) * v_ref
    return jnp.stack([x_ref, y_ref, yaw_ref, v_ref_vec], axis=1)

#=======================================#
#  MPC cost function over rollout trajectory
#=======================================#
def mpc_cost(
    u_flat: jax.Array,
    state0: jax.Array,
    params: FullParams,
    dt: float,
    horizon: int,
    a_max: float,
    steer_max: float,
    v_max: float,
    v_ref: float,
) -> jax.Array:
    controls = u_flat.reshape(horizon, 2)
    states   = rollout_dynamics(state0, controls, params.dyn, dt, a_max, steer_max, v_max)
    refs     = make_reference_line(state0, horizon, dt, v_ref)

    x_err    = states[:-1, 0] - refs[:-1, 0]
    y_err    = states[:-1, 1] - refs[:-1, 1]
    yaw_err  = wrap_angle(states[:-1, 2] - refs[:-1, 2])
    v_err    = states[:-1, 3] - refs[:-1, 3]

    pos_cost = params.cost.q_pos * jnp.sum(x_err**2 + y_err**2)
    yaw_cost = params.cost.q_yaw * jnp.sum(yaw_err**2)
    v_cost   = params.cost.q_v * jnp.sum(v_err**2)

    a_cost   = params.cost.r_a * jnp.sum(controls[:, 0] ** 2)
    d_cost   = params.cost.r_delta * jnp.sum(controls[:, 1] ** 2)

    xT_err   = states[-1, 0] - refs[-1, 0]
    yT_err   = states[-1, 1] - refs[-1, 1]
    yawT_err = wrap_angle(states[-1, 2] - refs[-1, 2])
    vT_err   = states[-1, 3] - refs[-1, 3]

    terminal = (
        params.cost.qf_pos * (xT_err**2 + yT_err**2)
        + params.cost.qf_yaw * yawT_err**2
        + params.cost.qf_v * vT_err**2
    )

    return pos_cost + yaw_cost + v_cost + a_cost + d_cost + terminal

#=======================================#
#  MPC solver and learning loop
#=======================================#
@partial(jax.jit, static_argnames=("horizon", "opt_iters"))
def solve_mpc(
    state0: jax.Array,
    params: FullParams,
    dt: float,
    horizon: int = 12,
    opt_iters: int = 40,
    lr: float = 0.08,
    a_max: float = 2.0,
    steer_max: float = 0.5,
    v_max: float = 8.0,
    v_ref: float = 2.5,
):
    u0 = jnp.zeros((horizon, 2))

    def cost_from_u(u_seq: jax.Array) -> jax.Array:
        return mpc_cost(
            u_seq.reshape(-1),
            state0,
            params,
            dt,
            horizon,
            a_max,
            steer_max,
            v_max,
            v_ref,
        )

    grad_fn = jax.grad(cost_from_u)

    def body(_, u_seq):
        g = grad_fn(u_seq)
        u_new = u_seq - lr * g
        u_new = u_new.at[:, 0].set(jnp.clip(u_new[:, 0], -a_max, a_max))
        u_new = u_new.at[:, 1].set(jnp.clip(u_new[:, 1], -steer_max, steer_max))
        return u_new

    u_star = jax.lax.fori_loop(0, opt_iters, body, u0)
    x_star = rollout_dynamics(state0, u_star, params.dyn, dt, a_max, steer_max, v_max)
    j_star = cost_from_u(u_star)
    return u_star, x_star, j_star

#=======================================#
#  Get first action u[0] from MPC
#=======================================#
def first_action_from_theta(
    theta: jax.Array,
    state0: jax.Array,
    dt: float,
    horizon: int,
    opt_iters: int,
    mpc_lr: float,
    a_max: float,
    steer_max: float,
    v_max: float,
    v_ref: float,
) -> jax.Array:
    params = theta_to_params(theta)
    u_star, _, _ = solve_mpc(
        state0=state0,
        params=params,
        dt=dt,
        horizon=horizon,
        opt_iters=opt_iters,
        lr=mpc_lr,
        a_max=a_max,
        steer_max=steer_max,
        v_max=v_max,
        v_ref=v_ref,
    )
    return u_star[0]

#=======================================#
#  Generate "expert trajectory" using MPC with known parameters
#=======================================#

def generate_expert_trajectory(
    theta_true: jax.Array,
    state0: jax.Array,
    steps: int,
    dt: float,
    horizon: int,
    opt_iters: int,
    mpc_lr: float,
    a_max: float,
    steer_max: float,
    v_max: float,
    v_ref: float,
):
    params_true = theta_to_params(theta_true)
    states = []
    actions = []

    state = state0
    for _ in range(steps):
        u_star, _, _ = solve_mpc(
            state0=state,
            params=params_true,
            dt=dt,
            horizon=horizon,
            opt_iters=opt_iters,
            lr=mpc_lr,
            a_max=a_max,
            steer_max=steer_max,
            v_max=v_max,
            v_ref=v_ref,
        )
        u0 = u_star[0]

        states.append(state)
        actions.append(u0)

        state = bicycle_step(state, u0, params_true.dyn, dt, a_max, steer_max, v_max)

    return jnp.stack(states), jnp.stack(actions)


def decode_theta(theta: jax.Array):
    p = theta_to_params(theta)
    return {
        "lf": float(p.dyn.lf),
        "lr": float(p.dyn.lr),
        "drag": float(p.dyn.drag),
        "q_pos": float(p.cost.q_pos),
        "q_yaw": float(p.cost.q_yaw),
        "q_v": float(p.cost.q_v),
        "r_a": float(p.cost.r_a),
        "r_delta": float(p.cost.r_delta),
    }


def save_learning_plots(output_dir: Path, losses, pred_actions, target_actions):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "learning_curves.png"

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

    t = jnp.arange(target_actions.shape[0])

    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

    axs[0].plot(losses, linewidth=2.0, color="tab:blue")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, target_actions[:, 0], label="expert a", linewidth=1.8)
    axs[1].plot(t, pred_actions[:, 0], label="learned a", linewidth=1.8)
    axs[1].set_title("Action Fit: Acceleration")
    axs[1].set_xlabel("Data index")
    axs[1].set_ylabel("a")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    axs[2].plot(t, target_actions[:, 1], label="expert delta", linewidth=1.8)
    axs[2].plot(t, pred_actions[:, 1], label="learned delta", linewidth=1.8)
    axs[2].set_title("Action Fit: Steering")
    axs[2].set_xlabel("Data index")
    axs[2].set_ylabel("delta")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


def save_learning_csv(output_dir: Path, losses):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "learning_loss.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iter,loss\n")
        for i, loss in enumerate(losses):
            f.write(f"{i},{float(loss):.8f}\n")
    return csv_path


def main():
    # MPC/rollout settings
    dt        = 0.1
    horizon   = 5
    mpc_iters = 200
    mpc_lr    = 0.02
    a_max     = 5.0
    steer_max = 1.
    v_max     = 20.0
    v_ref     = 2.5

    

    # Ground-truth parameters used to create "trajectory data"
    theta_true = jnp.array([
        0.20,   # lf_raw
        0.10,   # lr_raw
        -2.20,  # drag_raw
        1.10,   # q_pos_raw
        0.60,   # q_yaw_raw
        0.80,   # q_v_raw
        -0.30,  # r_a_raw
        -0.90,  # r_delta_raw
        1.30,   # qf_pos_raw
        0.90,   # qf_yaw_raw
        0.90,   # qf_v_raw
    ])

    # Initial guess to be learned
    theta = jnp.array([
        -0.30,
        -0.35,
        -1.00,
        0.20,
        0.10,
        0.20,
        -1.20,
        -1.50,
        0.20,
        0.20,
        0.20,
    ])

    state0 = jnp.array([0.0, -1.2, 0.35, 1.0])
    data_steps = 100

    states_data, actions_data = generate_expert_trajectory(
        theta_true=theta_true,
        state0=state0,
        steps=data_steps,
        dt=dt,
        horizon=horizon,
        opt_iters=mpc_iters,
        mpc_lr=mpc_lr,
        a_max=a_max,
        steer_max=steer_max,
        v_max=v_max,
        v_ref=v_ref,
    )

    # Build differentiable imitation loss over trajectory data.
    def imitation_loss(theta_local: jax.Array) -> jax.Array:
        pred_actions = jax.vmap(
            lambda st: first_action_from_theta(
                theta_local,
                st,
                dt,
                horizon,
                mpc_iters,
                mpc_lr,
                a_max,
                steer_max,
                v_max,
                v_ref,
            )
        )(states_data)
        mse = jnp.mean((pred_actions - actions_data) ** 2)
        reg = 1e-4 * jnp.mean(theta_local**2)
        return mse + reg

    loss_and_grad = jax.jit(jax.value_and_grad(imitation_loss))

    # Parameter learning loop
    train_lr = 0.1
    train_iters = 200
    losses = []

    init_loss = float(imitation_loss(theta))
    print("=== Nonlinear Bicycle Differentiable MPC Learning ===")
    print(f"Initial imitation loss: {init_loss:.6f}")

    for i in range(train_iters):
        loss_val, grad_val = loss_and_grad(theta)
        theta = theta - train_lr * grad_val
        losses.append(float(loss_val))

        if i % 10 == 0 or i == train_iters - 1:
            print(f"iter={i:03d} loss={float(loss_val):.6f}")

    final_loss = float(imitation_loss(theta))
    print(f"Final imitation loss: {final_loss:.6f}")

    pred_actions = jax.vmap(
        lambda st: first_action_from_theta(
            theta,
            st,
            dt,
            horizon,
            mpc_iters,
            mpc_lr,
            a_max,
            steer_max,
            v_max,
            v_ref,
        )
    )(states_data)

    true_decoded  = decode_theta(theta_true)
    est_decoded   = decode_theta(theta)

    print("\nTrue vs learned (selected parameters):")
    for key in ["lf", "lr", "drag", "q_pos", "q_yaw", "q_v", "r_a", "r_delta"]:
        print(f"{key:>8s}: true={true_decoded[key]:.4f}, learned={est_decoded[key]:.4f}")

    output_dir = Path(__file__).resolve().parent / "outputs_bicycle"
    loss_csv = save_learning_csv(output_dir, losses)
    plot_path = save_learning_plots(output_dir, losses, pred_actions, actions_data)

    print(f"\nSaved: {loss_csv}")
    if plot_path is None:
        print("Plot not generated: matplotlib is not installed.")
    else:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()