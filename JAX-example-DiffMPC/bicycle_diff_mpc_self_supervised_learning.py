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


def wrap_angle(a: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def _positive(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return jax.nn.softplus(x) + eps


def theta_to_cost_params(theta: jax.Array) -> CostParams:
    """
    theta_raw = [q_pos_raw, q_yaw_raw, q_v_raw,
                 r_a_raw, r_delta_raw,
                 qf_pos_raw, qf_yaw_raw, qf_v_raw]
    """
    return CostParams(
        q_pos  =_positive(theta[0]),
        q_yaw  =_positive(theta[1]),
        q_v    =_positive(theta[2]),
        r_a    =_positive(theta[3]),
        r_delta=_positive(theta[4]),
        qf_pos =_positive(theta[5]),
        qf_yaw =_positive(theta[6]),
        qf_v   =_positive(theta[7]),
    )


def known_dynamics_params() -> DynParams:
    # Known bicycle model (fixed, not learned), matched to imitation expert:
    # lf_raw=0.20, lr_raw=0.10, drag_raw=-2.20
    return DynParams(
        lf=_positive(jnp.array(0.20)) + 0.20,
        lr=_positive(jnp.array(0.10)) + 0.20,
        drag=_positive(jnp.array(-2.20), eps=1e-5),
    )


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

    a = jnp.clip(a_cmd, -a_max, a_max)
    delta = jnp.clip(delta_cmd, -steer_max, steer_max)

    wb = dyn.lf + dyn.lr
    beta = jnp.arctan((dyn.lr / wb) * jnp.tan(delta))

    x_next = x + dt * v * jnp.cos(yaw + beta)
    y_next = y + dt * v * jnp.sin(yaw + beta)
    yaw_next = yaw + dt * (v / dyn.lr) * jnp.sin(beta)
    v_next = jnp.clip(v + dt * (a - dyn.drag * v * jnp.abs(v)), 0.0, v_max)

    return jnp.array([x_next, y_next, wrap_angle(yaw_next), v_next])


def rollout_dynamics(
    state0: jax.Array,
    controls: jax.Array,
    dyn: DynParams,
    dt: float,
    a_max: float,
    steer_max: float,
    v_max: float,
) -> jax.Array:
    def step_fn(st, u):
        st_next = bicycle_step(st, u, dyn, dt, a_max, steer_max, v_max)
        return st_next, st_next

    _, states_future = jax.lax.scan(step_fn, state0, controls)
    return jnp.concatenate([state0[None, :], states_future], axis=0)


def make_reference_line(state0: jax.Array, horizon: int, dt: float, v_ref: float) -> jax.Array:
    x0 = state0[0]
    ks = jnp.arange(horizon + 1)
    x_ref = x0 + ks * dt * v_ref

    amp = 1.2
    freq = 0.25
    y_ref = amp * jnp.sin(freq * x_ref)
    dy_dx = amp * freq * jnp.cos(freq * x_ref)
    yaw_ref = jnp.arctan(dy_dx)
    v_ref_vec = jnp.ones_like(x_ref) * v_ref
    return jnp.stack([x_ref, y_ref, yaw_ref, v_ref_vec], axis=1)


def mpc_inner_objective(
    u_flat: jax.Array,
    state0: jax.Array,
    dyn: DynParams,
    cost: CostParams,
    dt: float,
    horizon: int,
    a_max: float,
    steer_max: float,
    v_max: float,
    v_ref: float,
) -> jax.Array:
    # Inner objective: optimized by MPC over control sequence u.
    controls = u_flat.reshape(horizon, 2)
    states = rollout_dynamics(state0, controls, dyn, dt, a_max, steer_max, v_max)
    refs = make_reference_line(state0, horizon, dt, v_ref)

    x_err = states[:-1, 0] - refs[:-1, 0]
    y_err = states[:-1, 1] - refs[:-1, 1]
    yaw_err = wrap_angle(states[:-1, 2] - refs[:-1, 2])
    v_err = states[:-1, 3] - refs[:-1, 3]

    stage = (
        cost.q_pos * jnp.sum(x_err**2 + y_err**2)
        + cost.q_yaw * jnp.sum(yaw_err**2)
        + cost.q_v * jnp.sum(v_err**2)
        + cost.r_a * jnp.sum(controls[:, 0] ** 2)
        + cost.r_delta * jnp.sum(controls[:, 1] ** 2)
    )

    xT_err = states[-1, 0] - refs[-1, 0]
    yT_err = states[-1, 1] - refs[-1, 1]
    yawT_err = wrap_angle(states[-1, 2] - refs[-1, 2])
    vT_err = states[-1, 3] - refs[-1, 3]
    terminal = (
        cost.qf_pos * (xT_err**2 + yT_err**2)
        + cost.qf_yaw * yawT_err**2
        + cost.qf_v * vT_err**2
    )
    return stage + terminal


@partial(jax.jit, static_argnames=("horizon", "opt_iters"))
def solve_mpc(
    state0: jax.Array,
    dyn: DynParams,
    cost: CostParams,
    dt: float,
    horizon: int = 12,
    opt_iters: int = 100,
    lr: float = 0.1,
    a_max: float = 2.0,
    steer_max: float = 0.5,
    v_max: float = 8.0,
    v_ref: float = 2.5,
):
    u0 = jnp.zeros((horizon, 2))

    def cost_from_u(u_seq: jax.Array) -> jax.Array:
        return mpc_inner_objective(
            u_seq.reshape(-1),
            state0,
            dyn,
            cost,
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
    return u_star


def rollout_closed_loop(
    theta_cost: jax.Array,
    state0: jax.Array,
    dyn: DynParams,
    sim_steps: int,
    dt: float,
    horizon: int,
    opt_iters: int,
    mpc_lr: float,
    a_max: float,
    steer_max: float,
    v_max: float,
    v_ref: float,
):
    cost = theta_to_cost_params(theta_cost)

    def one_step(state, _):
        u_seq = solve_mpc(
            state0=state,
            dyn=dyn,
            cost=cost,
            dt=dt,
            horizon=horizon,
            opt_iters=opt_iters,
            lr=mpc_lr,
            a_max=a_max,
            steer_max=steer_max,
            v_max=v_max,
            v_ref=v_ref,
        )
        u0 = u_seq[0]
        state_next = bicycle_step(state, u0, dyn, dt, a_max, steer_max, v_max)
        return state_next, (state, u0)

    state_last, (states_pre, actions) = jax.lax.scan(
        one_step,
        state0,
        xs=None,
        length=sim_steps,
    )
    states = jnp.concatenate([states_pre, state_last[None, :]], axis=0)
    return states, actions


def outer_parameter_objective(states: jax.Array, actions: jax.Array, dt: float, v_ref: float) -> jax.Array:
    # Outer objective: used to optimize the MPC objective parameters (theta_cost).
    refs = make_reference_line(states[0], states.shape[0] - 1, dt=dt, v_ref=v_ref)

    x_err   = states[:-1, 0] - refs[:-1, 0]
    y_err   = states[:-1, 1] - refs[:-1, 1]
    yaw_err = wrap_angle(states[:-1, 2] - refs[:-1, 2])
    v_err   = states[:-1, 3] - refs[:-1, 3]

    # Fixed task objective; intentionally different from the inner MPC objective.
    stage = (
        jnp.sum(x_err**2 + y_err**2)
        # +  100 * jnp.sum(yaw_err**2)
        # + 1.0 * jnp.sum(v_err**2)
        # + 0.08 * jnp.sum(actions[:, 0] ** 2)
        # + 0.20 * jnp.sum(actions[:, 1] ** 2)
    )

    # xT_err = states[-1, 0] - refs[-1, 0]
    # yT_err = states[-1, 1] - refs[-1, 1]
    # yawT_err = wrap_angle(states[-1, 2] - refs[-1, 2])
    # vT_err = states[-1, 3] - refs[-1, 3]

    # terminal = 7.0 * (xT_err**2 + yT_err**2) + 3.0 * yawT_err**2 + 2.0 * vT_err**2
    # return stage + terminal
    return stage


def decode_cost(theta: jax.Array):
    p = theta_to_cost_params(theta)
    return {
        "q_pos": float(p.q_pos),
        "q_yaw": float(p.q_yaw),
        "q_v": float(p.q_v),
        "r_a": float(p.r_a),
        "r_delta": float(p.r_delta),
        "qf_pos": float(p.qf_pos),
        "qf_yaw": float(p.qf_yaw),
        "qf_v": float(p.qf_v),
    }


def save_learning_csv(output_dir: Path, losses):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "self_supervised_loss.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iter,loss\n")
        for i, loss in enumerate(losses):
            f.write(f"{i},{float(loss):.8f}\n")
    return csv_path


def save_plots(output_dir: Path, losses, states_init, states_final, actions_final, dt: float, v_ref: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "self_supervised_learning.png"

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

    ref = make_reference_line(states_final[0], states_final.shape[0] - 1, dt=dt, v_ref=v_ref)
    t_u = jnp.arange(actions_final.shape[0])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(losses, color="tab:blue", linewidth=2.0)
    axs[0].set_title("Self-Supervised Training Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Objective")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(ref[:, 0], ref[:, 1], "--", label="reference", linewidth=1.8)
    axs[1].plot(states_init[:, 0], states_init[:, 1], label="initial MPC weights", linewidth=1.8)
    axs[1].plot(states_final[:, 0], states_final[:, 1], label="learned MPC weights", linewidth=2.0)
    axs[1].set_title("Closed-Loop Trajectory")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_u, actions_final[:, 0], label="a", linewidth=1.8)
    axs[2].plot(t_u, actions_final[:, 1], label="delta", linewidth=1.8)
    axs[2].set_title("Learned Policy First Actions")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Control")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    # MPC and rollout settings
    dt        = 0.1
    horizon   = 10
    mpc_iters = 200
    mpc_lr    = 0.02
    sim_steps = 50
    a_max     = 5.0
    steer_max = 1.
    v_max     = 20.0
    v_ref     = 2.5

    dyn = known_dynamics_params()
    state0 = jnp.array([0.0, -5.0, 0.0, 0.0])

    # Optimize only MPC weights (dynamics are known/fixed).
    # theta = jnp.array([
    #     0.20,    # q_pos_raw
    #     0.10,    # q_yaw_raw
    #     0.20,    # q_v_raw
    #     -1.20,   # r_a_raw
    #     -1.50,   # r_delta_raw
    #     0.20,    # qf_pos_raw
    #     0.20,    # qf_yaw_raw
    #     0.20,    # qf_v_raw
    # ])
    theta = jax.random.uniform(jax.random.PRNGKey(0), shape=(8,), minval=0, maxval=2)
    theta_init = theta

    def self_supervised_loss(theta_local: jax.Array) -> jax.Array:
        # Bilevel setup:
        # 1) Inner solve_mpc minimizes mpc_inner_objective w.r.t. controls.
        # 2) Outer loss minimizes outer_parameter_objective w.r.t. theta_local.
        states, actions = rollout_closed_loop(
            theta_cost= theta_local,
            state0    = state0,
            dyn       = dyn,
            sim_steps = sim_steps,
            dt        = dt,
            horizon   = horizon,
            opt_iters = mpc_iters,
            mpc_lr    = mpc_lr,
            a_max     = a_max,
            steer_max = steer_max,
            v_max     = v_max,
            v_ref     = v_ref,
        )
        task_cost = outer_parameter_objective(states, actions, dt=dt, v_ref=v_ref)
        reg = 1e-4 * jnp.mean(theta_local**2)
        return task_cost + reg

    init_loss = float(self_supervised_loss(theta_init))
    print("=== Bicycle MPC Self-Supervised Learning (Known Dynamics) ===")
    print(f"Initial rollout objective: {init_loss:.6f}")


    loss_and_grad = jax.jit(jax.value_and_grad(self_supervised_loss))
    train_lr    = 1e-3
    train_iters = 100
    losses      = []

    for i in range(train_iters):
        loss_val, grad_val = loss_and_grad(theta)
        theta = theta - train_lr * grad_val
        theta = jnp.clip(theta, 0., 3.)  # Avoid extreme values for stability.
        losses.append(float(loss_val))
        if i % 10 == 0 or i == train_iters - 1:
            print(f"iter={i:03d} loss={float(loss_val):.6f}")

    final_loss = float(self_supervised_loss(theta))
    print(f"Final rollout objective: {final_loss:.6f}")

    states_init, _ = rollout_closed_loop(
        theta_cost=theta_init,
        state0=state0,
        dyn=dyn,
        sim_steps=sim_steps,
        dt=dt,
        horizon=horizon,
        opt_iters=mpc_iters,
        mpc_lr=mpc_lr,
        a_max=a_max,
        steer_max=steer_max,
        v_max=v_max,
        v_ref=v_ref,
    )
    states_final, actions_final = rollout_closed_loop(
        theta_cost=theta,
        state0=state0,
        dyn=dyn,
        sim_steps=sim_steps,
        dt=dt,
        horizon=horizon,
        opt_iters=mpc_iters,
        mpc_lr=mpc_lr,
        a_max=a_max,
        steer_max=steer_max,
        v_max=v_max,
        v_ref=v_ref,
    )

    decoded      = decode_cost(theta)
    decoded_init = decode_cost(theta_init)
    print("\nLearned MPC weights:")
    for key in ["q_pos", "q_yaw", "q_v", "r_a", "r_delta", "qf_pos", "qf_yaw", "qf_v"]:
        # print(f"{key:>8s}: {decoded[key]:.4f}")
        print(f"{key:>8s}: true={decoded_init[key]:.4f}, learned={decoded[key]:.4f}")


    output_dir = Path(__file__).resolve().parent / "outputs_bicycle_selfsup"
    loss_csv = save_learning_csv(output_dir, losses)
    plot_path = save_plots(
        output_dir,
        losses,
        states_init,
        states_final,
        actions_final,
        dt=dt,
        v_ref=v_ref,
    )

    print(f"\nSaved: {loss_csv}")
    if plot_path is None:
        print("Plot not generated: matplotlib is not installed.")
    else:
        print(f"Saved: {plot_path}")

