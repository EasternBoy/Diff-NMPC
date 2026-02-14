import json
import numpy as np
import jax.numpy as jnp

from DiffMPCC.MPCCsolver import MPCConfigDYN
from DiffMPCC.casadi_outer_sensitivity import CasadiOuterSensitivityMPCC


def main():
    with open(
        "data/log_full_Vinit_8.0_c20.0_l3000.0_p100.0_weightslip0.5_thetaslip_100_150_290_310_non",
        "r",
    ) as f:
        data = json.load(f)

    cfg = MPCConfigDYN()
    cfg.TK = 20

    sens_mpcc = CasadiOuterSensitivityMPCC(cfg)

    X = jnp.array(data["x"])
    Y = jnp.array(data["y"])
    Yaw = jnp.array(data["yaw"])
    Yaw_rate = jnp.array(data["yaw_rate"])
    VX = jnp.array(data["vx"])
    VY = jnp.array(data["vy"])
    STR_angle = jnp.array(data["steer_angle"])

    horizon = 20
    pg_iters = 10
    lr = 1e-1

    for index in range(horizon):
        state = np.array(
            [X[index], Y[index], VX[index], Yaw[index], VY[index], Yaw_rate[index], STR_angle[index]],
            dtype=float,
        )

        dyn_param = np.array(
            [
                data["BR"][index],
                data["CR"][index],
                data["DR"][index] * (9.81 * cfg.MASS) / 2.0,
                data["BF"][index],
                data["CF"][index],
                data["DF"][index] * (9.81 * cfg.MASS) / 2.0,
                data["CM"][index],
            ],
            dtype=float,
        )

        q0 = np.array(
            [data["q_contour"][index], data["q_lag"][index], data["q_theta"][index]],
            dtype=float,
        )

        q_new, loss, grad_q = sens_mpcc.gradient_step_q(
            init_state=state,
            dyn_param=dyn_param,
            q=q0,
            lr=lr,
            iters=pg_iters,
        )

        print(f"index={index}")
        print(f"  q init: {q0}")
        print(f"  outer loss: {loss:.6f}")
        print(f"  grad q: {grad_q}")
        print(f"  q updated ({pg_iters} iters): {q_new}\n")


if __name__ == "__main__":
    main()
