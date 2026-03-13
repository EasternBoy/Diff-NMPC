import json
import importlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from MPCCsolver import MPCConfigDYN
import casadi_outer_sensitivity as cos
import time

def main():
    # In notebook sessions, force reload so newly added class methods are visible.
    log = {'time': [], 'x': [], 'y': [], 'vx': [], 'yaw': [], 'vy': [], 'yaw_rate': [], 'steer_angle': [],
           'acce': [], 'steering_rate':[], 'theta': [], 'thresh':[],
            'BR': [], 'CR': [], 'DR': [], 'BF': [], 'CF': [], 'DF': [], 'CM': [], 'mu_x': [], 'mu_y': [],
            'q_contour_cur': [], 'q_lag_cur': [], 'q_theta_cur': [], 'q_contour_next': [], 'q_lag_next': [], 'q_theta_next': []}
    importlib.reload(cos)
    CasadiOuterSensitivityMPCC_high_VY = cos.CasadiOuterSensitivityMPCC_high_VY
    CasadiOuterSensitivityMPCC_low_VY = cos.CasadiOuterSensitivityMPCC_low_VY
    with open(
        "more_data/scale0.25_TK20_log_Oschersleben_full_Vinit_8.0friction0.8",
        "r",
    ) as f:
        data = json.load(f)
    data_name = 'qt_800_friction0.8'
    cfg = MPCConfigDYN()
    cfg.TK = 20  # Inner MPC horizon

    sens_mpcc_h = CasadiOuterSensitivityMPCC_high_VY(cfg)
    sens_mpcc_l = CasadiOuterSensitivityMPCC_low_VY(cfg)

    X = jnp.array(data["x"])
    Y = jnp.array(data["y"])
    Yaw = jnp.array(data["yaw"])
    Yaw_rate = jnp.array(data["yaw_rate"])
    VX = jnp.array(data["vx"])
    VY = jnp.array(data["vy"])
    STR_angle = jnp.array(data["steer_angle"])
    theta = jnp.array(data["theta"])
    n_samples   = len(X) # Number of samples to run sensitivity
    outer_steps = 100   # Outer rollout horizon (can be > cfg.TK)
    pg_iters    = 10 # Number of projected gradient steps to take on q in each outer iteration
    lr          = 0.1 # Learning rate for projected gradient step on q
    index_start = 5
    # time_pl = jnp.array(data["time"])
    # plt.plot(time_pl, VY)
    # plt.show()
    # return 0
    normal_constant = 1
    print("n_sample:", n_samples)
    for index in range(n_samples-1):
        index += index_start
        start = time.time()
        state = np.array(
            [X[index], Y[index], VX[index], Yaw[index], VY[index], Yaw_rate[index], STR_angle[index]],
            dtype=float,
        )
        theta_in = theta[index]
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
            [data["q_contour"][index]/normal_constant, data["q_lag"][index]/normal_constant, data["q_theta"][index]/normal_constant],
            dtype=float,
        )

        thresh_hold = jnp.absolute(VY[index]) < 1.5

        print("thresh_hold:", thresh_hold)
        # gradient_step_q_closed_loop(self, init_state, theta_in, dyn_param, q, outer_steps, lr=1e-3, iters=1)
        if thresh_hold:
            q_new, loss, grad_q = sens_mpcc_h.gradient_step_q_closed_loop(
                init_state=state,
                theta_in=theta_in,
                dyn_param=dyn_param,
                q=q0,
                outer_steps=outer_steps,
                lr=lr,
                iters=pg_iters,
            )
            q_new = q_new*normal_constant 
        else:
            q_new, loss, grad_q = sens_mpcc_l.gradient_step_q_closed_loop(
                init_state=state,
                theta_in=theta_in,
                dyn_param=dyn_param,
                q=q0,
                outer_steps=outer_steps,
                lr=3.,
                iters=1,
            )
            q_new = q_new*normal_constant 

        print(f"index={index}/{n_samples}")
        print(f"Parameters: BR {data['BR'][index]}, DR {data['DR'][index]}")
        print(f"positions X: {X[index]}; Y: {Y[index]}")
        print(f"  q init: {q0}")
        print(f"  outer loss: {loss:.6f}")
        print(f"  grad q: {grad_q}")
        print(f"  outer_steps: {outer_steps}, inner_TK: {cfg.TK}")
        print(f"  q updated ({pg_iters} iters): {q_new}\n")
        print(f" solving time: {time.time() - start}")

        log['thresh'].append(str(thresh_hold))
        log['time'].append(float(data["time"][index]))
        log['x'].append(float(X[index]))
        log['y'].append(float(Y[index]))
        log['vx'].append(float(VX[index]))
        log['vy'].append(float(VY[index]))
        log['yaw'].append(float(Yaw[index]))
        log['yaw_rate'].append(float(Yaw_rate[index]))
        log['steer_angle'].append(float(STR_angle[index]))
        log['theta'].append(float(data["theta"][index]))

        log['BR'].append(float(data["BR"][index]))
        log['CR'].append(float(data["CR"][index]))
        log['DR'].append(float(data["DR"][index]))
        log['BF'].append(float(data["BF"][index]))
        log['CF'].append(float(data["CF"][index]))
        log['DF'].append(float(data["DF"][index]))
        log['CM'].append(float(data["CM"][index]))

        # 'q_contour_cur': [], 'q_lag_cur': [], 'q_theta_cur': [], 'q_contour_next': [], 'q_lag_next': [], 'q_theta_next': []

        log['q_contour_cur'].append(float(data["q_contour"][index]))
        log['q_lag_cur'].append(float(data["q_lag"][index]))
        log['q_theta_cur'].append(float(data["q_theta"][index]))

        log['q_contour_next'].append(float(q_new[0]))
        log['q_lag_next'].append(float(q_new[1]))
        log['q_theta_next'].append(float(q_new[2]))
   
        with open(f'{data_name}_outer_steps{outer_steps}_pg_iters{pg_iters}_lr{lr}', 'w') as f:
            json.dump(log, f)

if __name__ == "__main__":
    main()
