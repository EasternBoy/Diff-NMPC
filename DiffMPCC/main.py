from DiffMPCC.MPCCsolver import MPCConfigDYN, STMPCCPlannerCasadi

import json
import pandas as pd
import numpy as onp
import jax.numpy as jnp



if __name__ == '__main__':
    map_file     = 'data/rounded_rectangle_waypoints.csv'
    tpamap_name  = 'data/rounded_rectangle_tpamap.csv'
    tpadata_name = 'data/rounded_rectangle_tpadata.json'
    tpamap = onp.loadtxt(tpamap_name, delimiter=';', skiprows=1)

    tpadata = {}
    with open(tpadata_name) as f:
        tpadata = json.load(f)

    raceline   = onp.loadtxt(map_file, delimiter=";", skiprows=3)
    waypoints  = jnp.asarray(raceline)


    with open('data/log_full_Vinit_8.0_c20.0_l3000.0_p100.0_weightslip0.5_thetaslip_100_150_290_310_non', 'r') as f:
        data = json.load(f)
    print(data.keys())
    dyn_config = MPCConfigDYN()
    dyn_config.q_contour = data['q_contour'][0]
    dyn_config.q_lag     = data['q_lag'][0]
    dyn_config.q_theta   = data['q_theta'][0]

    # Set up MPCC solver
    BR = data['BR'][0]
    CR = data['CR'][0]
    DR = data['DR'][0]*(9.81*dyn_config.MASS)/2
    BF = data['BF'][0]
    CF = data['CF'][0]
    DF = data['DF'][0]*(9.81*dyn_config.MASS)/2
    CM = data['CM'][0]
    model_param = jnp.array([BR, CR, DR, BF, CF, DF, CM])

    planner_dyn_mpc = STMPCCPlannerCasadi(waypoints=waypoints, config=dyn_config, param=model_param)

    # Test the planner with all states from the dataset
    X = jnp.array(data['x'])
    Y = jnp.array(data['y'])
    Yaw = jnp.array(data['yaw'])
    Yaw_rate = jnp.array(data['yaw_rate'])
    VX = jnp.array(data['vx'])
    VY = jnp.array(data['vy'])
    STR_angle = jnp.array(data['steer_angle'])


    horizon = 10 # Ensure we have enough data for the horizon
    for index in range(horizon):
        test_state = jnp.array([X[index], Y[index], VX[index], Yaw[index], VY[index], Yaw_rate[index], STR_angle[index]])

        formatted_state = [float(f"{value:.3f}") for value in onp.asarray(test_state)]
        print("Test state at index %d: %s" % (index, formatted_state,))

        planner_dyn_mpc.config.q_contour = data['q_contour'][index]
        planner_dyn_mpc.config.q_lag     = data['q_lag'][index]
        planner_dyn_mpc.config.q_theta   = data['q_theta'][index]
        planner_dyn_mpc.param.BR = data['BR'][index]
        planner_dyn_mpc.param.CR = data['CR'][index]
        planner_dyn_mpc.param.DR = data['DR'][index]*(9.81*dyn_config.MASS)/2
        planner_dyn_mpc.param.BF = data['BF'][index]
        planner_dyn_mpc.param.CF = data['CF'][index]
        planner_dyn_mpc.param.DF = data['DF'][index]*(9.81*dyn_config.MASS)/2
        planner_dyn_mpc.param.CM = data['CM'][index]

        u, _, _, _, _ = planner_dyn_mpc.plan(test_state, model_param)
        u[0] = u[0] / planner_dyn_mpc.config.MASS
        print("Optimal acceleration:", u[0], "--Data acceleration:", data['acce'][index])
        print("Optimal steering speed:", u[1], "--Data steering speed:", data['steering_rate'][index+1], "\n")