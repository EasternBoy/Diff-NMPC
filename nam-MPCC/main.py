from dataclasses import dataclass, field
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline 
import pandas as pd
from argparse import Namespace
import json
import pandas as pd
from scipy.spatial import KDTree
import jax.numpy as jnp
import time

data = pd.read_csv("data/waypoints.csv")
df = data[["theta", "X", "Y"]].copy()
df = df.sort_values("theta")
df = df.groupby("theta", as_index=False).mean()

theta = df["theta"].to_numpy()
x = df["X"].to_numpy()
y = df["Y"].to_numpy()

spline_x = CubicSpline(theta, x, bc_type='natural')
spline_y = CubicSpline(theta, y, bc_type='natural')

theta_min = float(theta.min())
theta_max = float(theta.max())


class ThetaLookupTable:
    def __init__(self, spline_x, spline_y, theta_min, theta_max, n_samples=50000):
        # Dense sampling of the path
        self.theta_samples = np.linspace(theta_min, theta_max, n_samples)
        self.x_samples = spline_x(self.theta_samples)
        self.y_samples = spline_y(self.theta_samples)
        
        # Build KD-tree for fast nearest neighbor search
        self.positions = np.column_stack([self.x_samples, self.y_samples])
        self.kdtree = KDTree(self.positions)
    
    def query(self, x_query, y_query, k_neighbors=1):
        query_point = np.array([[x_query, y_query]])
        if k_neighbors == 1:
            # Simple nearest neighbor
            dist, idx = self.kdtree.query(query_point)
            return float(self.theta_samples[idx[0]])
        else:
            distances, indices = self.kdtree.query(query_point, k=k_neighbors)
            weights = 1.0 / (distances[0] + 1e-10)
            weights /= weights.sum()
            theta_weighted = np.sum(self.theta_samples[indices[0]] * weights)
            return float(theta_weighted)
    
    def query_batch(self, x_queries, y_queries, k_neighbors=1):

        query_points = np.column_stack([x_queries, y_queries])
        
        if k_neighbors == 1:
            distances, indices = self.kdtree.query(query_points)
            return self.theta_samples[indices].astype(float)
        else:
            distances, indices = self.kdtree.query(query_points, k=k_neighbors)
            # Weighted average for each query point
            thetas = []
            for i in range(len(query_points)):
                weights = 1.0 / (distances[i] + 1e-10)
                weights /= weights.sum()
                theta_weighted = np.sum(self.theta_samples[indices[i]] * weights)
                thetas.append(theta_weighted)
            return np.array(thetas)
find_theta = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=1000000)   

def lookup_xy(theta_query):
    return float(spline_x(theta_query)), float(spline_y(theta_query))

def lookup_phi(theta_query):
    dx = spline_x(theta_query, 1)
    dy = spline_y(theta_query, 1)
    return float(np.arctan2(dy, dx))
    
    
@dataclass
class MPCConfigDYN:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2   # length of input vector: u = = [acceleration, steering speed]
    TK: int = 30  # finite time horizon length kinematic
    '''
    Parameters for MPCC objective
    '''
    Rk_ca: list = field(
        default_factory=lambda: np.diag([0.0005, 2.0, 0.01]))  # input cost matrix, penalty for inputs - [accel, steering_speed, vi_speed]
    Rdk_ca: list = field(
        default_factory=lambda: np.diag([0.01, 2.0, 0.01]))  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed, vi_speed]

    q_contour: float = 50.0
    q_lag: float     = 3000.0
    q_theta: float   = 18.0

    '''
    Learning parameters for predictive model
    '''
    num_param: int = 7
    '''
    Model's parameters
    '''
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]  
    MAX_THETA: float = np.inf  # maximum a virtual theta for MPCC
    MIN_THETA: float = 0.0  # minimum a virtual theta for MPCC
    MAX_VI: float = 50.0 # maximum a virtual control input vi for MPCC
    MIN_VI: float = 0.0 # minimum a virtual control input vi for MPCC
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum ste
    MAX_ACCEL: float = 50.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]ering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 50.0  # maximum speed [m/s]
    MIN_SPEED: float = 2.0  # minimum backward speed [m/s]
    MIN_POS_X: float = -np.inf  # minimum horizontal direction (x) 
    MAX_POS_X: float =  np.inf  # maximum horizontal direction (x) 
    MIN_POS_Y: float = -np.inf  # minimum vertical direction (y) 
    MAX_POS_Y: float =  np.inf  # maximum vertical direction (y) 
    MIN_SPEED_LAT: float = -np.inf  # minimum latteral speed (m/s)
    MAX_SPEED_LAT: float =  np.inf # maximum latteral speed (m/s)

    # model parameters
    MASS: float = 1225.887  # Vehicle mass
    I_Z: float  = 1560.3729  # Vehicle inertia
    TORQUE_SPLIT: float = 0.0  # Torque distribution

    # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
    CR0: float = 2.3451
    CR2: float = -0.0095


class STMPCCPlannerCasadi:
    def __init__(self, config, waypoints=None, x0=None, param=None):
        self.waypoints = waypoints
        self.config = config    
        self.look_theta = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=1000000)
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.track_length = float(theta_max - theta_min)
        self.theta_index = self.config.NXK - 1
        
        self.input_o = np.zeros(self.config.NU) * np.nan  # NU = 2 (accel, steering)
        self.states_output = np.ones((self.config.NXK, self.config.TK + 1)) * np.nan
        
        self.q_contour = self.config.q_contour      # Contouring error weight
        self.q_lag     = self.config.q_lag              # Lag error weight  
        self.q_theta   = self.config.q_theta          # Progress maximization (negative = reward)
        
        self.DTK  = self.config.DTK
        self.MASS = self.config.MASS
        self.I_Z  = self.config.I_Z
        self.LF   = self.config.LF
        self.LR   = self.config.LR
        self.TORQUE_SPLIT = self.config.TORQUE_SPLIT

        self.CR0   = self.config.CR0
        self.CR2   = self.config.CR2
        self.u_his = [0, 0]


        start_point = 1
        if x0 is None:
            self.x0 = np.array([waypoints[start_point, 1], waypoints[start_point, 2], 8.0, 0.0 , 0.0, 0.0, 0.0])
        else:
            self.x0 = x0

        self.theta0 = find_theta.query(self.x0[0], self.x0[1], k_neighbors=30)


        BR = 15.9504
        CR = 1.3754
        DR = 4500.9280
        BF = 9.4246
        CF = 5.9139  
        DF = 4500.8218
        CM = 0.9459
        self.param = np.array([BR, CR, DR, BF, CF, DF, CM], dtype=float)

        self.mpc_prob_init()



    def plan(self, states, param, waypoints=None):

        theta_0 = find_theta.query(states[1], states[2], k_neighbors=30)

        if waypoints is not None:
            self.waypoints = waypoints

        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y = self.MPCC_Control(states, param, theta_0)

        return u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y

    def clip_input(self, u):
        u0 = ca.fmin(
            ca.fmax(u[0], self.config.MAX_DECEL * self.config.MASS),
            self.config.MAX_ACCEL * self.config.MASS)

        u1 = ca.fmin(
            ca.fmax(u[1], -self.config.MAX_STEER_V),
            self.config.MAX_STEER_V)

        return ca.vertcat(u0, u1)

    def clip_output(self, state):
        # state = [x, y, vx, yaw, vy, yaw_rate, steering_angle]
        vx = ca.fmin(
            ca.fmax(state[2], self.config.MIN_SPEED),
            self.config.MAX_SPEED)

        steering = ca.fmin(
            ca.fmax(state[6], self.config.MIN_STEER),
            self.config.MAX_STEER)

        return ca.vertcat(
            state[0],   # x
            state[1],   # y
            vx,
            state[3],   # yaw (no wrapping — sin/cos in dynamics handle any angle)
            state[4],   # vy
            state[5],   # yaw_rate
            steering    )
        
    def predictive_model(self, state, control_input, param):
        # state = [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements, NO theta)
        # control_input = [Fxr, delta_v]
        # Param is a CasADi vector; use indexing instead of unpacking (not iterable).
        # Order expected: [BR, CR, DR, BF, CF, DF, CM]
        self.BR = param[0]
        self.CR = param[1]
        self.DR = param[2]
        self.BF = param[3]
        self.CF = param[4]
        self.DF = param[5]
        self.CM = param[6]

        state = self.clip_output(state)
        control_input = self.clip_input(control_input)
        
        x = state[0]
        y = state[1]
        vx = state[2]
        yaw = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering_angle = state[6]
        
        Fxr = control_input[0]
        delta_v = control_input[1]

        # Safe velocity handling
        vx_safe = ca.fmax(ca.fabs(vx), 0.05)
        vx_safe = ca.sign(vx) * vx_safe

        # Tire slip angles
        alfa_f = steering_angle - ca.atan2(yaw_rate * self.LF + vy, vx_safe)
        alfa_r = ca.atan2(yaw_rate * self.LR - vy, vx_safe)

        # Pacejka tire model
        Ffy = self.DF * ca.sin(self.CF * ca.atan(self.BF * alfa_f))
        Fry = self.DR * ca.sin(self.CR * ca.atan(self.BR * alfa_r))

        # Longitudinal forces
        Fx = self.CM * Fxr - self.CR0 - self.CR2 * vx_safe ** 2.0
        Frx = Fx * (1.0 - self.TORQUE_SPLIT)
        Ffx = Fx * self.TORQUE_SPLIT

        # Vehicle dynamics (7 states)
        dx = vx_safe * ca.cos(yaw) - vy * ca.sin(yaw)
        dy = vx_safe * ca.sin(yaw) + vy * ca.cos(yaw)
        dvx = (1.0 / self.MASS) * (Frx - Ffy * ca.sin(steering_angle) + Ffx * ca.cos(steering_angle) + vy * yaw_rate * self.MASS)
        dyaw = yaw_rate
        dvy = (1.0 / self.MASS) * (Fry + Ffy * ca.cos(steering_angle) + Ffx * ca.sin(steering_angle) - vx_safe * yaw_rate * self.MASS)
        dyaw_rate = (1.0 / self.I_Z) * (Ffy * self.LF * ca.cos(steering_angle) - Fry * self.LR)
        dsteering = delta_v
        
        f = ca.vertcat(dx, dy, dvx, dyaw, dvy, dyaw_rate, dsteering)
        
        return f
    
    def rk4_step(self, x, u, param):
        dt = self.DTK
        k1 = self.predictive_model(x, u, param)
        k2 = self.predictive_model(x + dt/2 * k1, u, param)
        k3 = self.predictive_model(x + dt/2 * k2, u, param)
        k4 = self.predictive_model(x + dt * k3, u, param)
        x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def get_initial_guess(self, x0, param, theta0):
        """
        Build a rollout-based initial guess consistent with dynamics and theta progression.
        """
        # Allocate arrays
        states = np.zeros((self.config.NXK, self.config.TK + 1), dtype=float)
        controls = np.zeros((self.config.NU, self.config.TK), dtype=float)
        theta_arr = np.zeros(self.config.TK + 1, dtype=float)
        vi_arr = np.zeros(self.config.TK, dtype=float)

        # Initial state
        states[:, 0] = x0
        theta_arr[0] = theta0

        # Rollout with zero inputs
        for t in range(self.config.TK):
            u_t = np.array([0.0, 0.0], dtype=float)
            controls[:, t] = u_t
            x_next = self.rk4_step(states[:, t], u_t, param)
            states[:, t + 1] = np.array(x_next).astype(float).flatten()

        # Compute theta from path lookup and unwrap for continuity
        theta_lookup = np.zeros(self.config.TK + 1, dtype=float)

        for t in range(self.config.TK + 1):
            theta_lookup[t] = self.look_theta.query(states[0, t], states[1, t], k_neighbors=5)

        theta_unwrap = np.zeros_like(theta_lookup)
        theta_unwrap[0] = theta_lookup[0]
        for t in range(1, self.config.TK + 1):
            prev = theta_unwrap[t - 1]
            cand = theta_lookup[t]
            # Adjust by multiples of track length to minimize jump
            delta = cand - prev
            delta_wrapped = (delta + 0.5 * self.track_length) % self.track_length - 0.5 * self.track_length
            theta_unwrap[t] = prev + delta_wrapped

        theta_arr[:] = theta_unwrap
        for t in range(self.config.TK):
            vi_arr[t] = (theta_arr[t + 1] - theta_arr[t]) / self.DTK

        # Pack into decision vector
        x0_opt = np.zeros(self.n_states + self.n_controls + self.n_theta + self.n_vi, dtype=float)
        idx = 0
        x0_opt[idx:idx + self.n_states] = states.T.reshape(-1)
        idx += self.n_states
        x0_opt[idx:idx + self.n_controls] = controls.T.reshape(-1)
        idx += self.n_controls
        x0_opt[idx:idx + self.n_theta] = theta_arr
        idx += self.n_theta
        x0_opt[idx:idx + self.n_vi] = vi_arr

        return x0_opt
    
    def mpc_prob_init(self):
        xk      = ca.MX.sym('xk', self.config.NXK, self.config.TK + 1)   # NXK = 7 (no theta)
        uk      = ca.MX.sym('uk', self.config.NU, self.config.TK)        # NU = 2
        theta_k = ca.MX.sym('theta_k', self.config.TK + 1)          # Theta as separate variable
        vik     = ca.MX.sym('vik', self.config.TK)                      # v_theta
        
        # Parameters
        x0k    = ca.MX.sym('x0k', self.config.NXK)
        theta0 = ca.MX.sym('theta0')  # Initial theta
        param  = ca.MX.sym('param', self.config.num_param)

        # CasADi interpolants for symbolic evaluation of the reference (periodic extension)
        theta_grid = np.array(theta, dtype=float)
        x_grid = np.array(x, dtype=float)
        y_grid = np.array(y, dtype=float)
        phi_grid = np.unwrap(np.array([lookup_phi(t) for t in theta_grid]))

        # Sort by theta
        order = np.argsort(theta_grid)
        theta_grid = theta_grid[order]
        x_grid = x_grid[order]
        y_grid = y_grid[order]
        phi_grid = phi_grid[order]

        # Remove duplicates (strictly increasing required)
        mask = np.ones_like(theta_grid, dtype=bool)
        mask[1:] = theta_grid[1:] > theta_grid[:-1]
        theta_grid = theta_grid[mask]
        x_grid = x_grid[mask]
        y_grid = y_grid[mask]
        phi_grid = phi_grid[mask]

        L = float(self.track_length)

        # If the grid includes both endpoints of a full loop, drop the last point
        # to keep strict monotonicity after periodic extension.
        span = theta_grid[-1] - theta_grid[0]
        if np.isclose(span, L, rtol=0.0, atol=1e-8 * max(1.0, L)):
            theta_grid = theta_grid[:-1]
            x_grid = x_grid[:-1]
            y_grid = y_grid[:-1]
            phi_grid = phi_grid[:-1]

        # Periodic extension
        theta_ext = np.concatenate([theta_grid - L, theta_grid, theta_grid + L])
        x_ext = np.concatenate([x_grid, x_grid, x_grid])
        y_ext = np.concatenate([y_grid, y_grid, y_grid])
        phi_ext = np.concatenate([phi_grid - 2.0 * np.pi, phi_grid, phi_grid + 2.0 * np.pi])

        self.ref_x_fun = ca.interpolant('ref_x_fun', 'bspline', [theta_ext], x_ext)
        self.ref_y_fun = ca.interpolant('ref_y_fun', 'bspline', [theta_ext], y_ext)
        self.ref_phi_fun = ca.interpolant('ref_phi_fun', 'bspline', [theta_ext], phi_ext)

        objective = 0.0
        constraints = []
        lbg = []
        ubg = []

        # Dynamics constraints for vehicle states
        for t in range(self.config.TK):
            x_next = self.rk4_step(xk[:, t], uk[:, t], param)
            
            constraints.append(xk[:, t + 1] - x_next)
            lbg.extend([0.0] * self.config.NXK)
            ubg.extend([0.0] * self.config.NXK)
        
  
        for t in range(self.config.TK):
            theta_next = theta_k[t] + self.DTK * vik[t]
            constraints.append(theta_k[t + 1] - theta_next)
            lbg.append(0.0)
            ubg.append(0.0)

        # Cost function - contouring and lag errors
        for t in range(self.config.TK + 1):
            theta_t = ca.fmod(theta_k[t] - self.theta_min, self.track_length) + self.theta_min
            x_ref = self.ref_x_fun(theta_t)
            y_ref = self.ref_y_fun(theta_t)
            phi_t = self.ref_phi_fun(theta_t)
            sin_phi_t = ca.sin(phi_t)
            cos_phi_t = ca.cos(phi_t)

            # Position error relative to reference at theta_k[t]
            dx = xk[0, t] - x_ref
            dy = xk[1, t] - y_ref
            # Contouring error (perpendicular to path)
            e_c = sin_phi_t * dx - cos_phi_t * dy
            # Lag error (along path)
            e_l = -cos_phi_t * dx - sin_phi_t * dy
            objective += self.q_contour * e_c ** 2
            objective += self.q_lag * e_l ** 2
        # Progress reward - maximize theta progression (negative cost)
        for t in range(self.config.TK):
            objective += -self.q_theta * vik[t]

        # Input control effort
        for t in range(self.config.TK):
            p_u_1 = uk[0, t]
            p_u_2 = uk[1, t]
            p_vi = vik[t]
            p_u = ca.vertcat(p_u_1, p_u_2, p_vi)
            objective += p_u.T@self.config.Rk_ca@ p_u

        # Input smoothness
        for t in range(self.config.TK - 1):
            du_1 = uk[0, t + 1] - uk[0, t]
            du_2 = uk[1, t + 1] - uk[1, t]
            dvi = vik[t + 1] - vik[t]
            du = ca.vertcat(du_1, du_2, dvi)
            objective += du.T@self.config.Rdk_ca@ du

        # Initial condition constraints
        constraints.append(xk[:, 0] - x0k)
        lbg.extend([0.0] * self.config.NXK)
        ubg.extend([0.0] * self.config.NXK)
        
        # Initial theta constraint
        constraints.append(theta_k[0] - theta0)
        lbg.append(0.0)
        ubg.append(0.0)

        # Concatenate constraints
        g = ca.vertcat(*constraints)

        # Decision variable vector
        opt_variables = ca.vertcat(
            ca.reshape(xk, -1, 1),      # States (7 x (TK+1))
            ca.reshape(uk, -1, 1),      # Controls (2 x TK)
            theta_k,                     # Theta (TK+1)
            vik                          # v_theta (TK)
        )

        # Parameter vector
        opt_params = ca.vertcat(
            x0k,
            theta0,
            param
        )

        # Create NLP
        nlp = {
            'x': opt_variables,
            'f': objective,
            'g': g,
            'p': opt_params
        }

        opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 1000,  # Reduce iterations
        'ipopt.tol': 1e-2,  # Relax tolerance
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive',
        'print_time': 0,
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)       
        
        # Store sizes for unpacking
        self.n_states = self.config.NXK * (self.config.TK + 1)
        self.n_controls = self.config.NU * self.config.TK
        self.n_theta = self.config.TK + 1
        self.n_vi = self.config.TK

        # Bounds on decision variables
        self.lbx = []
        self.ubx = []

        # State bounds (7 states: x, y, vx, yaw, vy, yaw_rate, steering)
        # for t in range(self.config.TK + 1):
        self.lbx.extend([
            self.config.MIN_POS_X,                    # x
            self.config.MIN_POS_Y,                    # y
            self.config.MIN_SPEED,      # vx
            -np.inf,                        # yaw
            self.config.MIN_SPEED_LAT,                    # vy
            -np.inf,                    # yaw_rate
            self.config.MIN_STEER       # steering_angle
        ] * (self.config.TK + 1))
        self.ubx.extend([
            self.config.MAX_POS_X,                     # x
            self.config.MAX_POS_Y,                     # y
            self.config.MAX_SPEED,      # vx
            np.inf,                # yaw
            self.config.MAX_SPEED_LAT,                     # vy
            np.inf,                     # yaw_rate
            self.config.MAX_STEER       # steering_angle
        ] * (self.config.TK + 1))

        # # Control bounds
        # for t in range(self.config.TK):
        self.lbx.extend([
            self.config.MAX_DECEL * self.MASS,  # Fxr
            -self.config.MAX_STEER_V             # delta_v
        ] * self.config.TK)
        self.ubx.extend([
            self.config.MAX_ACCEL * self.MASS,  # Fxr
            self.config.MAX_STEER_V              # delta_v
        ] * self.config.TK)

        self.lbx.extend([self.config.MIN_THETA] * (self.config.TK + 1))
        self.ubx.extend([self.config.MAX_THETA] * (self.config.TK + 1))

        self.lbx.extend([self.config.MIN_VI] * self.config.TK)
        self.ubx.extend([self.config.MAX_VI] * self.config.TK)

        self.lbg = lbg
        self.ubg = ubg
        # Initial guess

        self.x0_opt = self.get_initial_guess(self.x0, self.param, self.theta0)

        self.solver(
                x0  = self.x0_opt,
                lbx = self.lbx,
                ubx = self.ubx,
                lbg = self.lbg,
                ubg = self.ubg,
                p   = opt_params)



    def mpc_prob_solve(self, x0, param, theta_0):
        """
        x0 should be [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements)
        theta is handled separately
        """
        # x0 = x0.copy()

        x0_opt = x0.copy()

        # Get current theta from vehicle position
        # print("initial states:", x0)
        # print("self.look_theta.query(x0[0], x0[1], k_neighbors=5)")

        # Ensure yaw continuity with previous warm-start solution.
        # Without this, a simulator wrap (e.g. 3.14 → -3.14) makes the
        # warm start inconsistent and the solver may diverge.
        if self.x0_opt is not None and np.any(self.x0_opt != 0):
            prev_yaw = self.x0_opt[3]  # yaw of first state in previous solution
            diff = x0[3] - prev_yaw
            x0[3] -= np.round(diff / (2.0 * np.pi)) * 2.0 * np.pi

        # Build parameter vector
        ca_para = np.concatenate([
             x0,
            [theta_0],
             param
        ])

        sol = self.solver(p = ca_para)

            
        solver_stats   = self.solver.stats()
        is_successful  = solver_stats['success']
        status_message = solver_stats['return_status']

        print("Solver status:", status_message)


        if is_successful:
            print(f"Solver succeeded with status: {status_message}")
            iterations = solver_stats['iter_count']
            print(f"IPOPT converged in {iterations} iterations.")
        else:
            print(f"Solver failed with status: {status_message}")

        # Extract solution
        x_opt = sol['x'].full().flatten()
        
        # Unpack states, controls, theta, and v_theta
        idx = 0
        states = x_opt[idx:idx + self.n_states].reshape((self.config.TK + 1, self.config.NXK)).T
        idx += self.n_states
        controls = x_opt[idx:idx + self.n_controls].reshape((self.config.TK, self.config.NU)).T
        idx += self.n_controls
        theta = x_opt[idx:idx + self.n_theta]
        vi = x_opt[idx:idx + self.n_vi]

        yaw_offset = np.round(states[3, 0] / (2.0 * np.pi)) * 2.0 * np.pi
        for t in range(self.config.TK + 1):
            x_opt[t * self.config.NXK + 3] -= yaw_offset
        self.x0_opt = x_opt 

        return controls, states, theta, status_message
            
        


    def MPCC_Control(self, x0_full, param, theta_0):
        """
        x0_full can be either:
        - [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements)
        """

        x0 = x0_full[:self.config.NXK]
        # Solve MPCC
        input_o, states_output, theta_output, status = self.mpc_prob_solve(x0, param, theta_0)

        if not np.any(np.isnan(states_output)):
            self.states_output = states_output
            self.input_o = input_o
            self.theta_output = theta_output
        else:
            raise ValueError("MPCC solver failed to find a valid solution.")
        
        # Extract control output (2 inputs: Fxr, delta_v)
        self.u_his = input_o
        u = self.input_o[:, 0]
        
        # Generate reference path based on optimized theta trajectory
        ref_path_x = np.zeros(self.config.TK + 1)
        ref_path_y = np.zeros(self.config.TK + 1)
        t0 = time.time()
        for t in range(self.config.TK + 1):
            theta_t = (self.theta_output[t] - self.theta_min) % self.track_length + self.theta_min
            ref_path_x[t], ref_path_y[t] = lookup_xy(theta_t)
        pred_x = states_output[0, :]
        pred_y = states_output[1, :]
        return u, ref_path_x, ref_path_y, pred_x, pred_y
  
if __name__ == '__main__':
    start_point = 10 # index on the trajectory to start from
    dyn_config = MPCConfigDYN()

    map_file     = 'data/rounded_rectangle_waypoints.csv'
    tpamap_name  = 'data/rounded_rectangle_tpamap.csv'
    tpadata_name = 'data/rounded_rectangle_tpadata.json'

    tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

    tpadata = {}
    with open(tpadata_name) as f:
        tpadata = json.load(f)

    raceline  = np.loadtxt(map_file, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    # Initialize with velocity with 5.0 <= <= 8.5 (or must increase the interation in the interation in the solver)
    planner_dyn_mpc   = STMPCCPlannerCasadi(waypoints=waypoints, config=dyn_config)

    with open('data/log_full_Vinit_8.0_c50.0_l3000.0_p18.0_weightslip0.5_thetaslip_50_200_275_325', 'r') as f:
        data = json.load(f)
    print(data.keys())


    X  = jnp.array(data['x'])
    Y  = jnp.array(data['y'])
    Yaw = jnp.array(data['yaw'])
    Yaw_rate = jnp.array(data['yaw_rate'])
    VX = jnp.array(data['vx'])
    VY = jnp.array(data['vy'])
    STR_angle = jnp.array(data['steer_angle'])


    print("=============================================== ")
    test_index = 100
    test_state = np.array([X[test_index], Y[test_index], VX[test_index], Yaw[test_index], VY[test_index], Yaw_rate[test_index], STR_angle[test_index]])
    print("Testing with state:", test_state)
    print("=============================================== ")


    BR = 15.9504
    CR = 1.3754
    DR = 4500.9280
    BF = 9.4246
    CF = 5.9139  
    DF = 4500.8218
    CM = 0.9459
    param = np.array([BR, CR, DR, BF, CF, DF, CM], dtype=float)

    u, _, _, _, _ = planner_dyn_mpc.plan(test_state, param)
    u[0] = u[0] / planner_dyn_mpc.config.MASS
    print("Optimal acceleration:", u[0])
    print("Optimal steering speed:", u[1])




