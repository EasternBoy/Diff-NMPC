import casadi as ca
import numpy as np

from DiffMPCC.MPCCsolver import (
    MPCConfigDYN,
    ThetaLookupTable,
    lookup_phi,
    n_neighbors,
    spline_x,
    spline_y,
    theta,
    theta_max,
    theta_min,
    x,
    y,
)


class CasadiOuterSensitivityMPCC:
    """
    MPCC + KKT sensitivity for d(outer_loss)/dq where q=[q_contour, q_lag, q_theta].
    https://web.casadi.org/blog/nlp_sens/

    Notes:
    - Uses equality-constrained KKT sensitivity. Active bound multipliers are not included,
      so gradients are approximate near active variable bounds.
    - Outer loss is defined on the solved trajectory and is independent of q directly;
      q influences loss through z*(q).
    """

    def __init__(self, config: MPCConfigDYN):
        self.config = config
        self.DTK = float(config.DTK)
        self.look_theta = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=200000)
        self.theta_min = float(theta_min)
        self.track_length = float(theta_max - theta_min)

        self._build_nlp_and_sensitivity()
        self.init_sol = np.zeros(self.nz, dtype=float)

    def _predictive_model_sym(self, state, control_input, dyn):
        # state = [x, y, vx, yaw, vy, yaw_rate, steering_angle]
        # control_input = [Fxr, delta_v]
        BR, CR, DR, BF, CF, DF, CM = dyn[0], dyn[1], dyn[2], dyn[3], dyn[4], dyn[5], dyn[6]

        vx = ca.fmin(ca.fmax(state[2], self.config.MIN_SPEED), self.config.MAX_SPEED)
        steering = ca.fmin(ca.fmax(state[6], self.config.MIN_STEER), self.config.MAX_STEER)

        X = state[0]
        Y = state[1]
        yaw = state[3]
        vy = state[4]
        yaw_rate = state[5]

        Fxr = ca.fmin(
            ca.fmax(control_input[0], self.config.MAX_DECEL * self.config.MASS),
            self.config.MAX_ACCEL * self.config.MASS,
        )
        delta_v = ca.fmin(ca.fmax(control_input[1], -self.config.MAX_STEER_V), self.config.MAX_STEER_V)

        vx_safe = ca.sign(vx) * ca.fmax(ca.fabs(vx), 0.05)

        alfa_f = steering - ca.atan2(yaw_rate * self.config.LF + vy, vx_safe)
        alfa_r = ca.atan2(yaw_rate * self.config.LR - vy, vx_safe)

        Ffy = DF * ca.sin(CF * ca.atan(BF * alfa_f))
        Fry = DR * ca.sin(CR * ca.atan(BR * alfa_r))

        Fx = CM * Fxr - self.config.CR0 - self.config.CR2 * vx_safe ** 2.0
        Frx = Fx * (1.0 - self.config.TORQUE_SPLIT)
        Ffx = Fx * self.config.TORQUE_SPLIT

        dx = vx_safe * ca.cos(yaw) - vy * ca.sin(yaw)
        dy = vx_safe * ca.sin(yaw) + vy * ca.cos(yaw)
        dvx = (1.0 / self.config.MASS) * (
            Frx - Ffy * ca.sin(steering) + Ffx * ca.cos(steering) + vy * yaw_rate * self.config.MASS
        )
        dyaw = yaw_rate
        dvy = (1.0 / self.config.MASS) * (
            Fry + Ffy * ca.cos(steering) + Ffx * ca.sin(steering) - vx_safe * yaw_rate * self.config.MASS
        )
        dyaw_rate = (1.0 / self.config.I_Z) * (Ffy * self.config.LF * ca.cos(steering) - Fry * self.config.LR)
        dsteering = delta_v

        return ca.vertcat(dx, dy, dvx, dyaw, dvy, dyaw_rate, dsteering)

    def _build_nlp_and_sensitivity(self):
        NXK = self.config.NXK
        NU = self.config.NU
        TK = self.config.TK

        xk      = ca.MX.sym("xk", NXK, TK + 1)
        uk      = ca.MX.sym("uk", NU, TK)
        theta_k = ca.MX.sym("theta_k", TK + 1)
        vik     = ca.MX.sym("vik", TK)

        x0k = ca.MX.sym("x0k", NXK)
        theta0 = ca.MX.sym("theta0")
        dyn = ca.MX.sym("dyn", self.config.num_param)
        q = ca.MX.sym("q", 3)

        theta_grid = np.asarray(theta, dtype=float)
        x_grid = np.asarray(x, dtype=float)
        y_grid = np.asarray(y, dtype=float)
        phi_grid = np.unwrap(np.asarray([lookup_phi(ti) for ti in theta_grid], dtype=float))

        order = np.argsort(theta_grid)
        theta_grid = theta_grid[order]
        x_grid = x_grid[order]
        y_grid = y_grid[order]
        phi_grid = phi_grid[order]

        dedup_mask = np.ones_like(theta_grid, dtype=bool)
        dedup_mask[1:] = theta_grid[1:] > theta_grid[:-1]
        theta_grid = theta_grid[dedup_mask]
        x_grid = x_grid[dedup_mask]
        y_grid = y_grid[dedup_mask]
        phi_grid = phi_grid[dedup_mask]

        L = self.track_length
        if np.isclose(theta_grid[-1] - theta_grid[0], L):
            theta_grid = theta_grid[:-1]
            x_grid = x_grid[:-1]
            y_grid = y_grid[:-1]
            phi_grid = phi_grid[:-1]

        theta_ext = np.concatenate([theta_grid - L, theta_grid, theta_grid + L])
        x_ext = np.concatenate([x_grid, x_grid, x_grid])
        y_ext = np.concatenate([y_grid, y_grid, y_grid])
        phi_ext = np.concatenate([phi_grid - 2.0 * np.pi, phi_grid, phi_grid + 2.0 * np.pi])

        ref_x_fun = ca.interpolant("ref_x_fun_sens", "bspline", [theta_ext], x_ext)
        ref_y_fun = ca.interpolant("ref_y_fun_sens", "bspline", [theta_ext], y_ext)
        ref_phi_fun = ca.interpolant("ref_phi_fun_sens", "bspline", [theta_ext], phi_ext)

        constraints = []
        lbg = []
        ubg = []
        inner_objective = 0.0
        outer_objective = 0.0

        for t in range(TK):
            x_next = xk[:, t] + self.DTK * self._predictive_model_sym(xk[:, t], uk[:, t], dyn)
            constraints.append(xk[:, t + 1] - x_next)
            lbg.extend([0.0] * NXK)
            ubg.extend([0.0] * NXK)

            theta_next = theta_k[t] + self.DTK * vik[t]
            constraints.append(theta_k[t + 1] - theta_next)
            lbg.append(0.0)
            ubg.append(0.0)

        for t in range(TK + 1):
            theta_t = ca.fmod(theta_k[t] - self.theta_min, self.track_length) + self.theta_min
            x_ref = ref_x_fun(theta_t)
            y_ref = ref_y_fun(theta_t)
            phi_t = ref_phi_fun(theta_t)

            dx = xk[0, t] - x_ref
            dy = xk[1, t] - y_ref
            e_c = ca.sin(phi_t) * dx - ca.cos(phi_t) * dy
            e_l = -ca.cos(phi_t) * dx - ca.sin(phi_t) * dy

            inner_objective += q[0] * e_c ** 2 + q[1] * e_l ** 2
            # Fixed outer objective to learn q from trajectory quality.
            # Define outer objective as trajectory tracking error, independent of q directly.
            outer_objective += e_c ** 2 + e_l ** 2 # Change by your design

        for t in range(TK):
            inner_objective += -q[2] * vik[t]
            u_aug            = ca.vertcat(uk[0, t], uk[1, t], vik[t])
            inner_objective += ca.mtimes([u_aug.T, self.config.Rk_ca, u_aug])
            # Define outer objective to also include control effort, which can influence the optimal trajectory and thus provide a learning signal for q.
            outer_objective += 1e-3 * (uk[0, t] ** 2 + uk[1, t] ** 2) # Change by your design

        for t in range(TK - 1):
            du_aug = ca.vertcat(uk[0, t + 1] - uk[0, t], uk[1, t + 1] - uk[1, t], vik[t + 1] - vik[t])
            inner_objective += ca.mtimes([du_aug.T, self.config.Rdk_ca, du_aug])

        constraints.append(xk[:, 0] - x0k)
        lbg.extend([0.0] * NXK)
        ubg.extend([0.0] * NXK)
        constraints.append(theta_k[0] - theta0)
        lbg.append(0.0)
        ubg.append(0.0)

        g = ca.vertcat(*constraints)
        z = ca.vertcat(ca.reshape(xk, -1, 1), ca.reshape(uk, -1, 1), theta_k, vik)
        p = ca.vertcat(x0k, theta0, dyn, q)

        nlp = {"x": z, "f": inner_objective, "g": g, "p": p}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-2,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
            "print_time": 0,
        }

        self.solver = ca.nlpsol("solver_sens", "ipopt", nlp, opts)

        self.nz = int(z.numel())
        self.ng = int(g.numel())
        self.np = int(p.numel())

        self.p_q_start = NXK + 1 + self.config.num_param

        self.lbx, self.ubx = self._build_bounds()
        self.lbg = np.asarray(lbg, dtype=float)
        self.ubg = np.asarray(ubg, dtype=float)

        lam_g      = ca.MX.sym("lam_g", self.ng) # Lagrange multipliers for equality constraints
        lagrangian = inner_objective + ca.dot(lam_g, g)
        r_kkt      = ca.vertcat(ca.gradient(lagrangian, z), g) # KKT residuals: [dL/dz; g(z,p)]
        w          = ca.vertcat(z, lam_g)

        Jw = ca.jacobian(r_kkt, w) 
        Jp = ca.jacobian(r_kkt, p)

        douter_dz = ca.jacobian(outer_objective, z)
        douter_dp = ca.jacobian(outer_objective, p)

        self.kkt_jac_fun    = ca.Function("kkt_jac_fun", [z, lam_g, p], [Jw, Jp])
        self.outer_grad_fun = ca.Function("outer_grad_fun", [z, p], [outer_objective, douter_dz, douter_dp])

    def _build_bounds(self):
        TK = self.config.TK
        lbx = []
        ubx = []

        lbx.extend(
            [
                self.config.MIN_POS_X,
                self.config.MIN_POS_Y,
                self.config.MIN_SPEED,
                -np.inf,
                self.config.MIN_SPEED_LAT,
                -np.inf,
                self.config.MIN_STEER,
            ]
            * (TK + 1)
        )
        ubx.extend(
            [
                self.config.MAX_POS_X,
                self.config.MAX_POS_Y,
                self.config.MAX_SPEED,
                np.inf,
                self.config.MAX_SPEED_LAT,
                np.inf,
                self.config.MAX_STEER,
            ]
            * (TK + 1)
        )

        lbx.extend([self.config.MAX_DECEL * self.config.MASS, -self.config.MAX_STEER_V] * TK)
        ubx.extend([self.config.MAX_ACCEL * self.config.MASS,  self.config.MAX_STEER_V] * TK)

        lbx.extend([self.config.MIN_THETA] * (TK + 1))
        ubx.extend([self.config.MAX_THETA] * (TK + 1))

        lbx.extend([self.config.MIN_VI] * TK)
        ubx.extend([self.config.MAX_VI] * TK)

        return np.asarray(lbx, dtype=float), np.asarray(ubx, dtype=float)

    def solve(self, init_state, dyn_param, q, theta0=None):
        x0   = np.asarray(init_state, dtype=float).reshape(-1)
        dyn  = np.asarray(dyn_param, dtype=float).reshape(-1)
        q_np = np.asarray(q, dtype=float).reshape(-1)

        if theta0 is None:
            theta0 = self.look_theta.query(float(x0[0]), float(x0[1]), k_neighbors=n_neighbors)

        p_vec = np.concatenate([x0, [float(theta0)], dyn, q_np])

        sol = self.solver(
            x0 =self.init_sol,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p  =p_vec,
        )

        z_star     = np.asarray(sol["x"], dtype=float).reshape(-1)
        lam_g_star = np.asarray(sol["lam_g"], dtype=float).reshape(-1)

        self.init_sol = z_star.copy()

        return {
            "z": z_star,
            "lam_g": lam_g_star,
            "p": p_vec,
            "theta0": float(theta0),
            "status": self.solver.stats().get("return_status", "unknown"),
            "success": bool(self.solver.stats().get("success", False)),
        }

    def outer_loss_and_grad_q(self, init_state, dyn_param, q, theta0=None):
        out = self.solve(init_state, dyn_param, q, theta0=theta0)

        z     = out["z"]
        lam_g = out["lam_g"]
        p_vec = out["p"]

        Jw, Jp = self.kkt_jac_fun(z, lam_g, p_vec)
        Jw = np.asarray(Jw, dtype=float)
        Jp = np.asarray(Jp, dtype=float)

        try:
            dw_dp = -np.linalg.solve(Jw, Jp)
        except np.linalg.LinAlgError:
            dw_dp = -np.linalg.lstsq(Jw, Jp, rcond=None)[0]

        dz_dp = dw_dp[: self.nz, :]

        outer_loss, douter_dz, douter_dp = self.outer_grad_fun(z, p_vec)
        douter_dz = np.asarray(douter_dz, dtype=float)
        douter_dp = np.asarray(douter_dp, dtype=float)

        grad_p = douter_dp + douter_dz @ dz_dp
        grad_q = grad_p[0, self.p_q_start : self.p_q_start + 3]

        return float(outer_loss), grad_q, out

    def gradient_step_q(self, init_state, dyn_param, q, lr=1e-3, iters=1):
        q_curr = np.asarray(q, dtype=float).reshape(-1)
        loss   = 0.0
        grad_q = np.zeros_like(q_curr)

        for _ in range(int(iters)):
            loss, grad_q, _ = self.outer_loss_and_grad_q(init_state, dyn_param, q_curr)
            q_curr = np.maximum(q_curr - lr * grad_q, 1e-6)

        return q_curr, float(loss), grad_q

    def _unpack_solution(self, z):
        NXK = self.config.NXK
        NU = self.config.NU
        TK = self.config.TK

        idx = 0
        n_states = NXK * (TK + 1)
        states = np.asarray(z[idx : idx + n_states], dtype=float).reshape(TK + 1, NXK)
        idx += n_states

        n_controls = NU * TK
        controls = np.asarray(z[idx : idx + n_controls], dtype=float).reshape(TK, NU)
        idx += n_controls

        theta_seq = np.asarray(z[idx : idx + (TK + 1)], dtype=float)
        idx += TK + 1
        vi_seq = np.asarray(z[idx : idx + TK], dtype=float)
        return states, controls, theta_seq, vi_seq

    def outer_loss_and_grad_q_closed_loop(self, init_state, dyn_param, q, outer_steps):
        """
        Closed-loop outer objective with horizon `outer_steps`.

        Each outer step solves a TK-horizon inner MPC, accumulates sensitivity
        gradient wrt q, then advances to the next state using the first predicted state.
        """
        state = np.asarray(init_state, dtype=float).reshape(-1)
        q_np = np.asarray(q, dtype=float).reshape(-1)
        total_loss = 0.0
        total_grad_q = np.zeros(3, dtype=float)

        theta0 = self.look_theta.query(float(state[0]), float(state[1]), k_neighbors=n_neighbors)

        for _ in range(int(outer_steps)): # Loop over outer steps
            out = self.solve(state, dyn_param, q_np, theta0=theta0)
            if not out["success"]:
                break

            z     = out["z"]
            lam_g = out["lam_g"]
            p_vec = out["p"]

            Jw, Jp = self.kkt_jac_fun(z, lam_g, p_vec)
            Jw = np.asarray(Jw, dtype=float)
            Jp = np.asarray(Jp, dtype=float)

            try:
                dw_dp = -np.linalg.solve(Jw, Jp)
            except np.linalg.LinAlgError:
                dw_dp = -np.linalg.lstsq(Jw, Jp, rcond=None)[0]

            dz_dp = dw_dp[: self.nz, :]

            outer_loss, douter_dz, douter_dp = self.outer_grad_fun(z, p_vec)
            douter_dz = np.asarray(douter_dz, dtype=float)
            douter_dp = np.asarray(douter_dp, dtype=float)
            grad_p    = douter_dp + douter_dz @ dz_dp
            grad_q    = grad_p[0, self.p_q_start : self.p_q_start + 3]

            total_loss   += float(outer_loss)
            total_grad_q += grad_q

            states, _, theta_seq, _ = self._unpack_solution(z)
            state  = states[1].copy()
            theta0 = float(theta_seq[1])

        return float(total_loss), total_grad_q

    def gradient_step_q_closed_loop(self, init_state, dyn_param, q, outer_steps, lr=1e-3, iters=1):
        q_curr = np.asarray(q, dtype=float).reshape(-1)
        loss = 0.0
        grad_q = np.zeros_like(q_curr)

        for _ in range(int(iters)):
            loss, grad_q = self.outer_loss_and_grad_q_closed_loop(
                init_state=init_state,
                dyn_param=dyn_param,
                q=q_curr,
                outer_steps=outer_steps,
            )
            q_curr = np.maximum(q_curr - lr * grad_q, 1e-6)

        return q_curr, float(loss), grad_q