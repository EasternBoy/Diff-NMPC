import casadi as ca
import numpy as np
import os
import sys
import ctypes
from pathlib import Path


def _preload_acados_macos_libs():
    if sys.platform != "darwin":
        return

    acados_root = os.environ.get("ACADOS_SOURCE_DIR", "/Users/duongtran/GitHub/acados")
    lib_dir = str(Path(acados_root) / "lib")
    if not Path(lib_dir).exists():
        return

    # Ensure dyld can resolve @rpath-linked acados dependencies in this process.
    dyld_paths = [p for p in os.environ.get("DYLD_LIBRARY_PATH", "").split(":") if p]
    fallback_paths = [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p]
    if lib_dir not in dyld_paths:
        dyld_paths.insert(0, lib_dir)
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(dyld_paths)
    if lib_dir not in fallback_paths:
        fallback_paths.insert(0, lib_dir)
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(fallback_paths)

    # Load in dependency order for macOS @rpath resolution.
    for lib_name in (
        "libblasfeo.0.dylib",
        "libblasfeo.dylib",
        "libhpipm.dylib",
        "libqpOASES_e.dylib",
        "libacados.dylib",
    ):
        lib_path = Path(lib_dir) / lib_name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_preload_acados_macos_libs()

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, get_tera
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

from MPCCsolver import (
    MPCConfigDYN,
    ThetaLookupTable,
    lookup_phi,
    n_neighbors,
    lin_iter_x,
    lin_iter_y,
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
        if not ACADOS_AVAILABLE:
            raise ImportError(
                "acados_template is not available in the current interpreter. "
                "Activate your local .venv and rerun."
            )
        self.config = config
        self.DTK = float(config.DTK)
        self.look_theta = ThetaLookupTable(lin_iter_x, lin_iter_y, theta_min, theta_max, n_samples=200000)
        self.theta_min = float(theta_min)
        self.track_length = float(theta_max - theta_min)
        self.backend = "acados"
        self.acados_solver = None
        self.solver = None
        self.use_predictive_model_in_warmstart = True

        self._ensure_tera_renderer()
        self._build_nlp_and_sensitivity()
        self.init_sol = np.zeros(self.nz, dtype=float)

    @staticmethod
    def _ensure_tera_renderer():
        """
        Keep a deterministic renderer version to avoid template rendering issues.
        """
        version = "0.2.0" if sys.platform == "darwin" else "0.0.34"
        try:
            tera_path = get_tera(tera_version=version, force_download=False)
            os.environ["TERA_PATH"] = os.path.abspath(tera_path)
        except Exception:
            pass

    def _predictive_model_sym(self, state, control_input, dyn):
        # state = [x, y, vx, yaw, vy, yaw_rate, steering_angle]
        # control_input = [Fxr, delta_v]
        BR, CR, DR, BF, CF, DF, CM = dyn[0], dyn[1], dyn[2], dyn[3], dyn[4], dyn[5], dyn[6]

        vx       = state[2]
        steering = state[6]

        X   = state[0]
        Y   = state[1]
        yaw = state[3]
        vy  = state[4]
        yaw_rate = state[5]

        Fxr     = control_input[0]
        delta_v = control_input[1]

        vx_safe = ca.sign(vx) * ca.fabs(vx)

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

        x0k    = ca.MX.sym("x0k", NXK)
        theta0 = ca.MX.sym("theta0")
        dyn    = ca.MX.sym("dyn", self.config.num_param)
        q      = ca.MX.sym("q", 3)

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
        ref_x_fun   = ca.interpolant("ref_x_fun_sens",   "linear", [theta_grid], x_grid)
        ref_y_fun   = ca.interpolant("ref_y_fun_sens",   "linear", [theta_grid], y_grid)
        ref_phi_fun = ca.interpolant("ref_phi_fun_sens", "linear", [theta_grid], phi_grid)

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
            theta_t = theta_k[t]
            x_ref   = ref_x_fun(theta_t)
            y_ref   = ref_y_fun(theta_t)
            phi_t   = ref_phi_fun(theta_t)

            dx = xk[0, t] - x_ref
            dy = xk[1, t] - y_ref

            e_c =  ca.sin(phi_t) * dx - ca.cos(phi_t) * dy
            e_l = -ca.cos(phi_t) * dx - ca.sin(phi_t) * dy

            # inner_objective += q[0] * e_c ** 2 + q[1] * e_l ** 2
            # Fixed outer objective to learn q from trajectory quality.
            # Define outer objective as trajectory tracking error, independent of q directly.
            outer_objective += e_c ** 2 + 1000*e_l ** 2 # Change by your design
            # 
        for t in range(TK):
            # inner_objective += -q[2] * vik[t]
            u_aug            = ca.vertcat(uk[0, t], uk[1, t], vik[t])
            inner_objective += ca.mtimes([u_aug.T, self.config.Rk_ca, u_aug])
            outer_objective += -100*vik[t]

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
        self.nlp = nlp

        self.nz = int(z.numel())
        self.ng = int(g.numel())
        self.np = int(p.numel())

        self.p_q_start = NXK + 1 + self.config.num_param

        self.lbx, self.ubx = self._build_bounds()
        self.lbg = np.asarray(lbg, dtype=float)
        self.ubg = np.asarray(ubg, dtype=float)
        n_dyn_x = TK * NXK
        n_dyn_theta = TK
        n_init_x = NXK
        n_init_theta = 1
        self._g_blocks = {
            "dyn_x": (0, n_dyn_x),
            "dyn_theta": (n_dyn_x, n_dyn_x + n_dyn_theta),
            "init_x": (n_dyn_x + n_dyn_theta, n_dyn_x + n_dyn_theta + n_init_x),
            "init_theta": (n_dyn_x + n_dyn_theta + n_init_x, n_dyn_x + n_dyn_theta + n_init_x + n_init_theta),
        }

        self._build_acados_solver(ref_x_fun, ref_y_fun, ref_phi_fun)

        grad_f = ca.gradient(inner_objective, z)
        jac_g  = ca.jacobian(g, z)
        self.grad_f_fun = ca.Function("grad_f_fun", [z, p], [grad_f])
        self.jac_g_fun  = ca.Function("jac_g_fun", [z, p], [jac_g])
        self.g_fun      = ca.Function("g_fun", [z, p], [g])

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
        x_sym = ca.MX.sym("x_sym", NXK)
        u_sym = ca.MX.sym("u_sym", NU)
        dyn_sym = ca.MX.sym("dyn_sym", self.config.num_param)
        if self.use_predictive_model_in_warmstart:
            x_next_sym = x_sym + self.DTK * self._predictive_model_sym(x_sym, u_sym, dyn_sym)
        else:
            x_next_sym = x_sym
        self.dyn_step_fun = ca.Function("dyn_step_fun_outer", [x_sym, u_sym, dyn_sym], [x_next_sym])

    def _build_acados_solver(self, ref_x_fun, ref_y_fun, ref_phi_fun):
        NXK = self.config.NXK
        TK  = self.config.TK
        NU_AUG = 3
        NX_AUG = NXK + 1 + NU_AUG

        x_aug = ca.MX.sym("x_aug", NX_AUG)
        u_aug = ca.MX.sym("u_aug", NU_AUG)
        p_ac  = ca.MX.sym("p_ac", self.config.num_param + 3)

        x_state     = x_aug[:NXK]
        theta_state = x_aug[NXK]
        u_prev      = x_aug[NXK + 1 : NXK + 1 + NU_AUG]
        dyn         = p_ac[: self.config.num_param]
        q           = p_ac[self.config.num_param :]

        x_next     = x_state + self.DTK * self._predictive_model_sym(x_state, u_aug[:2], dyn)
        theta_next = theta_state + self.DTK * u_aug[2]
        x_next_aug = ca.vertcat(x_next, theta_next, u_aug)

        theta_t = theta_state
        x_ref = ref_x_fun(theta_t)
        y_ref = ref_y_fun(theta_t)
        phi_t = ref_phi_fun(theta_t)

        dx = x_state[0] - x_ref
        dy = x_state[1] - y_ref
        e_c =  ca.sin(phi_t) * dx - ca.cos(phi_t) * dy
        e_l = -ca.cos(phi_t) * dx - ca.sin(phi_t) * dy
        du = u_aug - u_prev

        stage_cost = q[0] * e_c**2 + q[1] * e_l**2 - q[2] * u_aug[2]
        stage_cost += ca.mtimes([u_aug.T, self.config.Rk_ca, u_aug])
        stage_cost += ca.mtimes([du.T, self.config.Rdk_ca, du])

        model = AcadosModel()
        model.name = "outer_sens_mpcc_acados_model"
        model.x = x_aug
        model.u = u_aug
        model.p = p_ac
        model.disc_dyn_expr      = x_next_aug
        model.cost_expr_ext_cost = stage_cost
        model.cost_expr_ext_cost_e = ca.MX(0)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = TK
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.parameter_values = np.zeros(self.config.num_param + 3, dtype=float)

        def finite_lb(v):
            v = float(v)
            return v if np.isfinite(v) else -1e6

        def finite_ub(v):
            v = float(v)
            return v if np.isfinite(v) else 1e6

        idxbx = np.arange(NX_AUG, dtype=int)
        lbx = np.array(
            [
                finite_lb(self.config.MIN_POS_X),
                finite_lb(self.config.MIN_POS_Y),
                finite_lb(self.config.MIN_SPEED),
                -1e6,
                finite_lb(self.config.MIN_SPEED_LAT),
                -1e6,
                finite_lb(self.config.MIN_STEER),
                finite_lb(self.config.MIN_THETA),
                finite_lb(self.config.MAX_DECEL * self.config.MASS),
                finite_lb(-self.config.MAX_STEER_V),
                finite_lb(self.config.MIN_VI),
            ],
            dtype=float,
        )
        ubx = np.array(
            [
                finite_ub(self.config.MAX_POS_X),
                finite_ub(self.config.MAX_POS_Y),
                finite_ub(self.config.MAX_SPEED),
                1e6,
                finite_ub(self.config.MAX_SPEED_LAT),
                1e6,
                finite_ub(self.config.MAX_STEER),
                finite_ub(self.config.MAX_THETA),
                finite_ub(self.config.MAX_ACCEL * self.config.MASS),
                finite_ub(self.config.MAX_STEER_V),
                finite_ub(self.config.MAX_VI),
            ],
            dtype=float,
        )

        ocp.constraints.idxbx = idxbx
        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx
        ocp.constraints.idxbx_0 = idxbx.copy()
        ocp.constraints.lbx_0 = lbx.copy()
        ocp.constraints.ubx_0 = ubx.copy()
        ocp.constraints.idxbu = np.arange(NU_AUG, dtype=int)
        ocp.constraints.lbu = lbx[-NU_AUG:]
        ocp.constraints.ubu = ubx[-NU_AUG:]
        ocp.solver_options.tf = float(self.DTK * TK)
        ocp.solver_options.qp_solver       = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx  = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.regularize_method   = "CONVEXIFY"
        # Relax QP/NLP accuracy to reduce convergence stalls and favor strict success exit.
        ocp.solver_options.qp_solver_iter_max = 200
        ocp.solver_options.qp_solver_tol_stat = 1e-3
        ocp.solver_options.qp_solver_tol_eq = 1e-3
        ocp.solver_options.qp_solver_tol_ineq = 1e-3
        ocp.solver_options.qp_solver_tol_comp = 1e-3
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        ocp.solver_options.nlp_solver_tol_eq = 1e-3
        ocp.solver_options.nlp_solver_tol_ineq = 1e-3
        ocp.solver_options.nlp_solver_tol_comp = 1e-3
        ocp.solver_options.print_level = 0

        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_outer_sens_ocp.json")

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
        dyn  = np.asarray(dyn_param,  dtype=float).reshape(-1)
        q_np = np.clip(np.asarray(q, dtype=float).reshape(-1), 1e-9, 1e4)

        if theta0 is None:
            theta0 = self.look_theta.query(float(x0[0]), float(x0[1]), k_neighbors=n_neighbors)

        p_vec = np.concatenate([x0, [float(theta0)], dyn, q_np])
        p_ac  = np.concatenate([dyn, q_np])

        x0_warm = np.clip(np.asarray(self.init_sol, dtype=float), self.lbx, self.ubx)
        NXK     = self.config.NXK
        NU      = self.config.NU
        TK      = self.config.TK

        n_states       = NXK * (TK + 1)
        n_controls     = NU * TK
        idx            = 0
        states_guess   = x0_warm[idx : idx + n_states].reshape(TK + 1, NXK)
        idx           += n_states
        controls_guess = x0_warm[idx : idx + n_controls].reshape(TK, NU)
        idx           += n_controls
        theta_guess    = x0_warm[idx : idx + (TK + 1)]
        idx           += TK + 1
        vi_guess       = x0_warm[idx : idx + TK]
        u_prev0 = np.array([controls_guess[0, 0], controls_guess[0, 1], vi_guess[0]], dtype=float)
        x0_aug = np.concatenate([x0, [float(theta0)], u_prev0])

        def _apply_guess(controls_k, vi_k):
            states_guess_feas = np.zeros((TK + 1, NXK), dtype=float)
            theta_guess_feas = np.zeros(TK + 1, dtype=float)
            states_guess_feas[0, :] = x0
            theta_guess_feas[0] = float(theta0)
            for kk in range(TK):
                states_guess_feas[kk + 1, :] = np.asarray(
                    self.dyn_step_fun(states_guess_feas[kk, :], controls_k[kk, :], dyn), dtype=float
                ).reshape(-1)
                theta_guess_feas[kk + 1] = theta_guess_feas[kk] + self.DTK * vi_k[kk]

            for kk in range(TK):
                x_aug_guess = np.concatenate(
                    [
                        states_guess_feas[kk],
                        [theta_guess_feas[kk]],
                        [controls_k[kk, 0], controls_k[kk, 1], vi_k[kk]],
                    ]
                )
                u_aug_guess = np.array([controls_k[kk, 0], controls_k[kk, 1], vi_k[kk]], dtype=float)
                self.acados_solver.set(kk, "x", x_aug_guess)
                self.acados_solver.set(kk, "u", u_aug_guess)
                self.acados_solver.set(kk, "p", p_ac)

            xN_aug_guess = np.concatenate(
                [
                    states_guess_feas[TK],
                    [theta_guess_feas[TK]],
                    [controls_k[-1, 0], controls_k[-1, 1], vi_k[-1]],
                ]
            )
            self.acados_solver.set(TK, "x", xN_aug_guess)
            self.acados_solver.set(TK, "p", p_ac)

            self.acados_solver.constraints_set(0, "lbx", x0_aug)
            self.acados_solver.constraints_set(0, "ubx", x0_aug)
            self.acados_solver.set(0, "x", x0_aug)

        status_history = []
        _apply_guess(controls_guess, vi_guess)
        status = int(self.acados_solver.solve())
        status_history.append(status)
        if status != 0:
            # Retry once from a conservative feasible warm start.
            self.acados_solver.reset()
            controls_retry = np.zeros_like(controls_guess)
            vi_retry = np.full(TK, min(max(1.0, self.config.MIN_VI), self.config.MAX_VI), dtype=float)
            _apply_guess(controls_retry, vi_retry)
            status = int(self.acados_solver.solve())
            status_history.append(status)
            # Continue a few SQP calls from current iterate to improve convergence.
            for _ in range(3):
                if status == 0:
                    break
                for kk in range(TK):
                    self.acados_solver.set(kk, "p", p_ac)
                self.acados_solver.set(TK, "p", p_ac)
                self.acados_solver.constraints_set(0, "lbx", x0_aug)
                self.acados_solver.constraints_set(0, "ubx", x0_aug)
                status = int(self.acados_solver.solve())
                status_history.append(status)
        is_successful = status == 0
        status_message = "ACADOS_SUCCESS" if is_successful else f"ACADOS_STATUS_{status}"

        states_opt = np.zeros((TK + 1, NXK), dtype=float)
        theta_opt = np.zeros(TK + 1, dtype=float)
        controls_opt = np.zeros((TK, NU), dtype=float)
        vi_opt = np.zeros(TK, dtype=float)

        for t in range(TK):
            x_t = np.asarray(self.acados_solver.get(t, "x"), dtype=float).reshape(-1)
            u_t = np.asarray(self.acados_solver.get(t, "u"), dtype=float).reshape(-1)
            states_opt[t, :] = x_t[:NXK]
            theta_opt[t] = x_t[NXK]
            controls_opt[t, :] = u_t[:NU]
            vi_opt[t] = u_t[2]
        x_N = np.asarray(self.acados_solver.get(TK, "x"), dtype=float).reshape(-1)
        states_opt[TK, :] = x_N[:NXK]
        theta_opt[TK] = x_N[NXK]

        z_star = np.concatenate(
            [states_opt.reshape(-1), controls_opt.reshape(-1), theta_opt.reshape(-1), vi_opt.reshape(-1)]
        )

        lam_g_star = np.zeros(self.ng, dtype=float)

        has_nan = (
            (not np.isfinite(z_star).all())
            or (not np.isfinite(lam_g_star).all())
            or (not np.isfinite(p_vec).all())
        )
        max_constraint_violation = np.nan
        block_violations = {}
        try:
            g_val = np.asarray(self.g_fun(z_star, p_vec), dtype=float).reshape(-1)
            low_violation = np.maximum(self.lbg - g_val, 0.0)
            high_violation = np.maximum(g_val - self.ubg, 0.0)
            g_violation = np.maximum(low_violation, high_violation)
            max_constraint_violation = float(np.max(g_violation))
            for block_name, (i0, i1) in self._g_blocks.items():
                if i1 > i0:
                    block_violations[block_name] = float(np.max(g_violation[i0:i1]))
        except Exception:
            pass

        if has_nan:
            failure_reason = "NaN"
        elif status == 3 or (np.isfinite(max_constraint_violation) and max_constraint_violation > 1e-3):
            failure_reason = "infeasibility"
        elif status in (1, 2, 4):
            failure_reason = "convergence_stall"
        else:
            failure_reason = "unknown"

        # Accept converged-feasible iterates even if ACADOS reports nonzero status.
        accepted_feasible_stall = (
            (not is_successful)
            and (not has_nan)
            and status in (1, 2, 4)
            and np.isfinite(max_constraint_violation)
            and max_constraint_violation <= 1e-6
        )
        if accepted_feasible_stall:
            is_successful = True
            status_message = "ACADOS_SUCCESS_FEASIBLE_STALL"
            failure_reason = "accepted_feasible_stall"

        if is_successful and np.isfinite(z_star).all() and np.isfinite(p_vec).all():
            try:
                grad_f = np.asarray(self.grad_f_fun(z_star, p_vec), dtype=float).reshape(-1)
                jac_g = np.asarray(self.jac_g_fun(z_star, p_vec), dtype=float)
                if np.isfinite(grad_f).all() and np.isfinite(jac_g).all():
                    lam_g_star = -np.linalg.lstsq(jac_g.T, grad_f, rcond=None)[0]
            except Exception:
                lam_g_star = np.zeros(self.ng, dtype=float)

        if not is_successful:
            print(
                f"[ACADOS ERROR] {status_message} | reason={failure_reason} | "
                f"max_constraint_violation={max_constraint_violation}"
            )
            print(f"[ACADOS ERROR] status_history={status_history}")
            if block_violations:
                print(
                    "[ACADOS ERROR] violation_by_block "
                    + " ".join(f"{k}={v:.6g}" for k, v in block_violations.items())
                )
        else:
            print(
                f"[ACADOS OK] {status_message} | "
                f"max_constraint_violation={max_constraint_violation}"
            )

        if is_successful:
            self.init_sol = z_star.copy()

        return {
            "z": z_star,
            "lam_g": lam_g_star,
            "p": p_vec,
            "theta0": float(theta0),
            "status": status_message,
            "status_code": status,
            "success": is_successful,
            "failure_reason": failure_reason,
            "max_constraint_violation": max_constraint_violation,
        }

    def outer_loss_and_grad_q(self, init_state, dyn_param, q, theta0=None):
        out = self.solve(init_state, dyn_param, q, theta0=theta0)
        if not out["success"]:
            raise RuntimeError(
                "Inner ACADOS solve failed: "
                f"{out['status']} | reason={out.get('failure_reason', 'unknown')} | "
                f"max_constraint_violation={out.get('max_constraint_violation', np.nan)}"
            )

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
        q_np  = np.asarray(q, dtype=float).reshape(-1)
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
