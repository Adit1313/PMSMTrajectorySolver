import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class PMSMParams:
    """Container for PMSM physical parameters (plain numbers, not symbolic)."""

    def __init__(self, R=0.40, Ld=0.60e-3, Lq=0.60e-3, lambda_m=6.838e-3,
                 p=4, J=4.802e-6, B=1.0e-6, N=4000, Imax=3.5, Vmax=24.0):
        self.R = R
        self.Ld = Ld
        self.Lq = Lq
        self.lambda_m = lambda_m
        self.p = p
        self.J = J
        self.B = B
        self.N = N
        self.Imax = Imax
        self.Vmax = Vmax

def pmsm_dynamics(x, u, params):
    """ Builds the PMSM dynamics in the form dx/dt = f(x, u) using CasADi.
    
    
    Parameters
    ----------
    x : casadi.SX
        State vector [i_d, i_q, omega_m, theta_m]
    u : casadi.SX
        Input vector [v_d, v_q]
    params : PMSMParams
        Physical parameters of the PMSM

    Returns
    -------
    xdot : casadi.SX
        Time derivative of the state vector
    """
    id, iq, omega_m, theta_m = x[0], x[1], x[2], x[3]
    vd, vq = u[0], u[1]

    R, Ld, Lq, lambda_m, p, J, B = params.R, params.Ld, params.Lq, params.lambda_m, params.p, params.J, params.B

    omega_e = p * omega_m  # Electrical angular velocity
    did = (vd - R * id + Lq * omega_e * iq) / Ld
    diq = (vq - R * iq - Ld * omega_e * id - lambda_m * omega_e) / Lq

    Te = (3 / 2) * p * (lambda_m * iq + (Ld - Lq) * id * iq)  # Electromagnetic torque
    domega_m = (Te - B * omega_m) / J  # Mechanical dynamics
    dtheta_m = omega_m  # Rotor position dynamics

    return ca.vertcat(did, diq, domega_m, dtheta_m)

def build_dynamics_function(params):
    """Builds a CasADi function for the PMSM dynamics.

    Parameters
    ----------
    params : PMSMParams
        Physical parameters of the PMSM

    Returns
    -------
    f : casadi.Function
        CasADi function representing the dynamics dx/dt = f(x, u)
    """
    x = ca.SX.sym('x', 4)  # State vector [id, iq, omega_m, theta_m]
    u = ca.SX.sym('u', 2)  # Input vector [vd, vq]
    xdot = pmsm_dynamics(x, u, params)
    f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f

def measurement_function(x, params):
    id, iq, omega_m, theta_m = x[0], x[1], x[2], x[3]
    i_alpha = id * ca.cos(theta_m) - iq * ca.sin(theta_m)
    i_beta  = id * ca.sin(theta_m) + iq * ca.cos(theta_m)
    return ca.vertcat(i_alpha, i_beta)

def estimated_dq_currents(i_alpha, i_beta, theta_hat):
    """Controller's reconstructed dq currents, using ONLY theta_hat."""
    id_hat =  i_alpha * ca.cos(theta_hat) + i_beta * ca.sin(theta_hat)
    iq_hat = -i_alpha * ca.sin(theta_hat) + i_beta * ca.cos(theta_hat)
    return id_hat, iq_hat

def build_augmented_dynamics(f_dyn, params, Kp_pll, Ki_pll):
    """
    Builds the 6-state augmented dynamics:
    [id, iq, omega_m, theta_m, theta_hat, omega_hat]

    theta_hat / omega_hat follow a standard PI-PLL (VCO) structure,
    driven only by a measurement-derived error signal (id_hat), never
    by the true state directly.
    """
    x_sym = ca.SX.sym('x', 6)
    u_sym = ca.SX.sym('u', 2)

    id, iq, omega_m, theta_m, theta_hat, omega_hat = ca.vertsplit(x_sym)

    # True plant dynamics (unchanged, reuse existing f_dyn on first 4 states)
    dx_plant = f_dyn(x_sym[0:4], u_sym)

    # Measurement chain: true currents -> reconstructed via theta_hat
    h = measurement_function(x_sym, params)
    i_alpha, i_beta = h[0], h[1]
    id_hat, iq_hat = estimated_dq_currents(i_alpha, i_beta, theta_hat)

    # Error signal — function of MEASUREMENT-DERIVED quantities only.
    # id_hat != 0 indicates estimation error (assuming id-reference = 0 control).
    error_signal = id_hat

    # PI-PLL / VCO structure
    theta_hat_dot = omega_hat + Kp_pll * error_signal
    omega_hat_dot = Ki_pll * error_signal

    dx_aug = ca.vertcat(dx_plant, theta_hat_dot, omega_hat_dot)

    f_dyn_aug = ca.Function('f_dyn_aug', [x_sym, u_sym], [dx_aug])
    return f_dyn_aug

def rk4_step(f_dyn, x, u, dt):
    """Performs a single Runge-Kutta 4th order integration step.

    Parameters
    ----------
    f_dyn : casadi.Function
        CasADi function representing the dynamics dx/dt = f(x, u)
    x : casadi.SX
        Current state vector
    u : casadi.SX
        Current input vector
    dt : float
        Time step for integration

    Returns
    -------
    x_next : casadi.SX
        State vector after time step dt
    """
    k1 = f_dyn(x, u)
    k2 = f_dyn(x + dt / 2 * k1, u)
    k3 = f_dyn(x + dt / 2 * k2, u)
    k4 = f_dyn(x + dt * k3, u)
    
    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

def build_rk4_integrator(f_dyn, dt, num_substeps):
    """Builds a CasADi function for RK4 integration of the PMSM dynamics.

    Parameters
    ----------
    f_dyn : casadi.Function
        CasADi function representing the dynamics dx/dt = f(x, u)
    dt : float
        Time step for integration
    num_substeps : int
        Number of substeps for RK4 integration

    Returns
    -------
    rk4_integrator : casadi.Function
        CasADi function that performs RK4 integration over time step dt
    """
    x = ca.SX.sym('x', 4)  # State vector [id, iq, omega_m, theta_m]
    u = ca.SX.sym('u', 2)  # Input vector [vd, vq]

    dt_sub = dt / num_substeps
    x_sub = x
    for _ in range(num_substeps):
        x_sub = rk4_step(f_dyn, x_sub, u, dt_sub)
    
    rk4_integrator = ca.Function('rk4_integrator', [x, u], [x_sub], ['x', 'u'], ['x_next'])
    return rk4_integrator

def build_rk4_integrator_nd(f_dyn, dt, num_substeps, nx, nu):
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    dt_sub = dt / num_substeps
    x_sub = x
    for _ in range(num_substeps):
        x_sub = rk4_step(f_dyn, x_sub, u, dt_sub)
    return ca.Function('rk4_integrator', [x, u], [x_sub], ['x', 'u'], ['x_next'])

if __name__ == "__main__":
    # Parameters
    params = PMSMParams()
    T_HORIZON = 3.0                 # s -- Simulation end time
    DT = 1e-1                       # s -- shooting interval
    N = int(round(T_HORIZON / DT))  # Number of shooting intervals
    RK4_SUBSTEPS = 500             # Number of RK4 substeps per shooting interval
    TARGET_SPEED = params.N * 0.1   # Target speed (10% of rated speed)
    THETA_HAT_INIT_ERROR = 1.0      # Initial error in theta_hat for the PLL observer
    zeta = 0.9
    settling_time = 0.2
    omega_n = 4.0 / (zeta * settling_time)
    Kp_pll = 1
    Ki_pll = 1
    
    # Dynamics and Integrator
    f_dyn = build_dynamics_function(params)               # your existing 4-state plant dynamics
    f_dyn_aug = build_augmented_dynamics(f_dyn, params, Kp_pll, Ki_pll)
    rk4_integrator = build_rk4_integrator_nd(f_dyn_aug, DT, RK4_SUBSTEPS, nx=6, nu=2)

    # Optimizer Config
    opti = ca.Opti()
    X = opti.variable(6, N + 1)     # [id, iq, omega_m, theta_m, theta_hat, omega_hat]
    U = opti.variable(2, N)         # [vd, vq]
    id = X[0, :]
    iq = X[1, :]
    omega_m = X[2, :]
    theta_m = X[3, :]
    theta_hat = X[4, :]
    omega_hat = X[5, :]
    vd = U[0, :]
    vq = U[1, :]

    # Dynamics constraints
    for k in range(N):
        x_next = rk4_integrator(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    x_test = np.array([0, 0, 0, 0, 1.0, 0])   # theta_hat starts off by 1 rad
    u_test = np.array([0, 20])                 # fixed vq

    x = x_test.copy()
    theta_err_trace = []
    x = x_test.copy()
    for k in range(N):
        x_prev = x.copy()
        x = np.array(rk4_integrator(x, u_test)).flatten()
        if np.any(np.isnan(x)):
            print(f"NaN first appears at step {k}")
            print("x before:", x_prev)
            print("x after: ", x)
            break


    # Conditions
    opti.subject_to(X[0:4, 0] == ca.vertcat(0, 0, 0, 0))  # Initial state: [id=0, iq=0, omega_m=0, theta_m=0]
    opti.subject_to(theta_hat[0] == THETA_HAT_INIT_ERROR)
    opti.subject_to(omega_hat[0] == 0)
    opti.subject_to(omega_m[-1] == TARGET_SPEED)  # Final speed constraint
    
    opti.subject_to(vd**2 + vq**2 <= params.Vmax**2)
    opti.subject_to(id**2 + iq**2 <= params.Imax**2)
    opti.subject_to(opti.bounded(-2*params.Imax, id, 2*params.Imax))
    opti.subject_to(opti.bounded(-2*params.Imax, iq, 2*params.Imax))

    omega_max_search = 2 * TARGET_SPEED    # generous headroom, not unlimited
    opti.subject_to(opti.bounded(-omega_max_search, omega_m, omega_max_search))
    opti.subject_to(opti.bounded(-omega_max_search, omega_hat, omega_max_search))

    theta_bound = 20.0   # rad — generous, but not infinite; tune based on expected excursions
    opti.subject_to(opti.bounded(-theta_bound, theta_m, theta_bound))
    opti.subject_to(opti.bounded(-theta_bound, theta_hat, theta_bound))

    theta_error = theta_m - theta_hat
    tracking_cost = ca.sumsqr(theta_error) * DT

    theta_error = theta_m - theta_hat            # full length N+1
    running_cost = ca.sumsqr(theta_error) * DT   # confirm this covers all N+1 nodes

    terminal_weight = 100.0
    terminal_cost = terminal_weight * theta_error[-1]**2

    smoothness_cost = ca.sumsqr(omega_m[1:] - omega_m[:-1])
    w_smooth = 1e-2
    opti.minimize(tracking_cost + terminal_cost + w_smooth * smoothness_cost)
    
    # Warm start guess
    # Build a full, dynamically consistent initial guess by forward simulation
    x_guess = np.zeros((6, N + 1))
    x_guess[:, 0] = [0, 0, 0, 0, THETA_HAT_INIT_ERROR, 0]
    u_guess = np.zeros((2, N))
    u_guess[1, :] = params.Vmax * 0.1   # constant vq guess

    for k in range(N):
        x_guess[:, k + 1] = np.array(rk4_integrator(x_guess[:, k], u_guess[:, k])).flatten()

    opti.set_initial(X, x_guess)
    opti.set_initial(U, u_guess)

    # Solver setup
    opti.solver('ipopt', {
        'ipopt.print_level': 5,
        'print_time': False,
        'ipopt.max_iter': 1000,
        'ipopt.tol': 1e-6,
    })


    try:
        sol = opti.solve()
    except RuntimeError:
        print("Solve failed, dumping debug values...")
        print("theta_m:", opti.debug.value(theta_m))
        print("theta_hat:", opti.debug.value(theta_hat))
        print("omega_m:", opti.debug.value(omega_m))
        print(opti.debug.show_infeasibilities())
        raise

    # --- Extract & plot ---
    t = np.linspace(0, T_HORIZON, N + 1)
    theta_m_sol   = sol.value(theta_m)
    theta_hat_sol = sol.value(theta_hat)
    omega_sol     = sol.value(omega_m)
    error_sol     = theta_m_sol - theta_hat_sol

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(t, theta_m_sol, label='$\\theta_m$ (true)')
    axs[0].plot(t, theta_hat_sol, '--', label='$\\hat{\\theta}$ (estimate)')
    axs[0].set_ylabel('Position [rad]')
    axs[0].legend()

    axs[1].plot(t, error_sol)
    axs[1].axhline(0, color='k', linewidth=0.5)
    axs[1].set_ylabel('$\\theta_m - \\hat{\\theta}$ [rad]')
    axs[1].set_title('Estimation error')

    axs[2].plot(t, omega_sol)
    axs[2].set_ylabel('$\\omega_m$ [rad/s]')
    axs[2].set_xlabel('Time [s]')

    for ax in axs:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()