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

def build_dynamics_function_with_uncertainty(params, ke, sigma_base, k_bemf_noise, k_gain, Q_process):
    x_sym = ca.SX.sym('x', 5)   # [id, iq, omega_m, theta_m, P]
    u_sym = ca.SX.sym('u', 2)

    id, iq, omega_m, theta_m, P = ca.vertsplit(x_sym)

    # Original plant dynamics (unchanged)
    dx_plant = pmsm_dynamics(x_sym[0:4], u_sym, params)   # your existing 4-state dynamics function

    # BEMF signal and speed-proportional noise
    signal = ke * omega_m
    noise = sigma_base + k_bemf_noise * ca.fabs(omega_m)
    snr = signal / noise
    info_rate = k_gain * snr**2

    P_dot = -info_rate * P + Q_process

    dx_aug = ca.vertcat(dx_plant, P_dot)
    return ca.Function('f_dyn_uncertainty', [x_sym, u_sym], [dx_aug])

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
    params = PMSMParams()
    T_HORIZON = 3.0
    DT = 1e-1
    N = int(round(T_HORIZON / DT))
    RK4_SUBSTEPS = 500   # single smooth ODE, should be much less stiff than before

    TARGET_SPEED = params.N * 0.1
    P0 = 1.0**2         # initial uncertainty (rad^2) — matches your old THETA_HAT_INIT_ERROR=1.0

    # Tunable noise/signal model
    ke = params.lambda_m * params.p          # BEMF constant, adjust to your actual definition
    sigma_base = 0.01
    k_bemf_noise = 0.005      # <-- the "BEMF noise proportional to speed" coefficient
    k_gain = 1.0
    Q_process = 1e-4          # small floor so P never hits exactly zero (avoids /0 issues elsewhere)

    f_dyn_unc = build_dynamics_function_with_uncertainty(
        params, ke, sigma_base, k_bemf_noise, k_gain, Q_process)
    rk4_integrator = build_rk4_integrator_nd(f_dyn_unc, DT, RK4_SUBSTEPS, nx=5, nu=2)

    opti = ca.Opti()
    X = opti.variable(5, N + 1)   # [id, iq, omega_m, theta_m, P]
    U = opti.variable(2, N)

    id, iq, omega_m, theta_m, P = X[0,:], X[1,:], X[2,:], X[3,:], X[4,:]
    vd, vq = U[0,:], U[1,:]

    for k in range(N):
        x_next = rk4_integrator(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)

    opti.subject_to(X[0:4, 0] == ca.vertcat(0, 0, 0, 0))
    opti.subject_to(P[0] == P0)
    opti.subject_to(omega_m[-1] == TARGET_SPEED)
    opti.subject_to(vd**2 + vq**2 <= params.Vmax**2)
    opti.subject_to(id**2 + iq**2 <= params.Imax**2)
    opti.subject_to(P >= 0)   # variance can't go negative

    # Objective: minimize uncertainty as fast as possible — integral of P over horizon
    opti.minimize(ca.sumsqr(P) * DT)   # or ca.sum1(P)*DT, or weight terminal P more heavily

    opti.set_initial(omega_m, np.linspace(0, TARGET_SPEED, N + 1))
    opti.set_initial(P, P0 * np.exp(-np.linspace(0, 3, N+1)))
    opti.set_initial(vq, params.Vmax * 0.1)

    opti.solver('ipopt', {'ipopt.print_level': 5, 'print_time': False, 'ipopt.max_iter': 3000})

    try:
        sol = opti.solve()
    except RuntimeError:
        print("Solve failed, dumping debug values...")
        print("theta_m:", opti.debug.value(theta_m))
        print("omega_m:", opti.debug.value(omega_m))
        print(opti.debug.show_infeasibilities())
        raise

    # --- Extract & plot ---
    t = np.linspace(0, T_HORIZON, N + 1)
    theta_m_sol   = sol.value(theta_m)
    omega_sol     = sol.value(omega_m)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(t, theta_m_sol, label='$\\theta_m$ (true)')
    axs[0].set_ylabel('Position [rad]')
    axs[0].legend()

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