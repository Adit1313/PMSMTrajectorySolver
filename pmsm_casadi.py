import numpy as np
import casadi as ca


# Named motor presets, mapped onto PMSMParams' fields:
#   R=Rs, Ld, Lq, lambda_m=FluxPM, p, J, B, N=N_max (rated/max speed), Imax=I_rated.
# Vmax is bus voltage, not a motor parameter -- not given in a motor's datasheet
# fields above, so both presets keep the same 24.0 V default unless overridden.
MOTOR_PRESETS = {
    'BLY172S': dict(R=0.40, Ld=0.60e-3, Lq=0.60e-3, lambda_m=6.838e-3,
                     p=4, J=4.802e-6, B=1.0e-6, N=4000, Imax=3.5, Vmax=24.0),
    'Teknic2310P': dict(R=0.36, Ld=2.0e-4, Lq=2.0e-4, lambda_m=6.4e-3,
                         p=4, J=7.0616e-6, B=2.6369e-6, N=6000, Imax=7.1, Vmax=24.0),
}


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

    @classmethod
    def for_motor(cls, name):
        """Builds PMSMParams from a named preset in MOTOR_PRESETS, e.g.
        PMSMParams.for_motor('Teknic2310P')."""
        if name not in MOTOR_PRESETS:
            raise ValueError(f"Unknown motor '{name}'. Available: {list(MOTOR_PRESETS)}")
        return cls(**MOTOR_PRESETS[name])


def pmsm_dynamics(x, u, params):
    """Builds the PMSM dynamics in the form dx/dt = f(x, u) using CasADi.

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

    # (Ld - Lq)*id*iq is the reluctance-torque term; this motor is surface-mount
    # (Ld == Lq), so it's always zero here and torque comes only from iq.
    Te = (3 / 2) * p * (lambda_m * iq + (Ld - Lq) * id * iq)  # Electromagnetic torque
    domega_m = (Te - B * omega_m) / J  # Mechanical dynamics
    dtheta_m = omega_m  # Rotor position dynamics

    return ca.vertcat(did, diq, domega_m, dtheta_m)


def build_dynamics_function(params, sigma=None, Q_process=None):
    """Builds a CasADi function for the PMSM dynamics dx/dt = f(x, u).

    By default this builds the plain 4-state plant [id, iq, omega_m, theta_m].
    Passing both sigma and Q_process augments it with a 5th state, P (the
    sensorless estimator's rotor-angle variance), giving the 5-state
    [id, iq, omega_m, theta_m, P] used by the OCP: P_dot = -info_rate*P +
    Q_process, so a higher BEMF SNR (see info_rate()) drives the estimate's
    uncertainty down faster, while Q_process is a constant floor that keeps
    it from decaying to zero.

    Parameters
    ----------
    params : PMSMParams
        Physical parameters of the PMSM
    sigma : float, optional
        Gaussian measurement-noise standard deviation (see info_rate()).
        Enables the P state.
    Q_process : float, optional
        Process-noise floor for the P state. Enables the P state.

    Returns
    -------
    f : casadi.Function
        CasADi function representing the dynamics dx/dt = f(x, u)
    """
    with_uncertainty = sigma is not None and Q_process is not None
    nx = 5 if with_uncertainty else 4

    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', 2)  # Input vector [vd, vq]

    dx_plant = pmsm_dynamics(x[0:4], u, params)  # [id, iq, omega_m, theta_m] dynamics, unchanged either way

    if with_uncertainty:
        omega_m, P = x[2], x[4]
        P_dot = -info_rate(omega_m, params, sigma) * P + Q_process
        xdot = ca.vertcat(dx_plant, P_dot)
    else:
        xdot = dx_plant

    f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f


def measurement_function(x, params):
    id, iq, omega_m, theta_m = x[0], x[1], x[2], x[3]
    i_alpha = id * ca.cos(theta_m) - iq * ca.sin(theta_m)
    i_beta = id * ca.sin(theta_m) + iq * ca.cos(theta_m)
    return ca.vertcat(i_alpha, i_beta)


def estimated_dq_currents(i_alpha, i_beta, theta_hat):
    """Controller's reconstructed dq currents, using ONLY theta_hat."""
    id_hat = i_alpha * ca.cos(theta_hat) + i_beta * ca.sin(theta_hat)
    iq_hat = -i_alpha * ca.sin(theta_hat) + i_beta * ca.cos(theta_hat)
    return id_hat, iq_hat


def info_rate(omega_m, params, sigma, k_hfi=0.0):
    """Rate at which the estimator gains information about rotor angle.

    Noise model: the BEMF-based observer measures a signal of amplitude
      A = params.lambda_m * params.p * omega_m   -- BEMF magnitude [V],
          zero at standstill (no BEMF-based info without motion),
    corrupted by simple additive white Gaussian noise, n ~ N(0, sigma**2),
    with a single, constant standard deviation sigma [V] -- the standard
    textbook sensor/ADC noise assumption, regardless of speed.

    The Fisher information rate for estimating a phase from a sinusoidal
    signal of amplitude A in Gaussian noise of variance sigma**2 is A**2 /
    sigma**2, giving:

      info_rate = (A / sigma)**2 = (params.lambda_m * params.p * omega_m / sigma)**2

    k_hfi is an optional constant, speed-independent addition (e.g. a
    saliency-based high-frequency-injection estimator, which works even at
    standstill, unlike the BEMF term above).

    Works with both CasADi symbols and plain numbers/numpy arrays, so this
    is the single source of truth shared between the OCP dynamics below and
    any offline comparison of alternative speed trajectories.
    """
    signal = params.lambda_m * params.p * omega_m
    return (signal / sigma) ** 2 + k_hfi


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


def wrap_to_2pi(theta):
    """Wrap angle(s) into [0, 2*pi)."""
    return np.mod(theta, 2 * np.pi)


def solve_pmsm_ocp(params, T_HORIZON=3.0, DT=1e-1, RK4_SUBSTEPS=100,
                    P0=1.0 ** 2, sigma=20.0, Q_process=1e-4):
    """Sets up and solves the PMSM optimal control problem.

    sigma is the Gaussian measurement-noise standard deviation (see
    info_rate() in this module) that drives how fast the estimation
    variance P can be reduced by spinning up.

    Returns
    -------
    results : dict
        Dictionary containing the raw, untouched OCP output:
        t, theta_m, omega_m, P, id, iq, vd, vq
    """
    N = int(round(T_HORIZON / DT))
    TARGET_SPEED = params.N * 0.125

    f_dyn_unc = build_dynamics_function(params, sigma=sigma, Q_process=Q_process)
    rk4_integrator = build_rk4_integrator_nd(f_dyn_unc, DT, RK4_SUBSTEPS, nx=5, nu=2)

    opti = ca.Opti()
    X = opti.variable(5, N + 1)  # [id, iq, omega_m, theta_m, P]
    U = opti.variable(2, N)

    id, iq, omega_m, theta_m, P = X[0, :], X[1, :], X[2, :], X[3, :], X[4, :]
    vd, vq = U[0, :], U[1, :]

    for k in range(N):
        x_next = rk4_integrator(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    opti.subject_to(X[0:4, 0] == ca.vertcat(0, 0, 0, 0))
    opti.subject_to(P[0] == P0)
    opti.subject_to(omega_m[-1] == TARGET_SPEED)
    # info_rate only depends on omega_m**2, so without these bounds the
    # solver can rack up "information" by oscillating speed direction
    # wildly instead of ramping up to the target and holding.
    opti.subject_to(opti.bounded(0, omega_m, TARGET_SPEED))
    # Ld == Lq (surface-mount PM), so id contributes zero torque and is
    # otherwise unconstrained by the objective; without pinning it, the
    # solver parks it at an arbitrary value (e.g. the current limit) for
    # no benefit. Standard id=0 (max torque per amp) control for SPM motors.
    opti.subject_to(id == 0)
    opti.subject_to(vd ** 2 + vq ** 2 <= params.Vmax ** 2)
    opti.subject_to(id ** 2 + iq ** 2 <= params.Imax ** 2)
    opti.subject_to(P >= 0)  # variance can't go negative

    # Objective: minimize uncertainty as fast as possible - integral of P over horizon
    opti.minimize(ca.sumsqr(P) * DT)

    opti.set_initial(omega_m, np.linspace(0, TARGET_SPEED, N + 1))
    opti.set_initial(P, P0 * np.exp(-np.linspace(0, 3, N + 1)))
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

    t = np.linspace(0, T_HORIZON, N + 1)
    results = {
        't': t,
        'theta_m': sol.value(theta_m),
        'omega_m': sol.value(omega_m),
        'P': sol.value(P),
        'id': sol.value(id),
        'iq': sol.value(iq),
        'vd': sol.value(vd),
        'vq': sol.value(vq),
        'dt': DT,
    }
    return results


def save_raw_results(filepath, results):
    """Saves the raw, untouched OCP output to an .npz file.

    Parameters
    ----------
    filepath : str
        Destination path (should end in .npz)
    results : dict
        Dictionary as returned by solve_pmsm_ocp
    """
    np.savez(filepath, **results)
    return filepath


if __name__ == "__main__":
    MOTOR = 'BLY172S'  # or 'Teknic2310P' -- see MOTOR_PRESETS
    params = PMSMParams.for_motor(MOTOR)
    results = solve_pmsm_ocp(params, T_HORIZON=3.0, DT=1e-1, RK4_SUBSTEPS=500)
    save_raw_results('pmsm_raw_results.npz', results)
    print(f"Raw OCP results saved to pmsm_raw_results.npz (motor={MOTOR})")