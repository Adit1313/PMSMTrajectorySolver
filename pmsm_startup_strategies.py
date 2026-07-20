"""Compares startup strategies against the OCP-optimal spin-up trajectory,
using the same uncertainty (P) model from pmsm_casadi.py.

P_dot only depends on info_rate(omega_m, k_info, k_hfi) -- not on how omega_m
was achieved -- so alternative strategies are compared by prescribing a speed
trajectory omega_m(t) directly and integrating the same P dynamics, rather
than re-solving an optimization problem for each one.

Strategies compared
--------------------
1. optimal           -- the OCP solution from solve_pmsm_ocp (BEMF info only).
2. linear_ramp        -- naive constant-slope speed ramp over the full horizon
                         (BEMF info only). Baseline for "no clever startup".
3. align_then_ramp    -- rotor alignment: hold omega_m=0 for T_ALIGN (BEMF
                         info_rate is 0 while stationary, so P cannot decrease
                         during this phase -- it can only drift up via
                         Q_process), then a discrete reset P -> P_ALIGN at
                         T_ALIGN (representing the angle now being known from
                         the DC-injection lock), then ramp for the rest of the
                         horizon.
4. ramp_with_hfi      -- same speed profile as linear_ramp, but info_rate gets
                         a constant k_hfi added on top of the BEMF term,
                         representing a saliency-based high-frequency-
                         injection estimator that works independent of speed.
                         CAVEAT: HFI relies on Ld != Lq saliency; this model
                         uses Ld == Lq (surface-mount PM), so a real HFI
                         signal would be ~0 here. k_hfi is an idealized
                         stand-in for what a salient (e.g. IPM) design's HFI
                         estimator would buy you -- not derived from this
                         motor's actual (Ld, Lq).
5. align_ramp_hfi     -- alignment reset (3) and HFI (4) combined, to see the
                         combined ceiling.

Feasibility note: strategies 2-5 prescribe omega_m(t) directly rather than
simulating currents/voltages. A rough inverse-dynamics check (required iq
from the prescribed acceleration, via Te = 1.5*p*lambda_m*iq since id=0)
confirms every profile here stays far under Imax, since J is tiny -- so the
comparison is apples-to-apples with the OCP's actual current-constrained
trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

from pmsm_casadi import PMSMParams, info_rate, solve_pmsm_ocp

# ---- Shared config (must match the OCP run for a fair comparison) ----
T_HORIZON = 3.0
DT = 1e-1
RK4_SUBSTEPS = 500
P0 = 1.0 ** 2
K_INFO = 2e-6
Q_PROCESS = 1e-4

# Rotor-alignment strategy
T_ALIGN = 0.5     # seconds spent aligning before ramping
P_ALIGN = 0.05     # P snapped to this value right after alignment

# High-frequency-injection strategy
K_HFI = 0.2       # constant info rate, independent of speed (see caveat above)

# Colors: fixed categorical order (dataviz skill palette), assigned by
# strategy identity, not by rank/value.
COLORS = {
    'optimal': '#2a78d6',          # blue
    'linear_ramp': '#008300',      # green
    'align_then_ramp': '#e87ba4',  # magenta
    'ramp_with_hfi': '#eda100',    # yellow
    'align_ramp_hfi': '#1baf7a',   # aqua
}
LABELS = {
    'optimal': 'Optimal (OCP)',
    'linear_ramp': 'Linear ramp',
    'align_then_ramp': 'Align + ramp',
    'ramp_with_hfi': 'Ramp + HFI',
    'align_ramp_hfi': 'Align + ramp + HFI',
}


def linear_ramp_omega(t, target_speed, t_start, t_end):
    """Ramps linearly from 0 at t_start to target_speed at t_end, holding
    flat outside that window."""
    frac = np.clip((t - t_start) / (t_end - t_start), 0.0, 1.0)
    return target_speed * frac


def integrate_uncertainty(omega_ref, T_HORIZON, DT, RK4_SUBSTEPS, P0,
                           k_info, q_process, k_hfi=0.0,
                           reset_time=None, reset_value=None):
    """RK4-integrates P_dot = -info_rate(omega_ref(t), k_info, k_hfi)*P +
    q_process for a prescribed speed profile omega_ref(t), with an optional
    one-time discrete reset of P at reset_time.

    Returns the coarse (N+1)-point grid (t, P) matching the OCP's DT.
    """
    N = int(round(T_HORIZON / DT))
    dt_sub = DT / RK4_SUBSTEPS

    t_grid = np.linspace(0, T_HORIZON, N + 1)
    P_grid = np.empty(N + 1)
    P_grid[0] = P0

    def p_dot(t, P):
        return -info_rate(omega_ref(t), k_info, k_hfi) * P + q_process

    P = P0
    t = 0.0
    reset_pending = reset_time is not None
    for k in range(N):
        for _ in range(RK4_SUBSTEPS):
            k1 = p_dot(t, P)
            k2 = p_dot(t + dt_sub / 2, P + dt_sub / 2 * k1)
            k3 = p_dot(t + dt_sub / 2, P + dt_sub / 2 * k2)
            k4 = p_dot(t + dt_sub, P + dt_sub * k3)
            P = P + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt_sub
            if reset_pending and t >= reset_time:
                P = reset_value
                reset_pending = False
        P_grid[k + 1] = P
    return t_grid, P_grid


def check_feasibility(name, t, omega_m, params):
    """Rough inverse-dynamics sanity check: back out the iq (id=0) that
    would be required to produce this prescribed omega_m(t), and flag it if
    it would exceed Imax."""
    domega_dt = np.gradient(omega_m, t)
    Te_required = params.J * domega_dt + params.B * omega_m
    iq_required = Te_required / (1.5 * params.p * params.lambda_m)
    peak = np.max(np.abs(iq_required))
    status = "OK" if peak <= params.Imax else "EXCEEDS Imax!"
    print(f"  [{name}] peak |iq| required ~= {peak:.4f} A ({status}, Imax={params.Imax} A)")


def main():
    params = PMSMParams()
    TARGET_SPEED = params.N * 0.125

    strategies = {}

    # 1. Optimal (OCP) -- reuse the actual solver, BEMF info only.
    ocp_results = solve_pmsm_ocp(params, T_HORIZON=T_HORIZON, DT=DT,
                                  RK4_SUBSTEPS=RK4_SUBSTEPS, P0=P0,
                                  k_info=K_INFO, Q_process=Q_PROCESS)
    strategies['optimal'] = {'t': ocp_results['t'], 'omega_m': ocp_results['omega_m'],
                              'P': ocp_results['P']}

    # 2. Naive linear ramp over the whole horizon, BEMF info only.
    def omega_linear(t):
        return linear_ramp_omega(t, TARGET_SPEED, 0.0, T_HORIZON)

    t_grid, P_grid = integrate_uncertainty(
        omega_linear, T_HORIZON, DT, RK4_SUBSTEPS, P0, K_INFO, Q_PROCESS)
    strategies['linear_ramp'] = {'t': t_grid, 'omega_m': omega_linear(t_grid), 'P': P_grid}

    # 3. Align (hold + discrete P reset) then ramp for the remaining time.
    def omega_align(t):
        return linear_ramp_omega(t, TARGET_SPEED, T_ALIGN, T_HORIZON)

    t_grid, P_grid = integrate_uncertainty(
        omega_align, T_HORIZON, DT, RK4_SUBSTEPS, P0, K_INFO, Q_PROCESS,
        reset_time=T_ALIGN, reset_value=P_ALIGN)
    strategies['align_then_ramp'] = {'t': t_grid, 'omega_m': omega_align(t_grid), 'P': P_grid}

    # 4. Same naive ramp, plus a constant HFI info contribution.
    t_grid, P_grid = integrate_uncertainty(
        omega_linear, T_HORIZON, DT, RK4_SUBSTEPS, P0, K_INFO, Q_PROCESS,
        k_hfi=K_HFI)
    strategies['ramp_with_hfi'] = {'t': t_grid, 'omega_m': omega_linear(t_grid), 'P': P_grid}

    # 5. Alignment reset + HFI combined, on top of the align-then-ramp profile.
    t_grid, P_grid = integrate_uncertainty(
        omega_align, T_HORIZON, DT, RK4_SUBSTEPS, P0, K_INFO, Q_PROCESS,
        k_hfi=K_HFI, reset_time=T_ALIGN, reset_value=P_ALIGN)
    strategies['align_ramp_hfi'] = {'t': t_grid, 'omega_m': omega_align(t_grid), 'P': P_grid}

    # ---- Feasibility check (rough) ----
    print("Feasibility check (id=0, back out iq from prescribed omega_m(t)):")
    for name, data in strategies.items():
        check_feasibility(name, data['t'], data['omega_m'], params)

    # ---- Summary table ----
    print(f"\n{'Strategy':<22}{'Final P':>12}{'Integral P^2*dt':>20}")
    for name, data in strategies.items():
        final_P = data['P'][-1]
        cost = np.sum(data['P'] ** 2) * DT
        print(f"{LABELS[name]:<22}{final_P:>12.4f}{cost:>20.5f}")

    # ---- Plot ----
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axs[0]
    for name, data in strategies.items():
        ax.plot(data['t'], data['omega_m'], color=COLORS[name], linewidth=1.8,
                 label=LABELS[name])
    ax.set_ylabel('$\\omega_m$ [rad/s]')
    ax.set_title('Rotor speed by startup strategy')
    ax.legend(loc='lower right', fontsize=8)

    ax = axs[1]
    for name, data in strategies.items():
        ax.plot(data['t'], data['P'], color=COLORS[name], linewidth=1.8,
                 label=LABELS[name])
    ax.set_ylabel('Variance [rad$^2$]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Estimation error covariance (P) by startup strategy')
    ax.legend(loc='upper right', fontsize=8)

    for a in axs:
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pmsm_startup_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
