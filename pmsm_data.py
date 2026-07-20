import numpy as np
import matplotlib.pyplot as plt


def load_raw_results(filepath):
    """Loads the raw, untouched OCP output saved by pmsm_solver.save_raw_results.

    Parameters
    ----------
    filepath : str
        Path to the .npz file written by the solver script.

    Returns
    -------
    results : dict
        Dictionary with keys: t, theta_m, omega_m, P, id, iq, vd, vq, dt
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def wrap_to_2pi(theta):
    """Wrap angle(s) into [0, 2*pi)."""
    return np.mod(theta, 2 * np.pi)


def add_start_padding(t, signals, pad_duration, dt):
    """Prepends `pad_duration` seconds of zero-valued samples to the start
    of every signal, and shifts t so the original t=0 sample lands at
    t=pad_duration.

    Parameters
    ----------
    t : np.ndarray
        Original time vector (untouched OCP output).
    signals : dict[str, np.ndarray]
        Dictionary of signal name -> array, each same length as t.
    pad_duration : float
        Duration (s) of zero-padding to prepend.
    dt : float
        Time step used to build the padding samples.

    Returns
    -------
    t_out : np.ndarray
        Time vector with padding prepended and original samples shifted.
    signals_out : dict[str, np.ndarray]
        Signals with zero-padding prepended.
    """
    n_pad = int(round(pad_duration / dt))
    t_pad = np.arange(n_pad) * dt
    t_out = np.concatenate([t_pad, t + pad_duration])

    signals_out = {}
    for name, arr in signals.items():
        arr = np.asarray(arr).ravel()
        signals_out[name] = np.concatenate([np.zeros(n_pad), arr])

    return t_out, signals_out


def add_end_hold(t, signals, hold_duration, dt, hold_names=None):
    """Appends `hold_duration` seconds to t (continuing at the same dt),
    holding the listed signals flat at their final value. Signals not
    listed in `hold_names` are left at their original length.

    Parameters
    ----------
    t : np.ndarray
        Time vector to extend.
    signals : dict[str, np.ndarray]
        Dictionary of signal name -> array, each same length as t.
    hold_duration : float
        Duration (s) to append.
    dt : float
        Time step used to build the extra samples.
    hold_names : iterable[str], optional
        Names of signals to hold flat at the end. Defaults to
        {'omega_m', 'id', 'iq'}. Signals not in this set are returned
        untouched (not extended).

    Returns
    -------
    t_out : np.ndarray
        Time vector with hold samples appended.
    signals_out : dict[str, np.ndarray]
        Signals with hold values appended where applicable.
    """
    if hold_names is None:
        hold_names = {'omega_m', 'id', 'iq'}
    else:
        hold_names = set(hold_names)

    n_hold = int(round(hold_duration / dt))
    t_hold = t[-1] + np.arange(1, n_hold + 1) * dt
    t_out = np.concatenate([t, t_hold])

    signals_out = {}
    for name, arr in signals.items():
        arr = np.asarray(arr).ravel()
        if name in hold_names:
            hold_vals = np.full(n_hold, arr[-1])
            signals_out[name] = np.concatenate([arr, hold_vals])
        else:
            signals_out[name] = arr

    return t_out, signals_out


def save_results_matlab(filepath, t, signals, fmt='%.8g'):
    """Writes a plain-text file with MATLAB-style row-vector assignments,
    one variable per line, e.g.:

        t = [0 0.1 0.2 ... 10.0];
        theta_m = [0 0.0123 ... ];
        id = [0 1.02 ... ];

    This function performs NO padding or holding itself; pass it already
    processed t/signals (e.g. via add_start_padding / add_end_hold).

    Parameters
    ----------
    filepath : str
        Destination path for the text file.
    t : np.ndarray
        Time vector to write.
    signals : dict[str, np.ndarray]
        Dictionary of signal name -> array to write, in insertion order.
    fmt : str
        Numeric format string for each value.
    """
    lines = [f't = [{" ".join(fmt % v for v in np.asarray(t).ravel())}];']
    for name, arr in signals.items():
        arr = np.asarray(arr).ravel()
        values_str = ' '.join(fmt % v for v in arr)
        lines.append(f'{name} = [{values_str}];')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return filepath


def plot_pmsm_results(t, theta_m_sol, omega_sol, P_sol, id_sol, iq_sol,
                       params=None, savepath='pmsm_optimal_control.png'):
    """
    Produces a clean multi-panel figure:
      1) id, iq currents (with optional Imax reference lines)
      2) omega_m (speed)
      3) P (estimation variance)
    """
    theta_wrapped = wrap_to_2pi(theta_m_sol)  # noqa: F841 (kept for parity/future use)

    fig, axs = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    # --- id, iq ---
    ax = axs[0]
    ax.plot(t, id_sol, label='$i_d$', color='tab:orange')
    ax.plot(t, iq_sol, label='$i_q$', color='tab:green')
    if params is not None and hasattr(params, 'Imax'):
        ax.axhline(params.Imax, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(-params.Imax, color='k', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='$\\pm I_{max}$')
    ax.set_ylabel('Current [A]')
    ax.set_title('Stator currents (dq frame)')
    ax.legend(loc='upper right')

    # --- omega_m ---
    ax = axs[1]
    ax.plot(t, omega_sol, color='tab:red')
    ax.set_ylabel('$\\omega_m$ [rad/s]')
    ax.set_title('Rotor speed')

    # --- P (variance) ---
    ax = axs[2]
    ax.axhline(0, color='k', linewidth=0.5)
    ax.plot(t, P_sol, color='tab:purple', label='$P$ (variance)')
    ax.set_ylabel('Variance [rad$^2$]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Estimation error covariance')
    ax.legend(loc='upper right')

    for a in axs:
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()

    return fig, axs


if __name__ == "__main__":
    # ---- User-specified variables ----
    RAW_RESULTS_PATH = 'pmsm_raw_results.npz'
    MATLAB_OUTPUT_PATH = 'pmsm_results.txt'
    PLOT_OUTPUT_PATH = 'pmsm_optimal_control.png'

    PAD_DURATION = 3.0    # seconds of zero-padding prepended at the start
    HOLD_DURATION = 5.0   # seconds of held-flat values appended at the end
    PLOT = True          # whether to plot the results (True) or not (False)

    # Which signals get zero-padded at the start (all of them, by default)
    PAD_SIGNAL_NAMES = ['omega_m', 'id', 'iq']
    # Which signals get held flat at the end (matches original behavior)
    HOLD_SIGNAL_NAMES = {'omega_m', 'id', 'iq'}

    # ---- Fetch the untouched OCP output ----
    raw = load_raw_results(RAW_RESULTS_PATH)
    t_raw = raw['t']
    dt = float(raw['dt'])
    signals_raw = {name: raw[name] for name in PAD_SIGNAL_NAMES if name in raw}

    # ---- Apply padding and hold time ----
    t_padded, signals_padded = add_start_padding(t_raw, signals_raw, PAD_DURATION, dt)
    t_final, signals_final = add_end_hold(t_padded, signals_padded, HOLD_DURATION, dt,
                                           hold_names=HOLD_SIGNAL_NAMES)

    # ---- Export to MATLAB-style text file ----
    save_results_matlab(MATLAB_OUTPUT_PATH, t_final, signals_final)
    print(f"MATLAB-style results saved to {MATLAB_OUTPUT_PATH}")

    # ---- Plot using the untouched (unpadded) data, as in the original script ----
    if PLOT:
        plot_pmsm_results(t_raw, raw['theta_m'], raw['omega_m'], raw['P'],
                          raw['id'], raw['iq'], params=None, savepath=PLOT_OUTPUT_PATH)
        print(f"Plot saved to {PLOT_OUTPUT_PATH}")
    else:
        print("Plotting skipped (PLOT=False).")