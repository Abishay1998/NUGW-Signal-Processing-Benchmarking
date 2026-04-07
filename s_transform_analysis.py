"""
s_transform_analysis.py
-----------------------
S-transform (Stockwell 1996) hyperparameter optimisation for nonlinear
guided wave signals.

The standard S-transform is:
    S(τ, f) = ∫ x(t) · |f|/√(2π) · exp(-(τ-t)²f²/2) · exp(-2πift) dt

Implemented via the efficient FFT form:
    S(τ, f) = IFFT_τ[ X(α+f) · G(α, f) ]
where G(α, f) = exp(-2π²α²/f²) is the Gaussian window spectrum.

Sweep parameters
────────────────
  p_factor : Gaussian width scaling factor  — p ∈ {0.5, 1.0, 2.0, 4.0}
             Larger p → wider Gaussian → better frequency resolution
             Standard S-transform: p = 1.0
  f_lo_hz  : lower frequency bound for analysis  (not a free parameter,
             set by signal dominant frequency / 2)

Metrics (Stankovic 2001)
────────────────────────
  M_JPL  (eq. 2)  ↑ better  — optimisation objective
  RV₃    (eq. 4)  ↓ better  — reporting metric
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from load_signals import load_all

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

P_FACTORS: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
FREQ_MAX_MHZ: float = 3.0
N_FREQ_BINS: int = 256          # number of frequency slices in the S-matrix

SIGMA_F_BINS: float = 5.0
SIGMA_T_BINS: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    return 1.0 / (float(np.mean(np.diff(t_us))) * 1e-6)


def _stockwell(signal: np.ndarray, fs: float,
               p: float = 1.0,
               n_freq: int = N_FREQ_BINS,
               freq_max_hz: float | None = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the S-transform of *signal* using the FFT algorithm.

    Parameters
    ----------
    signal     : 1-D real array
    fs         : sampling frequency [Hz]
    p          : Gaussian width factor (1.0 = standard S-transform)
    n_freq     : number of equally-spaced frequency bins to evaluate
    freq_max_hz: upper frequency limit [Hz]; defaults to fs/2

    Returns
    -------
    freqs : array of length n_freq  [Hz]
    times : array of length N       [s]
    S     : complex array (n_freq × N)  — S-transform matrix
    """
    N         = len(signal)
    dt        = 1.0 / fs
    times     = np.arange(N) * dt
    f_max     = min(freq_max_hz or fs / 2, fs / 2)
    freqs     = np.linspace(0, f_max, n_freq + 1)[1:]   # exclude DC

    X         = np.fft.fft(signal)                       # shape (N,)
    alpha     = np.fft.fftfreq(N, d=dt)                  # shape (N,)

    S_matrix  = np.zeros((n_freq, N), dtype=complex)
    for i, f in enumerate(freqs):
        # Gaussian window in frequency domain: exp(-2π²α²/(p²f²))
        if abs(f) < 1e-10:
            S_matrix[i] = np.mean(signal) * np.ones(N)
            continue
        G       = np.exp(-2 * np.pi**2 * alpha**2 / (p**2 * f**2))
        # Shift X by f: X(α+f) implemented by rolling in frequency domain
        shift   = int(round(f / (fs / N)))
        X_shift = np.roll(X, -shift)
        S_matrix[i] = np.fft.ifft(X_shift * G)

    return freqs, times, S_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Q and mode-peak localisation
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_Q(shape: tuple[int, int], cf: int, ct: int,
                sigma_f: float, sigma_t: float) -> np.ndarray:
    nf, nt = shape
    df = np.arange(nf)[:, np.newaxis] - cf
    dt = np.arange(nt)[np.newaxis, :] - ct
    return np.exp(-df**2 / (2 * sigma_f**2) - dt**2 / (2 * sigma_t**2))


def _mode_tf_center(mode_signal: np.ndarray, fs: float,
                    p: float) -> tuple[int, int]:
    _, _, Sm = _stockwell(mode_signal, fs, p=p,
                          freq_max_hz=FREQ_MAX_MHZ * 1e6)
    power = np.abs(Sm) ** 2
    idx   = np.unravel_index(np.argmax(power), power.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _mjpl_global(signal: np.ndarray, fs: float, p: float,
                 mode_signal: np.ndarray) -> float:
    _, _, Sm = _stockwell(signal, fs, p=p, freq_max_hz=FREQ_MAX_MHZ * 1e6)
    rho  = np.abs(Sm) ** 2
    cf, ct = _mode_tf_center(mode_signal, fs, p)
    Q    = _gaussian_Q(rho.shape, cf, ct, SIGMA_F_BINS, SIGMA_T_BINS)
    Q2   = Q ** 2
    numer = np.sum(Q2 * rho**2)
    denom = np.sum(Q  * rho)
    if denom == 0:
        return 0.0
    return float(numer / (denom ** 2))


def _rv3(signal: np.ndarray, fs: float, p: float) -> float:
    _, _, Sm = _stockwell(signal, fs, p=p, freq_max_hz=FREQ_MAX_MHZ * 1e6)
    rho  = np.abs(Sm) ** 2
    vol  = np.sum(np.abs(rho))
    if vol == 0:
        return np.inf
    rho_norm = np.maximum(rho / vol, 1e-300)
    return float(-0.5 * np.log2(np.sum(rho_norm ** 3)))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_metrics(
    signal: np.ndarray,
    t_us: np.ndarray,
    mode_signal: np.ndarray,
    p_factors: list[float] = P_FACTORS,
) -> dict:
    """results[p] = {"mjpl": float, "rv3": float}"""
    fs = _fs_from_time(t_us)
    results: dict[float, dict] = {}
    for p in p_factors:
        try:
            results[p] = {
                "mjpl": _mjpl_global(signal, fs, p, mode_signal),
                "rv3":  _rv3(signal, fs, p),
            }
        except Exception as e:
            print(f"  [skip] p={p}: {e}")
    return results


def select_best(results: dict) -> dict:
    best_p, best_mjpl, best_rv3 = None, -np.inf, None
    for p, m in results.items():
        if m["mjpl"] > best_mjpl:
            best_mjpl = m["mjpl"]
            best_p    = p
            best_rv3  = m["rv3"]
    return {"p_factor": best_p, "mjpl": best_mjpl, "rv3": best_rv3}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(results: dict, best: dict, title_prefix: str) -> None:
    p_vals  = sorted(results.keys())
    mjpl_v  = [results[p]["mjpl"] for p in p_vals]
    rv3_v   = [results[p]["rv3"]  for p in p_vals]
    labels  = [str(p) for p in p_vals]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{title_prefix}  —  S-transform metric sweep", fontsize=12)

    for ax, vals, ylabel in [
        (axes[0], mjpl_v, "M_JPL  ↑ better"),
        (axes[1], rv3_v,  "RV₃  ↓ better"),
    ]:
        colours = ["#d62728" if p == best["p_factor"] else "#1f77b4"
                   for p in p_vals]
        ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.6)
        ax.set_xlabel("p factor")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_st(signal: np.ndarray, t_us: np.ndarray,
                 mode_signal: np.ndarray, best: dict, title: str) -> None:
    """Side-by-side: default S-transform (p=1) | best-parameter S-transform."""
    fs = _fs_from_time(t_us)

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, p, subtitle, mark_mode=False):
        freqs, times, Sm = _stockwell(signal, fs, p=p,
                                       freq_max_hz=FREQ_MAX_MHZ * 1e6)
        pdb = 20 * np.log10(np.abs(Sm) + 1e-12)
        im  = ax.pcolormesh(times * 1e6, freqs / 1e6, pdb,
                            shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="dB")
        ax.set_ylim(0, FREQ_MAX_MHZ)
        if mark_mode:
            cf, ct = _mode_tf_center(mode_signal, fs, p)
            try:
                ax.plot(times[ct] * 1e6, freqs[cf] / 1e6,
                        "+", color="cyan", markersize=16, markeredgewidth=2,
                        label="mode peak (Q centre)")
                ax.legend(fontsize=8, loc="upper right")
            except IndexError:
                pass
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(subtitle)

    _draw(ax_d, 1.0, "Default  (p=1.0, standard S-transform)")
    _draw(ax_b, best["p_factor"],
          f"Best  p={best['p_factor']}\n"
          f"M_JPL={best['mjpl']:.4e}  RV₃={best['rv3']:.3f}",
          mark_mode=True)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    data    = load_all("304 Steel")
    summary = []

    for d in data:
        dist = d["distance_mm"]
        print(f"\n{'='*64}\n  Distance: {dist} mm\n{'='*64}")

        for sig_key, signal, t_us, mode_signal, mode_name in [
            ("f",  d["f_signal"], d["time_f"],  d["s2_mode"], "S2"),
            ("2f", d["sig_2f"],   d["time_2f"], d["s4_mode"], "S4"),
        ]:
            print(f"  [{sig_key}]  localisation on {mode_name} mode")
            results = sweep_metrics(signal, t_us, mode_signal)
            best    = select_best(results)
            plot_sweep(results, best,
                       title_prefix=f"{sig_key} signal — {dist} mm")
            plot_best_st(signal, t_us, mode_signal, best,
                         title=f"Best S-transform — {sig_key} signal  {dist} mm  "
                               f"({mode_name} mode localisation)")
            summary.append({"dist": dist, "signal": sig_key,
                            "mode": mode_name, **best})

    col = "{:>6}  {:<4}  {:>10}  {:>14}  {:>8}"
    print(f"\n{'='*60}\n  Best S-transform summary  (objective: max M_JPL)\n{'='*60}")
    print("  " + col.format("Dist", "Sig", "p_factor", "M_JPL", "RV₃"))
    print("  " + col.format(*["-"*w for w in (6, 4, 10, 14, 8)]))
    for r in summary:
        print("  " + col.format(f"{r['dist']}mm", r["signal"],
                                f"{r['p_factor']}", f"{r['mjpl']:.4e}",
                                f"{r['rv3']:.3f}"))


if __name__ == "__main__":
    main()
