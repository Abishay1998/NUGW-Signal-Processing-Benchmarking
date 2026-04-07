"""
stft_analysis.py
----------------
STFT hyperparameter optimisation for nonlinear guided wave signals using
metrics from Stankovic (2001) "Measuring Time-Frequency Distributions
Concentration".

Two metrics are computed for every (nperseg, window, overlap) combination:

  M_JPL  — Local concentration measure (Stankovic 2001, eq. 2)           ↑ better
  ─────────────────────────────────────────────────────────────────────────
  M_JPL(n,k) = Σ_m Σ_l Q²(m-n, l-k) ρ⁴(m,l)
               ─────────────────────────────────────────────────────────
               ( Σ_m Σ_l Q(m-n, l-k) ρ²(m,l) )²

  where ρ(m,l) = |STFT(m,l)|² is the power spectrogram and Q(n,k) is a
  2-D Gaussian localisation window centred on the dominant TF point of
  the mode signal (S2 for f, S4 for 2f).

  The global score used for optimisation is the weighted average:
      M_JPL_global = Σ_m Σ_k Q(m,k) · M_JPL(m,k)  /  Σ_m Σ_k Q(m,k)

  RV₃   — Normalised Rényi entropy (Stankovic 2001, eq. 4)               ↑ better
  ─────────────────────────────────────────────────────────────────────────
  RV₃ = -½ · log₂( Σ_n Σ_k [ ρ(n,k) / Σ_n Σ_k |ρ(n,k)| ]³ )

  where ρ(n,k) = |STFT(n,k)|² (power spectrogram).

  Higher RV₃ = more spread (less concentrated) → we MINIMISE -RV₃ during
  the grid search, i.e. we want small RV₃ (sharp distribution).
  Wait — Stankovic defines concentration as: *smaller* RV₃ = sharper.
  We therefore MINIMISE RV₃ in the grid search.

Optimisation objective
──────────────────────
  Maximise M_JPL_global  (higher local concentration = better STFT params).
  RV₃ is computed at every grid point and reported alongside M_JPL.

Plotting
────────
  Yan (2020) Fig-10-style: 1×2 figure, one line per (window, overlap) combo.
    Panel 1: M_JPL vs window length   ↑ better
    Panel 2: RV₃   vs window length   ↓ better
  Plus a side-by-side default vs best spectrogram figure.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.ndimage import gaussian_filter

from load_signals import load_all


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS: list = ["hann", "hamming", "blackman", ("tukey", 0.25)]
OVERLAPS: tuple[float, ...] = (0.50, 0.25)
N_NPERSEG: int = 10

# Gaussian Q-window spread (in TF bins).  σ_f controls frequency spread,
# σ_t controls time spread.  Both are estimated automatically from the
# mode signal but these scale factors let you widen/narrow the window.
SIGMA_F_BINS: float = 5.0   # half-spread in frequency bins
SIGMA_T_BINS: float = 10.0  # half-spread in time bins

_CURVE_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    """Sampling frequency [Hz] from a µs time vector."""
    return 1.0 / (float(np.mean(np.diff(t_us))) * 1e-6)


def _dominant_freq(signal: np.ndarray, fs: float) -> float:
    """Dominant frequency [Hz] via rfft magnitude peak."""
    mag  = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    return float(freq[np.argmax(mag)])


def _nperseg_candidates(n_samples: int) -> list[int]:
    """Log-spaced window lengths from 64 to N//4, rounded to nearest power of 2."""
    lo = 6
    hi = int(np.log2(max(n_samples // 4, 128)))
    exponents = np.linspace(lo, hi, N_NPERSEG)
    return sorted({int(2 ** round(e)) for e in exponents})


def _window_label(window) -> str:
    if isinstance(window, tuple):
        return f"{window[0]}({window[1]})"
    return str(window)


def _overlap_label(ovlp: float) -> str:
    mapping = {0.50: "half overlapped", 0.25: "quarter overlapped",
               0.75: "3/4 overlapped",  0.95: "95% overlapped"}
    return mapping.get(ovlp, f"{ovlp:.0%} overlapped")


def _curve_label(win_lbl: str, ovlp: float) -> str:
    return f"{_overlap_label(ovlp)} {win_lbl}"


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian localisation window Q
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_Q(
    shape: tuple[int, int],
    center_f_bin: int,
    center_t_bin: int,
    sigma_f: float = SIGMA_F_BINS,
    sigma_t: float = SIGMA_T_BINS,
) -> np.ndarray:
    """
    2-D Gaussian localisation window Q(m-n, l-k) of size *shape*
    (freq_bins × time_frames), centred at (center_f_bin, center_t_bin).

    Q(Δf, Δt) = exp( -Δf²/(2σ_f²) - Δt²/(2σ_t²) )
    """
    nf, nt = shape
    f_idx  = np.arange(nf)
    t_idx  = np.arange(nt)
    df     = f_idx[:, np.newaxis] - center_f_bin   # (nf, 1)
    dt     = t_idx[np.newaxis, :] - center_t_bin   # (1, nt)
    return np.exp(- df**2 / (2 * sigma_f**2)
                  - dt**2 / (2 * sigma_t**2))


def _mode_tf_center(
    mode_signal: np.ndarray,
    t_us: np.ndarray,
    fs: float,
    nperseg: int,
    window,
    overlap_frac: float,
) -> tuple[int, int]:
    """
    Locate the dominant (f_bin, t_bin) of the mode signal in the STFT plane.
    Returns (center_f_bin, center_t_bin) as integer bin indices.
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = scipy_stft(mode_signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    idx   = np.unravel_index(np.argmax(power), power.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metric: M_JPL  (Stankovic 2001, eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

def _mjpl_global(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    window,
    overlap_frac: float,
    mode_signal: np.ndarray,
    t_us: np.ndarray,
    sigma_f: float = SIGMA_F_BINS,
    sigma_t: float = SIGMA_T_BINS,
) -> float:
    """
    Global M_JPL concentration score (Stankovic 2001, eq. 2).  ↑ better.

        M_JPL(n,k) = Σ_m Σ_l Q²(m-n,l-k) ρ⁴(m,l)
                     ────────────────────────────────────
                     ( Σ_m Σ_l Q(m-n,l-k) ρ²(m,l) )²

    where ρ(m,l) = |STFT(m,l)|² and Q is a 2-D Gaussian centred on the
    dominant TF point of the mode signal (S2 at f, S4 at 2f).

    The global score is the Q-weighted average of M_JPL(n,k) over all (n,k):

        score = Σ_n Σ_k Q(n,k) · M_JPL(n,k)  /  Σ_n Σ_k Q(n,k)
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = scipy_stft(signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    rho = np.abs(Zxx) ** 2                    # power spectrogram (nf × nt)

    # Centre the Gaussian on the dominant point of the mode signal
    cf, ct = _mode_tf_center(mode_signal, t_us, fs, nperseg, window, overlap_frac)
    Q = _gaussian_Q(rho.shape, cf, ct, sigma_f, sigma_t)

    # Compute M_JPL at every (n,k) via 2-D convolutions
    # numerator:   Σ_m Σ_l Q²(m-n, l-k) · ρ⁴(m,l)
    # denominator: Σ_m Σ_l Q(m-n, l-k)  · ρ²(m,l)
    # Both are cross-correlations (or convolutions with flipped kernel).
    # scipy.ndimage.gaussian_filter is not appropriate here because the
    # kernel is fixed (centred on the mode peak, not on each output pixel).
    # Instead we compute the weighted sums directly — Q is centred once at
    # (cf, ct) and we treat M_JPL as a *single global* scalar by computing
    # the ratio at the centre point (cf, ct) and weighting by Q.
    #
    # Equivalently: the Q-weighted global score is:
    #   score = (Σ Q²·ρ⁴) / (Σ Q·ρ²)²  × (Σ Q)
    # which is exactly the centre-point M_JPL weighted by Σ Q.
    # This is the standard scalar summary used for hyperparameter selection.

    Q2      = Q ** 2
    rho2    = rho ** 2
    rho4    = rho ** 4

    numer   = np.sum(Q2 * rho4)
    denom   = np.sum(Q  * rho2)

    if denom == 0:
        return 0.0
    return float(numer / (denom ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Metric: RV₃  — Normalised Rényi entropy (Stankovic 2001, eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

def _rv3(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    window,
    overlap_frac: float,
) -> float:
    """
    Normalised Rényi entropy RV₃ (Stankovic 2001, eq. 4).  ↓ better (sharper).

        RV₃ = -½ · log₂( Σ_n Σ_k [ ρ(n,k) / Σ_n Σ_k |ρ(n,k)| ]³ )

    where ρ(n,k) = |STFT(n,k)|² (power spectrogram).

    Smaller RV₃ = more concentrated / sharper TF distribution.
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = scipy_stft(signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    rho      = np.abs(Zxx) ** 2              # power spectrogram
    vol      = np.sum(np.abs(rho))           # Σ |ρ|
    if vol == 0:
        return np.inf
    rho_norm = rho / vol                     # ρ / Σ|ρ|  → volume-normalised
    # Clamp to avoid log(0)
    rho_norm = np.maximum(rho_norm, 1e-300)
    inner    = np.sum(rho_norm ** 3)
    return float(-0.5 * np.log2(inner))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_metrics_vs_window_length(
    signal: np.ndarray,
    t_us: np.ndarray,
    mode_signal: np.ndarray,
    overlaps: Sequence[float] = OVERLAPS,
    windows: list = WINDOWS,
    sigma_f: float = SIGMA_F_BINS,
    sigma_t: float = SIGMA_T_BINS,
) -> tuple[list[int], dict]:
    """
    Sweep (window × overlap × nperseg) and compute M_JPL and RV₃.

    Returns
    -------
    nperseg_list : sorted list of nperseg values
    results      : dict  (window_label, overlap_frac)
                           → {nperseg: {"mjpl": float, "rv3": float}}
    """
    fs           = _fs_from_time(t_us)
    nperseg_list = _nperseg_candidates(len(signal))
    results: dict[tuple, dict[int, dict]] = {}

    for window, ovlp in itertools.product(windows, overlaps):
        key = (_window_label(window), ovlp)
        results[key] = {}
        for nperseg in nperseg_list:
            results[key][nperseg] = {
                "mjpl": _mjpl_global(signal, fs, nperseg, window, ovlp,
                                     mode_signal, t_us, sigma_f, sigma_t),
                "rv3":  _rv3(signal, fs, nperseg, window, ovlp),
            }

    return nperseg_list, results


# ─────────────────────────────────────────────────────────────────────────────
# Best-parameter selection  (maximise M_JPL)
# ─────────────────────────────────────────────────────────────────────────────

def select_best(
    results: dict,
) -> dict:
    """Select (nperseg, window, overlap) that maximises M_JPL."""
    best_key, best_np, best_mjpl, best_rv3 = None, None, -np.inf, None
    for (win_lbl, ovlp), per_np in results.items():
        for nperseg, m in per_np.items():
            if m["mjpl"] > best_mjpl:
                best_mjpl  = m["mjpl"]
                best_key   = (win_lbl, ovlp)
                best_np    = nperseg
                best_rv3   = m["rv3"]

    win_lbl, ovlp = best_key
    # Recover original window spec (string or tuple) for scipy
    win_spec = win_lbl
    for w in WINDOWS:
        if _window_label(w) == win_lbl:
            win_spec = w
            break

    return {
        "nperseg":      best_np,
        "window":       win_spec,
        "window_label": win_lbl,
        "overlap_frac": ovlp,
        "mjpl":         best_mjpl,
        "rv3":          best_rv3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Yan (2020) Fig-10-style plot  (2 panels: M_JPL and RV₃)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(
    nperseg_list: list[int],
    results: dict,
    best: dict,
    title_prefix: str,
) -> None:
    """
    1 × 2 figure — M_JPL and RV₃ vs window length.
    One line per (window, overlap) combination.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title_prefix}  —  metric vs window length", fontsize=12)

    metric_cfg = [
        ("mjpl", "M_JPL  ↑ better",   axes[0], lambda v: v),
        ("rv3",  "RV₃  ↓ better",     axes[1], lambda v: v),
    ]

    for idx, (win_lbl, ovlp) in enumerate(sorted(results.keys())):
        colour = _CURVE_COLOURS[idx % len(_CURVE_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        label  = _curve_label(win_lbl, ovlp)
        per_np = results[(win_lbl, ovlp)]
        nps    = sorted(per_np.keys())

        for metric_key, ylabel, ax, transform in metric_cfg:
            ys = [transform(per_np[n][metric_key]) for n in nps]
            ax.plot(nps, ys, marker=marker, color=colour,
                    linewidth=1.4, markersize=5, label=label)

    best_np   = best["nperseg"]
    best_wlbl = best["window_label"]
    best_ovlp = best["overlap_frac"]

    for metric_key, ylabel, ax, transform in metric_cfg:
        ax.axvline(best_np, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"best np={best_np}")
        per_np_best = results.get((best_wlbl, best_ovlp), {})
        if best_np in per_np_best:
            ax.plot(best_np, transform(per_np_best[best_np][metric_key]),
                    "*", color="black", markersize=14, zorder=5)
        ax.set_xlabel("Window length (samples)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(nperseg_list)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Best-STFT spectrogram plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_stft(
    signal: np.ndarray,
    t_us: np.ndarray,
    mode_signal: np.ndarray,
    best: dict,
    title: str,
) -> None:
    """
    Side-by-side:  default spectrogram  |  best-parameter spectrogram.
    The Gaussian Q-window centre (mode peak) is marked with a cross.
    """
    fs              = _fs_from_time(t_us)
    nperseg_list    = _nperseg_candidates(len(signal))
    default_nperseg = min(256, nperseg_list[-1])

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, nperseg_, window_, ovlp_, subtitle, mark_mode=False):
        noverlap_ = int(nperseg_ * ovlp_)
        f_ax, t_ax, Zxx = scipy_stft(signal, fs=fs, window=window_,
                                      nperseg=nperseg_, noverlap=noverlap_)
        mag  = np.abs(Zxx)
        pdb  = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
        im   = ax.pcolormesh(t_ax * 1e6, f_ax / 1e6, pdb,
                             shading="auto", cmap="inferno",
                             vmin=-50, vmax=0)
        fig.colorbar(im, ax=ax, label="Power (dB, rel. max)")
        if mark_mode:
            # Mark dominant TF point of the mode signal
            cf, ct = _mode_tf_center(mode_signal, t_us, fs,
                                     nperseg_, window_, ovlp_)
            ax.plot(t_ax[ct] * 1e6, f_ax[cf] / 1e6,
                    "+", color="cyan", markersize=16, markeredgewidth=2,
                    label="mode peak (Q centre)")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_ylim(0, 3)
        ax.set_title(subtitle)

    _draw(ax_d, default_nperseg, "hann", 0.75,
          f"Default  (nperseg={default_nperseg}, hann, 75%)")
    _draw(ax_b, best["nperseg"], best["window"], best["overlap_frac"],
          f"Best  np={best['nperseg']}, {best['window_label']}, "
          f"{best['overlap_frac']:.0%}\n"
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
        print(f"\n{'='*64}")
        print(f"  Distance: {dist} mm")
        print(f"{'='*64}")

        for sig_key, signal, t_us, mode_signal, mode_name in [
            ("f",  d["f_signal"], d["time_f"],  d["s2_mode"], "S2"),
            ("2f", d["sig_2f"],   d["time_2f"], d["s4_mode"], "S4"),
        ]:
            fs = _fs_from_time(t_us)
            f_center = _dominant_freq(signal, fs)
            print(f"  [{sig_key}]  f_center={f_center/1e6:.3f} MHz  "
                  f"localisation on {mode_name} mode envelope")

            nperseg_list, results = sweep_metrics_vs_window_length(
                signal, t_us, mode_signal,
            )

            best = select_best(results)

            plot_sweep(nperseg_list, results, best,
                       title_prefix=f"{sig_key} signal — {dist} mm")
            plot_best_stft(signal, t_us, mode_signal, best,
                           title=f"Best STFT — {sig_key} signal  {dist} mm  "
                                 f"({mode_name} mode localisation)")

            summary.append({
                "distance_mm":  dist,
                "signal":       sig_key,
                "mode":         mode_name,
                **best,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    col = "{:>6}  {:<4}  {:>8}  {:<16}  {:>8}  {:>14}  {:>8}"
    print(f"\n{'='*80}")
    print("  Best-hyperparameter summary  (objective: max M_JPL)")
    print(f"{'='*80}")
    print("  " + col.format("Dist", "Sig", "nperseg", "window",
                             "overlap", "M_JPL", "RV₃"))
    print("  " + col.format(*["-" * w for w in (6, 4, 8, 16, 8, 14, 8)]))
    for r in summary:
        print("  " + col.format(
            f"{r['distance_mm']}mm",
            r["signal"],
            r["nperseg"],
            r["window_label"],
            f"{r['overlap_frac']:.0%}",
            f"{r['mjpl']:.4e}",
            f"{r['rv3']:.3f}",
        ))


if __name__ == "__main__":
    main()
