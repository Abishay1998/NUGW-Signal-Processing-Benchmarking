"""
stft_analysis.py
----------------
For each signal (f and 2f) at each propagation distance:
  1. Plot the STFT spectrogram with default parameters
  2. Optimise STFT hyperparameters (nperseg, window, overlap_frac) to
     maximise time-frequency resolution
  3. Plot the optimisation landscape and the best spectrogram

STFT Hyperparameters being optimised
-------------------------------------
  nperseg      : Window length in samples.
                 Longer → better frequency resolution, worse time resolution.
                 Range: 64 … N//4 (log-spaced)

  window       : Window function applied before FFT.
                 Controls spectral leakage / sidelobe level.
                 Options: 'hann', 'hamming', 'blackman', ('tukey', 0.25)

  overlap_frac : Fraction of nperseg used as overlap between frames.
                 Higher → finer time steps (more redundancy).
                 Range: 0.50 … 0.95

Optimisation objectives — Jin Yan (2020) metrics J₁ and J₂
-----------------------------------------------------------
Two metrics from: Jin Yan, "A Comparison of Time-Frequency Methods for
Real-Time Application to High-Rate Dynamic Systems", Vibration 2020, 3.

  J₁  =  (1/n) Σ |ω̂ᵢ − ωᵢ|          (mean absolute IF tracking error)

        Lower is better. Measures how accurately the STFT tracks the
        known excitation frequency at each time step. Ground truth ωᵢ
        is the dominant FFT frequency of the full signal (known for
        simulated data).

  J₂  =  ∬ log( |TFR(t,ω)|³ / Z ) dt dω      (normalised Rényi entropy)

        Z = ∬ |TFR(t,ω)|³ dt dω  normalises the distribution so J₂ is
        independent of TF plane size and signal amplitude.
        Higher (less negative) is better: rewards a sharp, concentrated
        TF distribution and penalises smearing across the TF plane.

The grid search minimises J₁ (IF tracking error) — directly optimises how
accurately the STFT tracks the known excitation frequency, which is available
for simulated data.  J₂ is then computed at the best configuration as a
reporting metric for cross-method comparison (as used by Jin Yan).
"""

from __future__ import annotations
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft

from load_signals import load_all


# ── Hyperparameter search grid ────────────────────────────────────────────────
WINDOWS       = ["hann", "hamming", "blackman", ("tukey", 0.25)]
OVERLAP_FRACS = [0.50, 0.65, 0.75, 0.85, 0.95]
N_NPERSEG     = 8          # number of window-length candidates (log-spaced)

COLOURS = {"f": "#2563EB", "2f": "#DC2626"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    """Sampling frequency in Hz from a time vector in µs."""
    return 1.0 / (np.mean(np.diff(t_us)) * 1e-6)


def _nperseg_candidates(n_samples: int) -> list[int]:
    """Log-spaced window lengths from 64 to N//4, rounded to nearest power of 2."""
    lo, hi = 6, int(np.log2(max(n_samples // 4, 128)))
    exponents = np.linspace(lo, hi, N_NPERSEG)
    return sorted({int(2 ** round(e)) for e in exponents})


def _j2(signal: np.ndarray, fs: float,
          nperseg: int, window, overlap_frac: float) -> float:
    """
    J₂ — Rényi entropy (Jin Yan 2020, eq. 8), normalised form.
    J₂ = ∬ log( |TFR(t,ω)|³ / Z ) dt dω,   Z = ∬ |TFR(t,ω)|³ dt dω

    Normalising by Z turns |TFR|³ into a probability-like distribution,
    so J₂ measures *concentration* independently of TF plane size or
    signal amplitude.  Higher (less negative) = sharper distribution.
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = stft(signal, fs=fs, window=window,
                     nperseg=nperseg, noverlap=noverlap)
    p = np.abs(Zxx) ** 3                          # |TFR|³  (freq × time)
    z = p.sum()
    if z == 0:
        return -np.inf
    p_norm = p / z                                 # normalise → sums to 1
    # Avoid log(0)
    p_norm = np.maximum(p_norm, 1e-300)
    return float(np.sum(np.log(p_norm)))


def _j1(signal: np.ndarray, fs: float,
        nperseg: int, window, overlap_frac: float,
        true_freq_hz: float) -> float:
    """
    J₁ — mean absolute instantaneous frequency tracking error (Jin Yan 2020, eq. 8).
    J₁ = (1/n) Σ |ω̂ᵢ − ωᵢ|   (in Hz)
    Lower is better.  true_freq_hz is the known excitation frequency.
    """
    noverlap = int(nperseg * overlap_frac)
    freqs, _, Zxx = stft(signal, fs=fs, window=window,
                         nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2               # (freq_bins, time_frames)
    # Estimated IF at each time frame = frequency bin with maximum power
    estimated_if = freqs[np.argmax(power, axis=0)]  # (time_frames,)
    return float(np.mean(np.abs(estimated_if - true_freq_hz)))


def _window_label(window) -> str:
    if isinstance(window, tuple):
        return f"{window[0]}({window[1]})"
    return window


# ── Core: optimise + plot for one signal ─────────────────────────────────────

def optimise_and_plot(
    signal: np.ndarray,
    t_us: np.ndarray,
    label: str,          # e.g. "f  — 200 mm"
    colour: str,
) -> dict:
    """
    Grid-search over (nperseg, window, overlap_frac), plot optimisation
    landscape and best spectrogram. Returns the best parameter dict.
    """
    fs = _fs_from_time(t_us)
    nperseg_list = _nperseg_candidates(len(signal))

    # Ground-truth excitation frequency = dominant FFT bin of the full signal
    fft_mag  = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    true_freq_hz = float(fft_freq[np.argmax(fft_mag)])

    # ── Grid search — minimise J₁ (mean IF tracking error) ──────────────────
    results = []   # (j1_score, nperseg, window, overlap_frac)
    for nperseg, window, ovlp in itertools.product(
            nperseg_list, WINDOWS, OVERLAP_FRACS):
        score = _j1(signal, fs, nperseg, window, ovlp, true_freq_hz)
        results.append((score, nperseg, window, ovlp))

    results.sort(key=lambda x: x[0])   # ascending — lower J₁ is better
    best_j1, best_nperseg, best_window, best_ovlp = results[0]

    # J₂ at best configuration (reporting metric for cross-method comparison)
    best_j2 = _j2(signal, fs, best_nperseg, best_window, best_ovlp)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"STFT optimisation  —  {label}", fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_score  = fig.add_subplot(gs[0, 0])   # score vs nperseg (best window/ovlp)
    ax_win    = fig.add_subplot(gs[0, 1])   # score vs window type
    ax_ovlp   = fig.add_subplot(gs[0, 2])   # score vs overlap_frac
    ax_stft_d = fig.add_subplot(gs[1, :2])  # default STFT
    ax_stft_b = fig.add_subplot(gs[1, 2])   # best STFT

    # ── Panel 1: J₁ vs nperseg ───────────────────────────────────────────────
    nperseg_scores: dict[int, list] = {}
    for score, np_, win, ovlp in results:
        nperseg_scores.setdefault(np_, []).append(score)
    np_vals = sorted(nperseg_scores)
    np_means = [np.mean(nperseg_scores[v]) for v in np_vals]
    ax_score.plot(np_vals, [v / 1e3 for v in np_means], "o-", color=colour)
    ax_score.axvline(best_nperseg, color="gray", linestyle="--", alpha=0.6)
    ax_score.set_xlabel("nperseg (samples)")
    ax_score.set_ylabel("Mean J₁ (kHz)  ↓ better")
    ax_score.set_title("J₁ vs window length")
    ax_score.set_xscale("log", base=2)
    ax_score.grid(True, alpha=0.3)

    # ── Panel 2: J₁ vs window type ───────────────────────────────────────────
    win_scores: dict[str, list] = {}
    for score, np_, win, ovlp in results:
        wl = _window_label(win)
        win_scores.setdefault(wl, []).append(score)
    win_labels = list(win_scores)
    win_means  = [np.mean(win_scores[w]) / 1e3 for w in win_labels]
    bars = ax_win.bar(win_labels, win_means, color=colour, alpha=0.8)
    best_wl = _window_label(best_window)
    for bar, wl in zip(bars, win_labels):
        if wl == best_wl:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
    ax_win.set_ylabel("Mean J₁ (kHz)  ↓ better")
    ax_win.set_title("J₁ vs window function")
    ax_win.set_ylim(0, max(win_means) * 1.15)
    ax_win.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: J₁ vs overlap_frac ──────────────────────────────────────────
    ovlp_scores: dict[float, list] = {}
    for score, np_, win, ovlp in results:
        ovlp_scores.setdefault(ovlp, []).append(score)
    ovlp_vals  = sorted(ovlp_scores)
    ovlp_means = [np.mean(ovlp_scores[v]) / 1e3 for v in ovlp_vals]
    ax_ovlp.plot([f"{v:.0%}" for v in ovlp_vals], ovlp_means, "s-", color=colour)
    best_ovlp_label = f"{best_ovlp:.0%}"
    ax_ovlp.axvline(best_ovlp_label, color="gray", linestyle="--", alpha=0.6)
    ax_ovlp.set_xlabel("Overlap fraction")
    ax_ovlp.set_ylabel("Mean J₁ (kHz)  ↓ better")
    ax_ovlp.set_title("J₁ vs overlap fraction")
    ax_ovlp.grid(True, alpha=0.3)

    # ── Panel 4: Default STFT (hann, nperseg=256, overlap=0.75) ──────────────
    def _plot_stft(ax, nperseg_, window_, ovlp_, title_):
        noverlap_ = int(nperseg_ * ovlp_)
        f_ax, t_ax, Zxx = stft(signal, fs=fs, window=window_,
                                nperseg=nperseg_, noverlap=noverlap_)
        power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        im = ax.pcolormesh(t_ax * 1e6, f_ax / 1e6, power_db,
                           shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="dB")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(title_)

    default_nperseg = min(256, nperseg_list[-1])
    _plot_stft(ax_stft_d, default_nperseg, "hann", 0.75,
               f"Default STFT  (nperseg={default_nperseg}, hann, 75% overlap)")
    _plot_stft(ax_stft_b, best_nperseg, best_window, best_ovlp,
               f"Best STFT  (nperseg={best_nperseg}, "
               f"{_window_label(best_window)}, {best_ovlp:.0%} overlap)\n"
               f"J\u2081={best_j1/1e3:.2f} kHz  |  J\u2082={best_j2:.1f}")

    plt.show()

    return {
        "nperseg":      best_nperseg,
        "window":       best_window,
        "overlap_frac": best_ovlp,
        "j2":           best_j2,
        "j1_hz":        best_j1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data = load_all("304 Steel")
    summary = []

    for d in data:
        dist = d["distance_mm"]
        print(f"\n{'='*60}")
        print(f"  Distance: {dist} mm")
        print(f"{'='*60}")

        print(f"  Optimising STFT for f signal …")
        best_f = optimise_and_plot(
            d["f_signal"], d["time_f"],
            label=f"f signal — {dist} mm",
            colour=COLOURS["f"],
        )

        print(f"  Optimising STFT for 2f signal …")
        best_2f = optimise_and_plot(
            d["sig_2f"], d["time_2f"],
            label=f"2f signal — {dist} mm",
            colour=COLOURS["2f"],
        )

        summary.append({
            "distance_mm": dist,
            "f":  best_f,
            "2f": best_2f,
        })

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Best hyperparameters summary")
    print(f"{'='*72}")
    print(f"  {'Dist':>6}  {'Sig':>4}  {'nperseg':>8}  {'window':<16}  "
          f"{'overlap':>8}  {'J2':>12}  {'J1 (kHz)':>10}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*16}  {'-'*8}  {'-'*12}  {'-'*10}")
    for row in summary:
        for sig_key in ("f", "2f"):
            b = row[sig_key]
            print(f"  {row['distance_mm']:>5}mm  {sig_key:>4}  "
                  f"{b['nperseg']:>8}  "
                  f"{_window_label(b['window']):<16}  "
                  f"{b['overlap_frac']:>7.0%}  "
                  f"{b['j2']:>12.1f}  "
                  f"{b['j1_hz']/1e3:>10.2f}")


if __name__ == "__main__":
    main()
