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

Optimisation objective — Time-Frequency Resolution Score (TFRS)
---------------------------------------------------------------
TFRS measures how well energy is concentrated around the dominant
frequency peak in the TF plane:

    TFRS = E_band / E_total

where E_band is the STFT energy within ±10 % of the dominant frequency,
and E_total is the total STFT energy. A higher score means the representation
is sharper and less smeared across irrelevant frequencies.
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


def _tfrs(signal: np.ndarray, fs: float,
          nperseg: int, window, overlap_frac: float) -> float:
    """
    Time-Frequency Resolution Score (TFRS).
    Higher is better — energy concentrated near the dominant frequency.
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = stft(signal, fs=fs, window=window,
                     nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2                      # (freq_bins, time_frames)
    e_total = power.sum()
    if e_total == 0:
        return 0.0
    # Dominant frequency bin (by total power across time)
    dominant_bin = int(np.argmax(power.sum(axis=1)))
    band = max(1, int(0.10 * dominant_bin))        # ±10 % of dominant bin index
    lo = max(0, dominant_bin - band)
    hi = min(power.shape[0], dominant_bin + band + 1)
    e_band = power[lo:hi, :].sum()
    return float(e_band / e_total)


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

    # ── Grid search ──────────────────────────────────────────────────────────
    results = []   # (score, nperseg, window, overlap_frac)
    for nperseg, window, ovlp in itertools.product(
            nperseg_list, WINDOWS, OVERLAP_FRACS):
        score = _tfrs(signal, fs, nperseg, window, ovlp)
        results.append((score, nperseg, window, ovlp))

    results.sort(key=lambda x: -x[0])
    best_score, best_nperseg, best_window, best_ovlp = results[0]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"STFT optimisation  —  {label}", fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_score  = fig.add_subplot(gs[0, 0])   # score vs nperseg (best window/ovlp)
    ax_win    = fig.add_subplot(gs[0, 1])   # score vs window type
    ax_ovlp   = fig.add_subplot(gs[0, 2])   # score vs overlap_frac
    ax_stft_d = fig.add_subplot(gs[1, :2])  # default STFT
    ax_stft_b = fig.add_subplot(gs[1, 2])   # best STFT

    # ── Panel 1: TFRS vs nperseg (best window & overlap fixed) ───────────────
    nperseg_scores: dict[int, list] = {}
    for score, np_, win, ovlp in results:
        nperseg_scores.setdefault(np_, []).append(score)
    np_vals = sorted(nperseg_scores)
    np_means = [np.mean(nperseg_scores[v]) for v in np_vals]
    ax_score.plot(np_vals, np_means, "o-", color=colour)
    ax_score.axvline(best_nperseg, color="gray", linestyle="--", alpha=0.6)
    ax_score.set_xlabel("nperseg (samples)")
    ax_score.set_ylabel("Mean TFRS")
    ax_score.set_title("Score vs window length")
    ax_score.set_xscale("log", base=2)
    ax_score.grid(True, alpha=0.3)

    # ── Panel 2: TFRS vs window type ─────────────────────────────────────────
    win_scores: dict[str, list] = {}
    for score, np_, win, ovlp in results:
        wl = _window_label(win)
        win_scores.setdefault(wl, []).append(score)
    win_labels = list(win_scores)
    win_means  = [np.mean(win_scores[w]) for w in win_labels]
    bars = ax_win.bar(win_labels, win_means, color=colour, alpha=0.8)
    best_wl = _window_label(best_window)
    for bar, wl in zip(bars, win_labels):
        if wl == best_wl:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
    ax_win.set_ylabel("Mean TFRS")
    ax_win.set_title("Score vs window function")
    ax_win.set_ylim(0, max(win_means) * 1.15)
    ax_win.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: TFRS vs overlap_frac ────────────────────────────────────────
    ovlp_scores: dict[float, list] = {}
    for score, np_, win, ovlp in results:
        ovlp_scores.setdefault(ovlp, []).append(score)
    ovlp_vals  = sorted(ovlp_scores)
    ovlp_means = [np.mean(ovlp_scores[v]) for v in ovlp_vals]
    ax_ovlp.plot([f"{v:.0%}" for v in ovlp_vals], ovlp_means, "s-", color=colour)
    best_ovlp_label = f"{best_ovlp:.0%}"
    ax_ovlp.axvline(best_ovlp_label, color="gray", linestyle="--", alpha=0.6)
    ax_ovlp.set_xlabel("Overlap fraction")
    ax_ovlp.set_ylabel("Mean TFRS")
    ax_ovlp.set_title("Score vs overlap fraction")
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
               f"TFRS = {best_score:.4f}")

    plt.show()

    return {
        "nperseg": best_nperseg,
        "window":  best_window,
        "overlap_frac": best_ovlp,
        "tfrs": best_score,
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
          f"{'overlap':>8}  {'TFRS':>8}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*16}  {'-'*8}  {'-'*8}")
    for row in summary:
        for sig_key in ("f", "2f"):
            b = row[sig_key]
            print(f"  {row['distance_mm']:>5}mm  {sig_key:>4}  "
                  f"{b['nperseg']:>8}  "
                  f"{_window_label(b['window']):<16}  "
                  f"{b['overlap_frac']:>7.0%}  "
                  f"{b['tfrs']:>8.4f}")


if __name__ == "__main__":
    main()
