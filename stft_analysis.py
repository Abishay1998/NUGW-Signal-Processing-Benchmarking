"""
stft_analysis.py
----------------
STFT hyperparameter optimisation for nonlinear guided wave signals.

For each signal (f / 2f) at each propagation distance the script:
  1. Sweeps (nperseg × window × overlap) and records J₁, J₂, J₃ per combo.
  2. Selects the best combo via a normalised composite score J_tot.
  3. Plots Yan (2020) Fig-10-style panels (one line per window/overlap combo).
  4. Shows the best-parameter STFT spectrogram.
  5. Computes J₄ (coefficient of variation across distances) to flag the most
     robust parameter sets.

Metrics — Jin Yan, Vibration 2020, 3, eq. 8
--------------------------------------------
  J₁  mean absolute IF tracking error                     ↓ better
        J₁ = (1/n) Σ |ω̂ᵢ − ωᵢ|   [Hz]

  J₂  Rényi entropy (Yan 2020 exact, no normalisation)    ↑ better
        J₂ = ΣΣ log( |STFT(t,ω)|³ )

  J₃  physics-based leakage ratio                         ↓ better
        J₃ = E_out / E_in
        where the TF mask is a rectangle centred on the expected wave
        packet: [f_c±Δf] × [t_arr±Δt]

  J₄  coefficient of variation of J₁ (or J₃) across distances  ↓ better
        J₄ = std(values) / mean(values)

Composite selection
-------------------
  J_tot = w₁·J̃₁ + w₂·(1−J̃₂) + w₃·J̃₃
  where J̃ᵢ = (Jᵢ − min) / (max − min)  (min–max normalisation)
"""

from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import stft as scipy_stft

from load_signals import load_all


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS: list = ["hann", "hamming", "blackman", ("tukey", 0.25)]
OVERLAPS: tuple[float, ...] = (0.50, 0.25)
N_NPERSEG: int = 10
COLOURS: dict[str, str] = {"f": "#2563EB", "2f": "#DC2626"}

DELTA_F_FRAC: float = 0.10   # Δf = DELTA_F_FRAC × f_center
ENVELOPE_THRESHOLD: float = 0.05  # fraction of peak envelope to define time mask


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    return 1.0 / (float(np.mean(np.diff(t_us))) * 1e-6)


def _dominant_freq(signal: np.ndarray, fs: float) -> float:
    mag  = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    return float(freq[np.argmax(mag)])


def _nperseg_candidates(n_samples: int) -> list[int]:
    lo = 6
    hi = int(np.log2(max(n_samples // 4, 128)))
    exponents = np.linspace(lo, hi, N_NPERSEG)
    return sorted({int(2 ** round(e)) for e in exponents})


def _window_label(window) -> str:
    if isinstance(window, tuple):
        return f"{window[0]}({window[1]})"
    return str(window)


def _overlap_label(overlap_frac: float) -> str:
    mapping = {0.50: "half overlapped", 0.25: "quarter overlapped",
               0.75: "3/4 overlapped",  0.95: "95% overlapped"}
    return mapping.get(overlap_frac, f"{overlap_frac:.0%} overlapped")


def _curve_label(win_lbl: str, overlap_frac: float) -> str:
    return f"{_overlap_label(overlap_frac)} {win_lbl}"


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _j1(signal: np.ndarray, fs: float,
        nperseg: int, window, overlap_frac: float,
        true_freq_hz: float) -> float:
    """J₁ — mean absolute IF tracking error [Hz] (Yan 2020 eq. 8). ↓ better."""
    noverlap = int(nperseg * overlap_frac)
    freqs, _, Zxx = scipy_stft(signal, fs=fs, window=window,
                                nperseg=nperseg, noverlap=noverlap)
    power  = np.abs(Zxx) ** 2
    est_if = freqs[np.argmax(power, axis=0)]
    return float(np.mean(np.abs(est_if - true_freq_hz)))


def _j2_yan(signal: np.ndarray, fs: float,
            nperseg: int, window, overlap_frac: float) -> float:
    """
    J₂ — Rényi entropy, Yan (2020) exact definition (eq. 8). ↑ better.
    J₂ = ΣΣ log( |STFT(t,ω)|³ )   — no normalisation by Z.
    """
    noverlap = int(nperseg * overlap_frac)
    _, _, Zxx = scipy_stft(signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    mag   = np.abs(Zxx)
    floor = 1e-12 * mag.max() if mag.max() > 0 else 1e-12
    mag   = np.maximum(mag, floor)
    return float(np.sum(np.log(mag ** 3)))


def _j3(signal: np.ndarray, fs: float,
        nperseg: int, window, overlap_frac: float,
        f_center_hz: float,
        delta_f_hz: float,
        mode_signal: np.ndarray,
        t_us: np.ndarray,
        envelope_threshold: float = ENVELOPE_THRESHOLD) -> float:
    """
    J₃ — physics-based TF leakage ratio. ↓ better.
    J₃ = E_out / E_in

    The TF mask is built directly from the actual mode signal column
    extracted from the Excel file (S2 for f, S4 for 2f):

      Frequency mask : [f_center − Δf,  f_center + Δf]
      Time mask      : contiguous region where |envelope(mode_signal)| ≥
                       envelope_threshold × peak envelope

    This avoids any assumption about group velocity.
    """
    # ── Time mask from mode signal envelope ──────────────────────────────
    envelope = np.abs(mode_signal)
    threshold = envelope_threshold * envelope.max() if envelope.max() > 0 else 0.0
    above = envelope >= threshold
    if not above.any():
        # Mode is silent — mask covers full time range (conservative)
        t_lo_us, t_hi_us = t_us[0], t_us[-1]
    else:
        idx = np.where(above)[0]
        t_lo_us = float(t_us[idx[0]])
        t_hi_us = float(t_us[idx[-1]])

    # ── Compute STFT ─────────────────────────────────────────────────────────────────
    noverlap = int(nperseg * overlap_frac)
    freqs, t_ax, Zxx = scipy_stft(signal, fs=fs, window=window,
                                   nperseg=nperseg, noverlap=noverlap)
    t_stft_us = t_ax * 1e6

    f_mask  = (freqs >= f_center_hz - delta_f_hz) & (freqs <= f_center_hz + delta_f_hz)
    t_mask  = (t_stft_us >= t_lo_us) & (t_stft_us <= t_hi_us)
    in_mask = np.outer(f_mask, t_mask)

    power = np.abs(Zxx) ** 2
    e_in  = power[in_mask].sum()
    e_out = power[~in_mask].sum()

    if e_in == 0:
        return np.inf
    return float(e_out / e_in)


def _j4(values: list[float]) -> float:
    """
    J₄ — coefficient of variation (robustness across distances). ↓ better.
    J₄ = std(values) / mean(values)
    """
    arr = np.asarray(values, dtype=float)
    mu  = arr.mean()
    if mu == 0:
        return np.inf
    return float(arr.std() / mu)


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _composite_score(j1_arr: np.ndarray, j2_arr: np.ndarray, j3_arr: np.ndarray,
                     w1: float = 0.3, w2: float = 0.3, w3: float = 0.4
                     ) -> np.ndarray:
    """J_tot = w₁·J̃₁ + w₂·(1−J̃₂) + w₃·J̃₃   (lower = better)."""
    finite_j3 = np.where(np.isinf(j3_arr),
                         np.nanmax(np.where(np.isinf(j3_arr), np.nan, j3_arr)),
                         j3_arr)
    return (w1 * _normalise(j1_arr)
            + w2 * (1.0 - _normalise(j2_arr))
            + w3 * _normalise(finite_j3))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep function
# ─────────────────────────────────────────────────────────────────────────────

def sweep_metrics_vs_window_length(
    signal: np.ndarray,
    t_us: np.ndarray,
    mode_signal: np.ndarray,
    f_center_hz: float,
    delta_f_hz: float,
    overlaps: Sequence[float] = OVERLAPS,
    windows: list = WINDOWS,
) -> tuple[list[int], dict]:
    """
    Sweep (window × overlap × nperseg) and compute J₁, J₂, J₃ at every point.

    Parameters
    ----------
    signal      : total sum signal (in-plane + out-of-plane)
    t_us        : time vector [µs]
    mode_signal : individual mode signal column (S2 for f, S4 for 2f);
                  its envelope defines the time mask for J₃
    f_center_hz : dominant excitation frequency [Hz]
    delta_f_hz  : half-bandwidth of the frequency mask for J₃ [Hz]

    Returns
    -------
    nperseg_list : sorted list of nperseg values
    results      : dict  (window_label, overlap_frac)
                           → {nperseg: {"j1": float, "j2": float, "j3": float}}
    """
    fs           = _fs_from_time(t_us)
    nperseg_list = _nperseg_candidates(len(signal))
    results: dict[tuple, dict[int, dict]] = {}

    for window, ovlp in itertools.product(windows, overlaps):
        key = (_window_label(window), ovlp)
        results[key] = {}
        for nperseg in nperseg_list:
            results[key][nperseg] = {
                "j1": _j1(signal, fs, nperseg, window, ovlp, f_center_hz),
                "j2": _j2_yan(signal, fs, nperseg, window, ovlp),
                "j3": _j3(signal, fs, nperseg, window, ovlp,
                           f_center_hz, delta_f_hz,
                           mode_signal, t_us),
            }

    return nperseg_list, results


# ─────────────────────────────────────────────────────────────────────────────
# Best-parameter selection
# ─────────────────────────────────────────────────────────────────────────────

def select_best(
    nperseg_list: list[int],
    results: dict,
    w1: float = 0.3,
    w2: float = 0.3,
    w3: float = 0.4,
) -> tuple[dict, float]:
    """Select (nperseg, window, overlap) that minimises J_tot."""
    combos, j1s, j2s, j3s = [], [], [], []
    for (win_lbl, ovlp), per_np in results.items():
        for nperseg, m in per_np.items():
            combos.append((win_lbl, ovlp, nperseg))
            j1s.append(m["j1"])
            j2s.append(m["j2"])
            j3s.append(m["j3"])

    jtot     = _composite_score(np.array(j1s), np.array(j2s), np.array(j3s), w1, w2, w3)
    best_idx = int(np.argmin(jtot))
    best_win_lbl, best_ovlp, best_np = combos[best_idx]

    # Recover original window spec (string or tuple) for scipy
    win_spec = best_win_lbl
    for w in WINDOWS:
        if _window_label(w) == best_win_lbl:
            win_spec = w
            break

    return {
        "nperseg":      best_np,
        "window":       win_spec,
        "window_label": best_win_lbl,
        "overlap_frac": best_ovlp,
        "j1":           j1s[best_idx],
        "j2":           j2s[best_idx],
        "j3":           j3s[best_idx],
        "j_tot":        float(jtot[best_idx]),
    }, float(jtot[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Yan (2020) Fig-10-style plot
# ─────────────────────────────────────────────────────────────────────────────

_CURVE_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def plot_yan_style(
    nperseg_list: list[int],
    results: dict,
    best: dict,
    title_prefix: str,
) -> None:
    """1 × 3 figure — J₁, J₂, J₃ vs window length, one line per (window, overlap)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{title_prefix}  —  metric vs window length  (Yan 2020 style)",
                 fontsize=12)

    metric_cfg = [
        ("j1", "J₁ (kHz)  ↓ better", axes[0], lambda v: v / 1e3),
        ("j2", "J₂  ↑ better",        axes[1], lambda v: v),
        ("j3", "J₃  ↓ better",        axes[2], lambda v: v),
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

    axes[0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Best-STFT spectrogram plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_stft(
    signal: np.ndarray,
    t_us: np.ndarray,
    best: dict,
    title: str,
) -> None:
    """Side-by-side default vs best-parameter spectrograms."""
    fs              = _fs_from_time(t_us)
    nperseg_list    = _nperseg_candidates(len(signal))
    default_nperseg = min(256, nperseg_list[-1])

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, nperseg_, window_, ovlp_, subtitle):
        noverlap_ = int(nperseg_ * ovlp_)
        f_ax, t_ax, Zxx = scipy_stft(signal, fs=fs, window=window_,
                                      nperseg=nperseg_, noverlap=noverlap_)
        pdb = 20 * np.log10(np.abs(Zxx) + 1e-12)
        im  = ax.pcolormesh(t_ax * 1e6, f_ax / 1e6, pdb,
                            shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="dB")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(subtitle)

    _draw(ax_d, default_nperseg, "hann", 0.75,
          f"Default  (nperseg={default_nperseg}, hann, 75%)")
    _draw(ax_b, best["nperseg"], best["window"], best["overlap_frac"],
          f"Best  np={best['nperseg']}, {best['window_label']}, "
          f"{best['overlap_frac']:.0%}\n"
          f"J₁={best['j1']/1e3:.2f} kHz  "
          f"J₂={best['j2']:.1f}  "
          f"J₃={best['j3']:.3f}  "
          f"J_tot={best['j_tot']:.4f}")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(w1: float = 0.3, w2: float = 0.3, w3: float = 0.4) -> None:
    data    = load_all("304 Steel")
    summary = []

    # Accumulators for J₄
    j1_acc: dict[tuple, list[float]] = {}
    j3_acc: dict[tuple, list[float]] = {}

    # ── Per-distance, per-signal sweep ───────────────────────────────────────
    for d in data:
        dist = d["distance_mm"]
        print(f"\n{'='*64}")
        print(f"  Distance: {dist} mm")
        print(f"{'='*64}")

        for sig_key, signal, t_us, mode_signal in [
            ("f",  d["f_signal"], d["time_f"],  d["s2_mode"]),
            ("2f", d["sig_2f"],   d["time_2f"], d["s4_mode"]),
        ]:
            fs       = _fs_from_time(t_us)
            f_center = _dominant_freq(signal, fs)
            delta_f  = DELTA_F_FRAC * f_center

            print(f"  [{sig_key}]  f_center={f_center/1e6:.3f} MHz  "
                  f"Δf={delta_f/1e3:.1f} kHz  "
                  f"mask from {'S2' if sig_key == 'f' else 'S4'} mode envelope")

            nperseg_list, results = sweep_metrics_vs_window_length(
                signal, t_us,
                mode_signal = mode_signal,
                f_center_hz = f_center,
                delta_f_hz  = delta_f,
            )

            best, _ = select_best(nperseg_list, results, w1, w2, w3)

            # Accumulate for J₄
            for (win_lbl, ovlp), per_np in results.items():
                for nperseg, m in per_np.items():
                    k = (sig_key, win_lbl, ovlp, nperseg)
                    j1_acc.setdefault(k, []).append(m["j1"])
                    j3_acc.setdefault(k, []).append(m["j3"])

            plot_yan_style(nperseg_list, results, best,
                           title_prefix=f"{sig_key} signal — {dist} mm")
            plot_best_stft(signal, t_us, best,
                           title=f"Best STFT — {sig_key} signal  {dist} mm")

            summary.append({"distance_mm": dist, "signal": sig_key, **best})

    # ── J₄ robustness ranking ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  J₄ — most stable configurations across distances (top 5 per signal)")
    print(f"{'='*70}")
    for sig_key in ("f", "2f"):
        rows = []
        for (sk, win_lbl, ovlp, nperseg), j1_vals in j1_acc.items():
            if sk != sig_key:
                continue
            j3_vals = j3_acc[(sk, win_lbl, ovlp, nperseg)]
            rows.append({
                "nperseg": nperseg, "win_lbl": win_lbl, "ovlp": ovlp,
                "j4_j1": _j4(j1_vals), "j4_j3": _j4(j3_vals),
            })
        rows.sort(key=lambda r: r["j4_j1"])
        print(f"\n  Signal: {sig_key}")
        print(f"  {'nperseg':>8}  {'window':<16}  {'overlap':>8}  "
              f"{'J4(J1)':>10}  {'J4(J3)':>10}")
        print(f"  {'-'*8}  {'-'*16}  {'-'*8}  {'-'*10}  {'-'*10}")
        for r in rows[:5]:
            print(f"  {r['nperseg']:>8}  {r['win_lbl']:<16}  "
                  f"{r['ovlp']:>7.0%}  "
                  f"{r['j4_j1']:>10.4f}  "
                  f"{r['j4_j3']:>10.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    col = "{:>6}  {:<4}  {:>8}  {:<16}  {:>8}  {:>10}  {:>12}  {:>8}  {:>8}"
    print(f"\n{'='*90}")
    print("  Best-hyperparameter summary")
    print(f"{'='*90}")
    print("  " + col.format("Dist", "Sig", "nperseg", "window",
                             "overlap", "J1 (kHz)", "J2", "J3", "J_tot"))
    print("  " + col.format(*["-"*w for w in (6, 4, 8, 16, 8, 10, 12, 8, 8)]))
    for r in summary:
        print("  " + col.format(
            f"{r['distance_mm']}mm", r["signal"],
            r["nperseg"], r["window_label"],
            f"{r['overlap_frac']:.0%}",
            f"{r['j1']/1e3:.2f}",
            f"{r['j2']:.1f}",
            f"{r['j3']:.4f}",
            f"{r['j_tot']:.4f}",
        ))


if __name__ == "__main__":
    main()
