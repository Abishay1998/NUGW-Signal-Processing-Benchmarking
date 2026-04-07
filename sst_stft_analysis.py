"""
sst_stft_analysis.py
--------------------
STFT-based Synchrosqueezing Transform (SST-STFT) hyperparameter optimisation
for nonlinear guided wave signals using ssqueezepy.

SST-STFT reassigns STFT coefficients along the frequency axis using the
instantaneous frequency estimate, giving a sharper TF representation than
a plain STFT while retaining the invertibility of the STFT.

Sweep parameters
────────────────
  nperseg : STFT window length  (same log-spaced candidates as stft_analysis)
  window  : window function  — same WINDOWS as stft_analysis
  overlap : overlap fraction  — OVERLAPS

Metrics (Stankovic 2001)
────────────────────────
  M_JPL  (eq. 2)  ↑ better  — optimisation objective
  RV₃    (eq. 4)  ↓ better  — reporting metric
"""

from __future__ import annotations

import itertools
import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from load_signals import load_all

try:
    from ssqueezepy import ssq_stft
    _SSQ_OK = True
except Exception as _e:
    _SSQ_OK = False
    print(f"[warn] ssqueezepy unavailable: {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (mirror stft_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS: list = ["hann", "hamming", "blackman", ("tukey", 0.25)]
OVERLAPS: tuple[float, ...] = (0.50, 0.25)
N_NPERSEG: int = 10
FREQ_MAX_MHZ: float = 3.0

SIGMA_F_BINS: float = 5.0
SIGMA_T_BINS: float = 10.0

_CURVE_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    return 1.0 / (float(np.mean(np.diff(t_us))) * 1e-6)


def _nperseg_candidates(n_samples: int) -> list[int]:
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


def _sst_stft_compute(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    window,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SST-STFT.  Returns (freqs_hz, times_s, power).
    power = |Tx|² where Tx is the synchrosqueezed STFT.
    """
    if not _SSQ_OK:
        raise RuntimeError("ssqueezepy is not available.")

    noverlap = int(nperseg * overlap_frac)
    hop      = nperseg - noverlap

    win_arr  = _make_window(window, nperseg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tx, _, ssq_freqs, *_ = ssq_stft(
            signal,
            window=win_arr,
            n_fft=nperseg,
            hop_len=hop,
            fs=fs,
        )

    power = np.abs(Tx) ** 2
    times = np.arange(Tx.shape[1]) * hop / fs

    freqs = np.asarray(ssq_freqs, dtype=float)
    if freqs.max() <= 1.0:
        freqs = freqs * (fs / 2)

    mask  = freqs <= FREQ_MAX_MHZ * 1e6
    return freqs[mask], times, power[mask]


def _make_window(window, nperseg: int) -> np.ndarray:
    """Convert window spec to numpy array for ssqueezepy."""
    from scipy.signal import get_window
    return get_window(window, nperseg)


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
                    nperseg: int, window, overlap_frac: float) -> tuple[int, int]:
    _, _, power = _sst_stft_compute(mode_signal, fs, nperseg, window, overlap_frac)
    idx = np.unravel_index(np.argmax(power), power.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _mjpl_global(signal: np.ndarray, fs: float,
                 nperseg: int, window, overlap_frac: float,
                 mode_signal: np.ndarray, t_us: np.ndarray) -> float:
    _, _, rho = _sst_stft_compute(signal, fs, nperseg, window, overlap_frac)
    cf, ct = _mode_tf_center(mode_signal, fs, nperseg, window, overlap_frac)
    Q     = _gaussian_Q(rho.shape, cf, ct, SIGMA_F_BINS, SIGMA_T_BINS)
    numer = np.sum(Q**2 * rho**2)
    denom = np.sum(Q    * rho)
    if denom == 0:
        return 0.0
    return float(numer / (denom ** 2))


def _rv3(signal: np.ndarray, fs: float,
         nperseg: int, window, overlap_frac: float) -> float:
    _, _, rho = _sst_stft_compute(signal, fs, nperseg, window, overlap_frac)
    vol = np.sum(np.abs(rho))
    if vol == 0:
        return np.inf
    rho_norm = np.maximum(rho / vol, 1e-300)
    return float(-0.5 * np.log2(np.sum(rho_norm ** 3)))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_metrics_vs_window_length(
    signal: np.ndarray,
    t_us: np.ndarray,
    mode_signal: np.ndarray,
    overlaps: Sequence[float] = OVERLAPS,
    windows: list = WINDOWS,
) -> tuple[list[int], dict]:
    """
    Returns (nperseg_list, results) where
    results[(win_lbl, ovlp)][nperseg] = {"mjpl": float, "rv3": float}
    """
    fs           = _fs_from_time(t_us)
    nperseg_list = _nperseg_candidates(len(signal))
    results: dict[tuple, dict[int, dict]] = {}

    for window, ovlp in itertools.product(windows, overlaps):
        key = (_window_label(window), ovlp)
        results[key] = {}
        for nperseg in nperseg_list:
            try:
                results[key][nperseg] = {
                    "mjpl": _mjpl_global(signal, fs, nperseg, window, ovlp,
                                         mode_signal, t_us),
                    "rv3":  _rv3(signal, fs, nperseg, window, ovlp),
                }
            except Exception as e:
                print(f"  [skip] {key} np={nperseg}: {e}")

    return nperseg_list, results


def select_best(results: dict) -> dict:
    best_key, best_np, best_mjpl, best_rv3 = None, None, -np.inf, None
    for (win_lbl, ovlp), per_np in results.items():
        for nperseg, m in per_np.items():
            if m["mjpl"] > best_mjpl:
                best_mjpl  = m["mjpl"]
                best_key   = (win_lbl, ovlp)
                best_np    = nperseg
                best_rv3   = m["rv3"]

    win_lbl, ovlp = best_key
    win_spec = win_lbl
    for w in WINDOWS:
        if _window_label(w) == win_lbl:
            win_spec = w
            break

    return {"nperseg": best_np, "window": win_spec, "window_label": win_lbl,
            "overlap_frac": ovlp, "mjpl": best_mjpl, "rv3": best_rv3}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(nperseg_list: list[int], results: dict,
               best: dict, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title_prefix}  —  SST-STFT metric vs window length",
                 fontsize=12)

    metric_cfg = [
        ("mjpl", "M_JPL  ↑ better", axes[0]),
        ("rv3",  "RV₃  ↓ better",   axes[1]),
    ]

    for idx, (win_lbl, ovlp) in enumerate(sorted(results.keys())):
        colour = _CURVE_COLOURS[idx % len(_CURVE_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        label  = _curve_label(win_lbl, ovlp)
        per_np = results[(win_lbl, ovlp)]
        nps    = sorted(per_np.keys())

        for metric_key, ylabel, ax in metric_cfg:
            ys = [per_np[n][metric_key] for n in nps if metric_key in per_np[n]]
            ax.plot(nps[:len(ys)], ys, marker=marker, color=colour,
                    linewidth=1.4, markersize=5, label=label)

    best_np   = best["nperseg"]
    best_wlbl = best["window_label"]
    best_ovlp = best["overlap_frac"]

    for metric_key, ylabel, ax in metric_cfg:
        ax.axvline(best_np, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"best np={best_np}")
        per_np_best = results.get((best_wlbl, best_ovlp), {})
        if best_np in per_np_best and metric_key in per_np_best[best_np]:
            ax.plot(best_np, per_np_best[best_np][metric_key],
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


def plot_best_sst_stft(signal: np.ndarray, t_us: np.ndarray,
                       mode_signal: np.ndarray, best: dict, title: str) -> None:
    """Side-by-side: default SST-STFT (hann, 256, 75%) | best params."""
    fs              = _fs_from_time(t_us)
    nperseg_list    = _nperseg_candidates(len(signal))
    default_nperseg = min(256, nperseg_list[-1])

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, nperseg_, window_, ovlp_, subtitle, mark_mode=False):
        try:
            freqs, times, power = _sst_stft_compute(signal, fs, nperseg_,
                                                     window_, ovlp_)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center")
            ax.set_title(subtitle)
            return
        pdb = 10 * np.log10(power + 1e-30)
        im  = ax.pcolormesh(times * 1e6, freqs / 1e6, pdb,
                            shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="dB")
        ax.set_ylim(0, FREQ_MAX_MHZ)
        if mark_mode:
            try:
                cf, ct = _mode_tf_center(mode_signal, fs, nperseg_, window_, ovlp_)
                ax.plot(times[ct] * 1e6, freqs[cf] / 1e6,
                        "+", color="cyan", markersize=16, markeredgewidth=2,
                        label="mode peak (Q centre)")
                ax.legend(fontsize=8, loc="upper right")
            except Exception:
                pass
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Frequency (MHz)")
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
    if not _SSQ_OK:
        print("ssqueezepy not available — cannot run sst_stft_analysis.")
        return

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
            nperseg_list, results = sweep_metrics_vs_window_length(
                signal, t_us, mode_signal)
            if not any(results[k] for k in results):
                print("  No valid results — skipping.")
                continue
            best = select_best(results)
            plot_sweep(nperseg_list, results, best,
                       title_prefix=f"{sig_key} signal — {dist} mm")
            plot_best_sst_stft(signal, t_us, mode_signal, best,
                               title=f"Best SST-STFT — {sig_key} signal  {dist} mm  "
                                     f"({mode_name} mode localisation)")
            summary.append({"dist": dist, "signal": sig_key,
                            "mode": mode_name, **best})

    col = "{:>6}  {:<4}  {:>8}  {:<16}  {:>8}  {:>14}  {:>8}"
    print(f"\n{'='*80}\n  Best SST-STFT summary  (objective: max M_JPL)\n{'='*80}")
    print("  " + col.format("Dist", "Sig", "nperseg", "window",
                             "overlap", "M_JPL", "RV₃"))
    print("  " + col.format(*["-"*w for w in (6, 4, 8, 16, 8, 14, 8)]))
    for r in summary:
        print("  " + col.format(
            f"{r['dist']}mm", r["signal"], r["nperseg"],
            r["window_label"], f"{r['overlap_frac']:.0%}",
            f"{r['mjpl']:.4e}", f"{r['rv3']:.3f}",
        ))


if __name__ == "__main__":
    main()
