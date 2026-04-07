"""
wt_analysis.py
--------------
CWT hyperparameter optimisation for nonlinear guided wave signals.

Sweep parameters
────────────────
  wavelet_family : pywt.wavelist(kind="continuous")  subset chosen for
                   ultrasonic work: cmor, morl, mexh, cgau, shan
  n_voices       : number of scales per octave  {8, 16, 32}

Metrics (Stankovic 2001)
────────────────────────
  M_JPL  (eq. 2)  — local concentration, Q centred on S2/S4 mode peak  ↑ better
  RV₃    (eq. 4)  — normalised Rényi entropy                            ↓ better

Optimisation: maximise M_JPL_global.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pywt

from load_signals import load_all

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Wavelet families to sweep (continuous wavelets available in PyWavelets)
WAVELET_FAMILIES: list[str] = ["cmor1.5-1.0", "morl", "mexh", "cgau4", "shan1.5-1.0"]
N_VOICES_LIST: list[int] = [8, 16, 32]       # scales per octave
N_OCTAVES: int = 6                             # total octave range
FREQ_MAX_MHZ: float = 3.0                      # y-axis cap

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


def _build_scales(n_voices: int, fs: float, freq_max_hz: float) -> np.ndarray:
    """
    Log-spaced scales covering [~100 kHz, freq_max_hz] for the given fs.
    Uses the 'morl' centre frequency as reference; adjusted per family at
    call time by the caller.
    """
    # Generic: scales = f_c * fs / freqs
    # We return the scale array; centre-frequency adjustment is done in _cwt.
    f_min = 0.1e6
    f_max = min(freq_max_hz, fs / 2.1)
    freqs  = np.geomspace(f_min, f_max, N_OCTAVES * n_voices)
    return freqs   # return frequency array; convert to scales inside _cwt


def _cwt(signal: np.ndarray, fs: float, wavelet: str,
         n_voices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CWT scalogram.  Returns (freqs_hz, times_s, power) where
    power = |CWT|² (shape: n_freqs × n_samples).
    """
    try:
        w = pywt.ContinuousWavelet(wavelet)
        fc = pywt.central_frequency(w)
    except Exception:
        fc = 1.0
    freqs = _build_scales(n_voices, fs, FREQ_MAX_MHZ * 1e6)
    scales = fc * fs / freqs
    scales = scales[scales > 0]
    coefs, freqs_out = pywt.cwt(signal, scales=scales,
                                 wavelet=wavelet, sampling_period=1.0 / fs)
    power = np.abs(coefs) ** 2
    times = np.arange(signal.shape[0]) / fs
    return freqs_out, times, power


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
                    wavelet: str, n_voices: int) -> tuple[int, int]:
    _, _, power = _cwt(mode_signal, fs, wavelet, n_voices)
    idx = np.unravel_index(np.argmax(power), power.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _mjpl_global(signal: np.ndarray, fs: float, wavelet: str, n_voices: int,
                 mode_signal: np.ndarray) -> float:
    _, _, rho = _cwt(signal, fs, wavelet, n_voices)
    cf, ct = _mode_tf_center(mode_signal, fs, wavelet, n_voices)
    Q  = _gaussian_Q(rho.shape, cf, ct, SIGMA_F_BINS, SIGMA_T_BINS)
    Q2 = Q ** 2
    numer = np.sum(Q2 * rho**2)
    denom = np.sum(Q  * rho)
    if denom == 0:
        return 0.0
    return float(numer / (denom ** 2))


def _rv3(signal: np.ndarray, fs: float, wavelet: str, n_voices: int) -> float:
    _, _, rho = _cwt(signal, fs, wavelet, n_voices)
    vol = np.sum(np.abs(rho))
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
    wavelet_families: list[str] = WAVELET_FAMILIES,
    n_voices_list: list[int] = N_VOICES_LIST,
) -> dict:
    """
    Returns results[( wavelet, n_voices )] = {"mjpl": float, "rv3": float}
    """
    fs = _fs_from_time(t_us)
    results: dict[tuple, dict] = {}
    for wav, nv in itertools.product(wavelet_families, n_voices_list):
        try:
            results[(wav, nv)] = {
                "mjpl": _mjpl_global(signal, fs, wav, nv, mode_signal),
                "rv3":  _rv3(signal, fs, wav, nv),
            }
        except Exception as e:
            print(f"  [skip] {wav} nv={nv}: {e}")
    return results


def select_best(results: dict) -> dict:
    best_key, best_mjpl, best_rv3 = None, -np.inf, None
    for key, m in results.items():
        if m["mjpl"] > best_mjpl:
            best_mjpl = m["mjpl"]
            best_key  = key
            best_rv3  = m["rv3"]
    wav, nv = best_key
    return {"wavelet": wav, "n_voices": nv, "mjpl": best_mjpl, "rv3": best_rv3}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(results: dict, best: dict, title_prefix: str) -> None:
    """Bar-chart style: M_JPL and RV₃ for each (wavelet, n_voices) combo."""
    keys    = sorted(results.keys())
    labels  = [f"{w}\nnv={nv}" for w, nv in keys]
    mjpl_v  = [results[k]["mjpl"] for k in keys]
    rv3_v   = [results[k]["rv3"]  for k in keys]
    best_lbl = f"{best['wavelet']}\nnv={best['n_voices']}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix}  —  CWT metric sweep", fontsize=12)

    for ax, vals, ylabel in [
        (axes[0], mjpl_v, "M_JPL  ↑ better"),
        (axes[1], rv3_v,  "RV₃  ↓ better"),
    ]:
        colours = ["#d62728" if lbl == best_lbl else "#1f77b4" for lbl in labels]
        ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Wavelet / voices")
        plt.setp(ax.get_xticklabels(), fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_best_cwt(signal: np.ndarray, t_us: np.ndarray,
                  mode_signal: np.ndarray, best: dict, title: str) -> None:
    """Side-by-side: default CWT (morl, 16) | best-parameter CWT."""
    fs = _fs_from_time(t_us)

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, wav, nv, subtitle, mark_mode=False):
        freqs, times, power = _cwt(signal, fs, wav, nv)
        pdb  = 10 * np.log10(power / (power.max() + 1e-30) + 1e-30)
        im   = ax.pcolormesh(times * 1e6, freqs / 1e6, pdb,
                             shading="auto", cmap="inferno",
                             vmin=-50, vmax=0, cmap="jet")
        fig.colorbar(im, ax=ax, label="Power (dB, rel. max)")
        ax.set_ylim(0, FREQ_MAX_MHZ)
        if mark_mode:
            cf, ct = _mode_tf_center(mode_signal, fs, wav, nv)
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

    _draw(ax_d, "morl", 16, "Default  (morl, 16 voices)")
    _draw(ax_b, best["wavelet"], best["n_voices"],
          f"Best  {best['wavelet']}, nv={best['n_voices']}\n"
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
            plot_best_cwt(signal, t_us, mode_signal, best,
                          title=f"Best CWT — {sig_key} signal  {dist} mm  "
                                f"({mode_name} mode localisation)")
            summary.append({"dist": dist, "signal": sig_key,
                            "mode": mode_name, **best})

    col = "{:>6}  {:<4}  {:<20}  {:>8}  {:>14}  {:>8}"
    print(f"\n{'='*72}\n  Best CWT hyperparameter summary  (objective: max M_JPL)\n{'='*72}")
    print("  " + col.format("Dist", "Sig", "wavelet", "n_voices", "M_JPL", "RV₃"))
    print("  " + col.format(*["-"*w for w in (6, 4, 20, 8, 14, 8)]))
    for r in summary:
        print("  " + col.format(f"{r['dist']}mm", r["signal"],
                                r["wavelet"], r["n_voices"],
                                f"{r['mjpl']:.4e}", f"{r['rv3']:.3f}"))


if __name__ == "__main__":
    main()
