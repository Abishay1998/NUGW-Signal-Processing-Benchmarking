"""
sst_cwt_analysis.py
-------------------
CWT-based Synchrosqueezing Transform (SST-CWT) hyperparameter optimisation
for nonlinear guided wave signals using ssqueezepy.

SST-CWT reassigns each CWT coefficient to a sharper instantaneous-frequency
ridge, producing a more concentrated TF representation than the raw CWT.

Sweep parameters
────────────────
  wavelet  : ssqueezepy wavelet name — subset: 'gmw', 'morlet', 'bump'
  nv       : number of voices per octave  {8, 16, 32}

Metrics (Stankovic 2001)
────────────────────────
  M_JPL  (eq. 2)  ↑ better  — optimisation objective
  RV₃    (eq. 4)  ↓ better  — reporting metric
"""

from __future__ import annotations

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from load_signals import load_all

try:
    from ssqueezepy import ssq_cwt, Wavelet
    from ssqueezepy.utils import p2up
    _SSQ_OK = True
except Exception as _e:
    _SSQ_OK = False
    print(f"[warn] ssqueezepy unavailable: {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WAVELET_NAMES: list[str] = ["gmw", "morlet", "bump"]
N_VOICES_LIST: list[int]  = [8, 16, 32]
FREQ_MAX_MHZ: float = 3.0

SIGMA_F_BINS: float = 5.0
SIGMA_T_BINS: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_time(t_us: np.ndarray) -> float:
    return 1.0 / (float(np.mean(np.diff(t_us))) * 1e-6)


def _sst_cwt(signal: np.ndarray, fs: float,
             wavelet_name: str, nv: int,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (freqs_hz, times_s, power) where power = |Tx|² (SST-CWT).
    freqs are in Hz, times in seconds.
    """
    if not _SSQ_OK:
        raise RuntimeError("ssqueezepy is not available.")

    wav = Wavelet(wavelet_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tx, _, ssq_freqs, *_ = ssq_cwt(signal, wav, fs=fs, nv=nv)

    power = np.abs(Tx) ** 2                          # (n_freqs × N)
    times = np.arange(len(signal)) / fs

    # ssq_freqs may be in normalised units [0,1] — convert to Hz
    freqs = np.asarray(ssq_freqs, dtype=float)
    if freqs.max() <= 1.0:
        freqs = freqs * (fs / 2)
    # Clip to freq_max
    mask  = freqs <= FREQ_MAX_MHZ * 1e6
    return freqs[mask], times, power[mask]


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
                    wavelet_name: str, nv: int) -> tuple[int, int]:
    _, _, power = _sst_cwt(mode_signal, fs, wavelet_name, nv)
    idx = np.unravel_index(np.argmax(power), power.shape)
    return int(idx[0]), int(idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _mjpl_global(signal: np.ndarray, fs: float,
                 wavelet_name: str, nv: int,
                 mode_signal: np.ndarray) -> float:
    _, _, rho = _sst_cwt(signal, fs, wavelet_name, nv)
    cf, ct = _mode_tf_center(mode_signal, fs, wavelet_name, nv)
    Q     = _gaussian_Q(rho.shape, cf, ct, SIGMA_F_BINS, SIGMA_T_BINS)
    numer = np.sum(Q**2 * rho**2)
    denom = np.sum(Q    * rho)
    if denom == 0:
        return 0.0
    return float(numer / (denom ** 2))


def _rv3(signal: np.ndarray, fs: float,
         wavelet_name: str, nv: int) -> float:
    _, _, rho = _sst_cwt(signal, fs, wavelet_name, nv)
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
    wavelet_names: list[str] = WAVELET_NAMES,
    n_voices_list: list[int] = N_VOICES_LIST,
) -> dict:
    """results[(wavelet, nv)] = {"mjpl": float, "rv3": float}"""
    fs = _fs_from_time(t_us)
    results: dict[tuple, dict] = {}
    for wav, nv in itertools.product(wavelet_names, n_voices_list):
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
    keys   = sorted(results.keys())
    labels = [f"{w}\nnv={nv}" for w, nv in keys]
    mjpl_v = [results[k]["mjpl"] for k in keys]
    rv3_v  = [results[k]["rv3"]  for k in keys]
    best_lbl = f"{best['wavelet']}\nnv={best['n_voices']}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix}  —  SST-CWT metric sweep", fontsize=12)

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


def plot_best_sst(signal: np.ndarray, t_us: np.ndarray,
                  mode_signal: np.ndarray, best: dict, title: str) -> None:
    """Side-by-side: default SST-CWT (gmw, 16) | best-parameter SST-CWT."""
    fs = _fs_from_time(t_us)

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    def _draw(ax, wav, nv, subtitle, mark_mode=False):
        freqs, times, power = _sst_cwt(signal, fs, wav, nv)
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

    _draw(ax_d, "gmw", 16, "Default  (gmw, 16 voices)")
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
    if not _SSQ_OK:
        print("ssqueezepy not available — cannot run sst_cwt_analysis.")
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
            results = sweep_metrics(signal, t_us, mode_signal)
            if not results:
                print("  No valid results — skipping.")
                continue
            best    = select_best(results)
            plot_sweep(results, best,
                       title_prefix=f"{sig_key} signal — {dist} mm")
            plot_best_sst(signal, t_us, mode_signal, best,
                          title=f"Best SST-CWT — {sig_key} signal  {dist} mm  "
                                f"({mode_name} mode localisation)")
            summary.append({"dist": dist, "signal": sig_key,
                            "mode": mode_name, **best})

    col = "{:>6}  {:<4}  {:<12}  {:>8}  {:>14}  {:>8}"
    print(f"\n{'='*68}\n  Best SST-CWT summary  (objective: max M_JPL)\n{'='*68}")
    print("  " + col.format("Dist", "Sig", "wavelet", "n_voices", "M_JPL", "RV₃"))
    print("  " + col.format(*["-"*w for w in (6, 4, 12, 8, 14, 8)]))
    for r in summary:
        print("  " + col.format(f"{r['dist']}mm", r["signal"],
                                r["wavelet"], r["n_voices"],
                                f"{r['mjpl']:.4e}", f"{r['rv3']:.3f}"))


if __name__ == "__main__":
    main()
