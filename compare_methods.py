"""
compare_methods.py
------------------
Cross-method comparison of the *optimised* M_JPL and RV₃ values.

For every (distance, signal) pair this script runs the full hyperparameter
sweep for each TF method, selects the best configuration, then plots the
resulting M_JPL and RV₃ values side-by-side so the five methods can be
directly compared.

Methods
───────
  STFT      — stft_analysis.py
  CWT       — wt_analysis.py
  S-transform — s_transform_analysis.py
  SST-CWT   — sst_cwt_analysis.py
  SST-STFT  — sst_stft_analysis.py

Output figures
──────────────
  Figure 1 :  M_JPL  at optimum — grouped bars, one group per distance,
              one bar per method.  Separate subplot per signal (f / 2f).
  Figure 2 :  RV₃   at optimum — same layout.
  Figure 3 :  Combined 2×2 grid (f M_JPL | f RV₃ / 2f M_JPL | 2f RV₃)
              with a single shared legend.
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from load_signals import load_all

# ── per-method sweep + select_best imports ───────────────────────────────────
import stft_analysis
import wt_analysis
import s_transform_analysis

try:
    import sst_cwt_analysis
    _SST_CWT_OK = sst_cwt_analysis._SSQ_OK
except Exception:
    _SST_CWT_OK = False

try:
    import sst_stft_analysis
    _SST_STFT_OK = sst_stft_analysis._SSQ_OK
except Exception:
    _SST_STFT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────────────

def _run_stft(signal, t_us, mode_signal):
    _, results = stft_analysis.sweep_metrics_vs_window_length(
        signal, t_us, mode_signal)
    return stft_analysis.select_best(results)


def _run_cwt(signal, t_us, mode_signal):
    results = wt_analysis.sweep_metrics(signal, t_us, mode_signal)
    return wt_analysis.select_best(results)


def _run_st(signal, t_us, mode_signal):
    results = s_transform_analysis.sweep_metrics(signal, t_us, mode_signal)
    return s_transform_analysis.select_best(results)


def _run_sst_cwt(signal, t_us, mode_signal):
    if not _SST_CWT_OK:
        return None
    results = sst_cwt_analysis.sweep_metrics(signal, t_us, mode_signal)
    if not results:
        return None
    return sst_cwt_analysis.select_best(results)


def _run_sst_stft(signal, t_us, mode_signal):
    if not _SST_STFT_OK:
        return None
    _, results = sst_stft_analysis.sweep_metrics_vs_window_length(
        signal, t_us, mode_signal)
    if not any(results[k] for k in results):
        return None
    return sst_stft_analysis.select_best(results)


METHODS: list[tuple[str, callable]] = [
    ("STFT",      _run_stft),
    ("CWT",       _run_cwt),
    ("S-Tr",      _run_st),
    ("SST-CWT",   _run_sst_cwt),
    ("SST-STFT",  _run_sst_stft),
]

METHOD_COLOURS = {
    "STFT":     "#1f77b4",
    "CWT":      "#ff7f0e",
    "S-Tr":     "#2ca02c",
    "SST-CWT":  "#d62728",
    "SST-STFT": "#9467bd",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_results(data: list[dict]) -> dict:
    """
    Returns a nested dict:
        out[signal_key]["f" | "2f"][distance_mm][method_name] = {"mjpl": , "rv3": }
    """
    out: dict = {"f": {}, "2f": {}}

    for d in data:
        dist = d["distance_mm"]
        for sig_key, signal, t_us, mode_signal, mode_name in [
            ("f",  d["f_signal"], d["time_f"],  d["s2_mode"], "S2"),
            ("2f", d["sig_2f"],   d["time_2f"], d["s4_mode"], "S4"),
        ]:
            out[sig_key][dist] = {}
            for method_name, run_fn in METHODS:
                print(f"  [{dist}mm  {sig_key}  {method_name}] ...", end=" ", flush=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        best = run_fn(signal, t_us, mode_signal)
                    if best is None:
                        print("skipped")
                        out[sig_key][dist][method_name] = None
                    else:
                        print(f"M_JPL={best['mjpl']:.3e}  RV₃={best['rv3']:.3f}")
                        out[sig_key][dist][method_name] = {
                            "mjpl": best["mjpl"],
                            "rv3":  best["rv3"],
                        }
                except Exception as e:
                    print(f"ERROR: {e}")
                    out[sig_key][dist][method_name] = None

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grouped_bar(ax, distances, results_by_dist, metric_key: str,
                 method_names: list[str], ylabel: str, title: str) -> None:
    """Draw grouped bars on *ax*."""
    n_methods  = len(method_names)
    n_groups   = len(distances)
    bar_width  = 0.8 / n_methods
    x_centres  = np.arange(n_groups)

    for i, mname in enumerate(method_names):
        offsets = x_centres + (i - (n_methods - 1) / 2) * bar_width
        vals = []
        for dist in distances:
            entry = results_by_dist[dist].get(mname)
            vals.append(entry[metric_key] if entry is not None else np.nan)
        ax.bar(offsets, vals, width=bar_width * 0.9,
               color=METHOD_COLOURS.get(mname, "grey"),
               label=mname, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_centres)
    ax.set_xticklabels([f"{d} mm" for d in distances])
    ax.set_xlabel("Propagation distance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_comparison(results: dict) -> None:
    """Produce the three comparison figures."""
    method_names = [m for m, _ in METHODS]
    distances_f  = sorted(results["f"].keys())
    distances_2f = sorted(results["2f"].keys())

    # ── Figure 1 : M_JPL ─────────────────────────────────────────────────────
    fig1, (ax1f, ax1_2f) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Optimised M_JPL — method comparison", fontsize=13)
    _grouped_bar(ax1f,   distances_f,  results["f"],
                 "mjpl", method_names, "M_JPL  ↑ better", "f signal")
    _grouped_bar(ax1_2f, distances_2f, results["2f"],
                 "mjpl", method_names, "M_JPL  ↑ better", "2f signal")
    handles = [mpatches.Patch(color=METHOD_COLOURS[m], label=m)
               for m in method_names]
    fig1.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

    # ── Figure 2 : RV₃ ───────────────────────────────────────────────────────
    fig2, (ax2f, ax2_2f) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Optimised RV₃ — method comparison", fontsize=13)
    _grouped_bar(ax2f,   distances_f,  results["f"],
                 "rv3",  method_names, "RV₃  ↓ better", "f signal")
    _grouped_bar(ax2_2f, distances_2f, results["2f"],
                 "rv3",  method_names, "RV₃  ↓ better", "2f signal")
    fig2.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

    # ── Figure 3 : Combined 2×2 ──────────────────────────────────────────────
    fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig3.suptitle("Method comparison — M_JPL and RV₃ at optimum parameters",
                  fontsize=13)

    cfg = [
        (axes[0, 0], "f",  "mjpl", "M_JPL  ↑ better", "f signal — M_JPL"),
        (axes[0, 1], "f",  "rv3",  "RV₃  ↓ better",   "f signal — RV₃"),
        (axes[1, 0], "2f", "mjpl", "M_JPL  ↑ better",  "2f signal — M_JPL"),
        (axes[1, 1], "2f", "rv3",  "RV₃  ↓ better",   "2f signal — RV₃"),
    ]
    for ax, sig_key, metric, ylabel, title in cfg:
        dists = sorted(results[sig_key].keys())
        _grouped_bar(ax, dists, results[sig_key], metric,
                     method_names, ylabel, title)

    fig3.legend(handles=handles, loc="lower center", ncol=len(method_names),
                fontsize=9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    col = "{:<12}  {:<4}  {:>8}  {:>14}  {:>8}"
    header = col.format("Method", "Sig", "Dist(mm)", "M_JPL", "RV₃")
    sep    = "-" * len(header)
    print(f"\n{'='*60}")
    print("  Best-hyperparameter results — all methods & distances")
    print(f"{'='*60}")
    print("  " + header)
    print("  " + sep)
    for sig_key in ("f", "2f"):
        for dist in sorted(results[sig_key].keys()):
            for method_name in [m for m, _ in METHODS]:
                entry = results[sig_key][dist].get(method_name)
                if entry is None:
                    mjpl_s = "N/A"
                    rv3_s  = "N/A"
                else:
                    mjpl_s = f"{entry['mjpl']:.4e}"
                    rv3_s  = f"{entry['rv3']:.3f}"
                print("  " + col.format(method_name, sig_key,
                                        str(dist), mjpl_s, rv3_s))
        print("  " + sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading signals …")
    data = load_all("304 Steel")
    print(f"Loaded {len(data)} distances: "
          f"{[d['distance_mm'] for d in data]}\n")

    print("Running sweeps for all methods …")
    results = collect_results(data)

    print_summary(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
