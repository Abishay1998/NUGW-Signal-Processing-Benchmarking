"""
load_signals.py
---------------
Loads simulated guided wave signals from the 'Simulated Signals' folder.

For each propagation distance the function returns:
  - time_f   : time vector for the fundamental frequency (µs)
  - f_signal : sum of In-plane + Out-of-plane 'Sum Propagated signal' at f
  - time_2f  : time vector for the second harmonic frequency (µs)
  - sig_2f   : sum of In-plane + Out-of-plane 'Sum Propagated signal' at 2f

File naming convention (inside each <distance>/ folder):
  In-plane_TemporalResponse@...          → fundamental, in-plane
  Out-of-plane_TemporalResponse@...      → fundamental, out-of-plane
  In-plane_A2_TemporalResponse@...       → 2nd harmonic, in-plane
  Out-of-plane_A2_TemporalResponse@...   → 2nd harmonic, out-of-plane

The first column is always 'Propagation time (micsec)'.
The last column is always 'Sum Propagated signal (nm)'.
In-plane and out-of-plane share the same time axis within each frequency group,
so they are summed directly (no interpolation needed).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import openpyxl


# Root folder containing material subfolders
DATA_ROOT = Path(__file__).parent / "Simulated Signals"


def _read_time_and_sum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the first (time) and last (Sum Propagated signal) columns from an
    Excel file, skipping the header row.

    Returns
    -------
    time   : ndarray (µs)
    signal : ndarray (nm)
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # skip header

    time_vals, sig_vals = [], []
    for row in rows:
        t = row[0]
        s = row[-1]
        if t is not None and s is not None:
            time_vals.append(float(t))
            sig_vals.append(float(s))

    wb.close()
    return np.array(time_vals), np.array(sig_vals)


def load_distance(distance_dir: Path) -> dict:
    """
    Load all four files for one propagation distance and combine them.

    Parameters
    ----------
    distance_dir : Path to the distance folder (e.g. .../200mm/)

    Returns
    -------
    dict with keys:
        distance_mm : int   — propagation distance in mm
        time_f      : ndarray (µs)  — time axis for fundamental
        f_signal    : ndarray (nm)  — in-plane + out-of-plane sum at f
        time_2f     : ndarray (µs)  — time axis for 2nd harmonic
        sig_2f      : ndarray (nm)  — in-plane + out-of-plane sum at 2f
    """
    files = list(distance_dir.glob("*.xlsx"))
    if len(files) != 4:
        raise FileNotFoundError(
            f"Expected 4 .xlsx files in {distance_dir}, found {len(files)}: "
            + ", ".join(f.name for f in files)
        )

    # Classify files by name
    ip_f = oop_f = ip_2f = oop_2f = None
    for f in files:
        name = f.name
        is_a2 = "A2" in name
        is_ip = name.startswith("In-plane")
        if is_ip and not is_a2:
            ip_f = f
        elif not is_ip and not is_a2:
            oop_f = f
        elif is_ip and is_a2:
            ip_2f = f
        elif not is_ip and is_a2:
            oop_2f = f

    for label, path in [("In-plane f", ip_f), ("Out-of-plane f", oop_f),
                        ("In-plane 2f", ip_2f), ("Out-of-plane 2f", oop_2f)]:
        if path is None:
            raise FileNotFoundError(f"Could not find '{label}' file in {distance_dir}")

    # Load and combine
    time_f, ip_f_sig   = _read_time_and_sum(ip_f)
    _,      oop_f_sig  = _read_time_and_sum(oop_f)
    time_2f, ip_2f_sig  = _read_time_and_sum(ip_2f)
    _,       oop_2f_sig = _read_time_and_sum(oop_2f)

    f_signal = ip_f_sig + oop_f_sig
    sig_2f   = ip_2f_sig + oop_2f_sig

    # Parse distance from folder name (e.g. "200mm" → 200)
    distance_mm = int("".join(filter(str.isdigit, distance_dir.name)))

    return {
        "distance_mm": distance_mm,
        "time_f":      time_f,
        "f_signal":    f_signal,
        "time_2f":     time_2f,
        "sig_2f":      sig_2f,
    }


def load_all(material: str = "304 Steel") -> list[dict]:
    """
    Load data for all distances under a given material folder.

    Parameters
    ----------
    material : subfolder name under DATA_ROOT (default: '304 Steel')

    Returns
    -------
    List of dicts (one per distance), sorted by distance_mm.
    Each dict has keys: distance_mm, time_f, f_signal, time_2f, sig_2f.
    """
    material_dir = DATA_ROOT / material
    if not material_dir.exists():
        raise FileNotFoundError(f"Material folder not found: {material_dir}")

    distance_dirs = sorted(
        [d for d in material_dir.iterdir() if d.is_dir()],
        key=lambda d: int("".join(filter(str.isdigit, d.name)))
    )

    if not distance_dirs:
        raise FileNotFoundError(f"No distance subfolders found in {material_dir}")

    results = [load_distance(d) for d in distance_dirs]
    print(f"Loaded {len(results)} distances for '{material}':")
    for r in results:
        print(f"  {r['distance_mm']:>4d} mm | "
              f"f signal: {len(r['time_f'])} pts, "
              f"t=[{r['time_f'][0]:.2f}, {r['time_f'][-1]:.2f}] µs | "
              f"2f signal: {len(r['time_2f'])} pts, "
              f"t=[{r['time_2f'][0]:.2f}, {r['time_2f'][-1]:.2f}] µs")
    return results


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all("304 Steel")

    # Spot-check: print peak amplitude at each distance
    print("\nPeak amplitudes (nm):")
    print(f"  {'Distance':>8}  {'|f_signal| max':>16}  {'|sig_2f| max':>14}")
    for d in data:
        print(f"  {d['distance_mm']:>5} mm   "
              f"{np.max(np.abs(d['f_signal'])):>16.6f}   "
              f"{np.max(np.abs(d['sig_2f'])):>14.6f}")
