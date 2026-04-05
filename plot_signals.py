"""
plot_signals.py
---------------
Loads all simulated guided wave signals and plots the f and 2f signals
for each propagation distance (200mm, 250mm, 300mm, 350mm).

Each distance gets one figure with two subplots:
  - Top    : combined signal at fundamental frequency f
  - Bottom : combined signal at second harmonic 2f
"""

import matplotlib.pyplot as plt
from load_signals import load_all


def plot_signals(data: list[dict]) -> None:
    for d in data:
        dist  = d["distance_mm"]
        t_f   = d["time_f"]
        f_sig = d["f_signal"]
        t_2f  = d["time_2f"]
        sig_2f = d["sig_2f"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        fig.suptitle(f"304 Steel — {dist} mm propagation distance", fontsize=13)

        # Fundamental (f)
        ax1.plot(t_f, f_sig, color="#2563EB", linewidth=0.8)
        ax1.set_ylabel("Amplitude (nm)", fontsize=10)
        ax1.set_title("Fundamental frequency (f)  —  In-plane + Out-of-plane", fontsize=10)
        ax1.set_xlabel("Propagation time (µs)", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Second harmonic (2f)
        ax2.plot(t_2f, sig_2f, color="#DC2626", linewidth=0.8)
        ax2.set_ylabel("Amplitude (nm)", fontsize=10)
        ax2.set_title("Second harmonic (2f)  —  In-plane + Out-of-plane", fontsize=10)
        ax2.set_xlabel("Propagation time (µs)", fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = load_all("304 Steel")
    plot_signals(data)
