# Copilot Instructions

This project compares signal processing techniques for nonlinear guided waves.

## Project Structure
- `signals/` — synthetic nonlinear guided wave signal generation
- `methods/` — individual analysis method implementations (FFT, STFT, Wavelet, HHT, SHG)
- `utils/` — plotting helpers and comparison metrics
- `main.py` — top-level comparison runner
- `requirements.txt` — Python dependencies

## Coding Conventions
- Use NumPy/SciPy for numerical work
- Use Matplotlib for all plots
- Each method in `methods/` must expose a `analyse(signal, fs)` function returning a dict of results
- Keep signal generation and analysis cleanly separated
