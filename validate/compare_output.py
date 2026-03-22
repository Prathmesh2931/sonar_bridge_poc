#!/usr/bin/env python3
"""
validate/compare_output.py

Simple numpy reference for the sonar pipeline:
- Eq.14 → scatter amplitude
- Eq.8  → phase accumulation (frequency domain)
- IFFT  → time → range

Used to sanity-check GPU output.

Run:
    cargo run --release -- --export-bin output/sonar_intensity.bin
    python3 validate/compare_output.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# Config (keep in sync with main.rs)
# -----------------------------------------------------------------------------
N_BEAMS     = 96
N_RAYS      = 512
N_FREQ      = 256
SOUND_SPEED = 1500.0
BANDWIDTH   = 29_500_000.0
MAX_RANGE   = 30.0
H_FOV       = np.radians(120.0)
V_FOV       = np.radians(20.0)
MU_DEFAULT  = 0.5
R_FLOOR     = 10.0
SEED        = 42
FRAME       = 0


# Reference pipeline (Eq.14 + Eq.8 + IFFT)
def run_reference(add_speckle: bool = False) -> np.ndarray:
    """
    Returns intensity [n_beams, n_freq]

    Eq.14 → amplitude per ray
    Eq.8  → accumulate phase over frequency
    IFFT  → convert to time domain (range)
    """
    rng = np.random.default_rng(SEED + FRAME)

    # Angle grids
    beam_angles = H_FOV * (np.arange(N_BEAMS) / max(N_BEAMS - 1, 1) - 0.5)
    ray_angles  = V_FOV * (np.arange(N_RAYS)  / max(N_RAYS  - 1, 1) - 0.5)

    # Flat floor setup
    depth = np.full((N_BEAMS, N_RAYS), R_FLOOR, dtype=np.float32)
    reflectivity = np.full((N_BEAMS, N_RAYS), MU_DEFAULT, dtype=np.float32)

    normals = np.zeros((N_BEAMS, N_RAYS, 3), dtype=np.float32)
    normals[:, :, 1] = 1.0  # facing sensor

    # Frequency grid (same convention as WGSL)
    delta_f = BANDWIDTH / N_FREQ
    f_idx = np.arange(N_FREQ, dtype=np.float64)

    if N_FREQ % 2 == 0:
        freqs = delta_f * (-N_FREQ + 2.0 * (f_idx + 1.0)) / 2.0
    else:
        freqs = delta_f * (-(N_FREQ - 1.0) + 2.0 * (f_idx + 1.0)) / 2.0

    spectrum = np.zeros((N_BEAMS, N_FREQ), dtype=complex)

    for b in range(N_BEAMS):
        for ray in range(N_RAYS):
            r = depth[b, ray]
            if r <= 1e-3 or r > MAX_RANGE:
                continue

            # Ray direction (sensor frame)
            ba = beam_angles[b]
            ra = ray_angles[ray]

            rd = np.array([
                np.sin(ba) * np.cos(ra),
                np.sin(ra),
                np.cos(ba) * np.cos(ra)
            ])
            rd /= np.linalg.norm(rd)

            # Incidence angle (Eq.14)
            cos_inc = abs(np.dot(rd, normals[b, ray]))

            # Area element per ray (Eq.14)
            dth = H_FOV / max(N_BEAMS - 1, 1)
            dphi = V_FOV / max(N_RAYS - 1, 1)
            dA = r**2 * dth * dphi

            # Simple spreading loss
            TL = 1.0 / r

            # Eq.14 amplitude (deterministic part)
            mu = reflectivity[b, ray]
            A_det = np.sqrt(mu) * cos_inc * np.sqrt(dA) * TL

            if add_speckle:
                # Complex Gaussian noise → speckle
                xi = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                amp = A_det * xi
            else:
                amp = A_det

            # Eq.8 phase accumulation
            k = 2 * np.pi * freqs / SOUND_SPEED
            phi = 2 * r * k

            spectrum[b] += amp * np.exp(1j * phi)

    # IFFT → time domain
    p_time = np.fft.ifft(spectrum, axis=1) * N_FREQ

    # Intensity
    return np.abs(p_time) ** 2


# Helpers
def range_axis():
    return np.arange(N_FREQ) * SOUND_SPEED / (2.0 * BANDWIDTH)


def expected_bin(range_m: float) -> int:
    # Expected peak bin from r = t*c/(2B)
    t = int(round(range_m * 2.0 * BANDWIDTH / SOUND_SPEED)) % N_FREQ
    return t


# Main
def main():

    print("Running reference (no speckle)...")
    I_ref = run_reference(add_speckle=False)

    print("Running reference (with speckle)...")
    I_speckle = run_reference(add_speckle=True)

    r_axis = range_axis()
    range_res = SOUND_SPEED / (2 * BANDWIDTH)
    exp_bin = expected_bin(R_FLOOR)

    print(f"\nRange resolution: {range_res:.6f} m/bin")
    print(f"Expected bin: {exp_bin}")
    print(f"Peak (no speckle): {np.argmax(I_ref.sum(axis=0))}")
    print(f"Peak (speckle):    {np.argmax(I_speckle.sum(axis=0))}")

    # Load GPU output if present
    gpu_file = "output/sonar_intensity.bin"
    I_gpu = None

    if os.path.exists(gpu_file):
        print("\nLoading GPU output...")
        I_gpu = np.fromfile(gpu_file, dtype=np.float32).reshape(N_BEAMS, N_FREQ)

        gpu_peak = np.argmax(I_gpu.sum(axis=0))
        print(f"GPU peak: {gpu_peak} (expected {exp_bin})")

        diff = abs(gpu_peak - exp_bin)
        if diff <= 1:
            print(f"PASS (diff={diff})")
        else:
            print(f"FAIL (diff={diff})")

    else:
        print("\n[info] GPU output not found")

    # ---- plotting ----
    ncols = 3 if I_gpu is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    # Reference (clean)
    ax = axes[0]
    im = ax.imshow(I_ref.T, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
    ax.set_title('Reference (no speckle)')
    ax.set_xlabel('Beam'); ax.set_ylabel('Range [m]')
    ax.axhline(R_FLOOR, color='cyan', ls='--')
    plt.colorbar(im, ax=ax)

    # Reference (speckle)
    ax = axes[1]
    im = ax.imshow(I_speckle.T, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
    ax.set_title('Reference (speckle)')
    ax.set_xlabel('Beam')
    ax.axhline(R_FLOOR, color='cyan', ls='--')
    plt.colorbar(im, ax=ax)

    # GPU (if exists)
    if I_gpu is not None:
        ax = axes[2]
        im = ax.imshow(I_gpu.T, aspect='auto', cmap='hot', origin='lower',
                       extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
        ax.set_title('GPU output')
        ax.set_xlabel('Beam')
        ax.axhline(R_FLOOR, color='cyan', ls='--')
        plt.colorbar(im, ax=ax)

        # Compare profiles (ignore speckle noise)
        ref = I_ref.sum(axis=0)
        gpu = I_gpu.sum(axis=0)

        ref /= (ref.max() + 1e-12)
        gpu /= (gpu.max() + 1e-12)

        err = np.max(np.abs(ref - gpu))
        print(f"\nProfile error: {err:.4f}")

    plt.tight_layout()
    os.makedirs("validate", exist_ok=True)
    plt.savefig("validate/comparison.png", dpi=150)
    print("\nSaved: validate/comparison.png")
    plt.show()


if __name__ == "__main__":
    main()