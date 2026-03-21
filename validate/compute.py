#!/usr/bin/env python3
"""
validate/compare_output.py
--------------------------
Numpy reference implementation of Choi et al. (2021) Eq.14 + Eq.8 + IFFT.
Runs the same flat-floor synthetic scene as main.rs Stage 3 and compares
output against the wgpu GPU pipeline.

Usage:
    # First run the Rust demo and export binary output:
    cargo run --release -- --export-bin output/sonar_intensity.bin
    # Then:
    python3 validate/compare_output.py

If the GPU output file is absent, the script still runs the numpy reference
and shows the expected range image — useful for proposal demo screenshots.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# Config — must match main.rs Stage 2/3
# =============================================================================
N_BEAMS     = 96
N_RAYS      = 512
N_FREQ      = 256          # FFT bins (power-of-2)
SOUND_SPEED = 1500.0       # c [m/s]
BANDWIDTH   = 29_500_000.0 # B [Hz]  (29.5 MHz NPS sonar)
MAX_RANGE   = 30.0         # [m]
H_FOV       = np.radians(120.0)
V_FOV       = np.radians(20.0)
MU_DEFAULT  = 0.5          # reflectivity
R_FLOOR     = 10.0         # floor range [m]
SEED        = 42
FRAME       = 0

# =============================================================================
# Numpy reference: Choi et al. Eq.14 + Eq.8 + IFFT
# Deterministic (no speckle RNG) for clean comparison
# =============================================================================
def run_reference(add_speckle: bool = False) -> np.ndarray:
    """
    Returns intensity array [n_beams, n_freq] — |p_j(t)|^2.

    Eq.14: a_i = sqrt(mu_i) * cos(alpha_i) * sqrt(dA_i) * TL(r_i)  [deterministic]
           Full: a_i *= (xi_re + i*xi_im) / sqrt(2)  when add_speckle=True

    Eq.8:  P_j(f) = sum_i [ a_i * exp(i * 2 * k(f) * r_i) ]
           k(f) = 2*pi*f / c

    Range: p_j(t) = IFFT{ P_j(f) }
           I_j(t) = |p_j(t)|^2
           range_t = t * c / (2 * B)
    """
    rng = np.random.default_rng(SEED + FRAME)

    # Beam and ray angle grids
    beam_angles = H_FOV * (np.arange(N_BEAMS) / max(N_BEAMS - 1, 1) - 0.5)  # [N_BEAMS]
    ray_angles  = V_FOV * (np.arange(N_RAYS)  / max(N_RAYS  - 1, 1) - 0.5)  # [N_RAYS]

    # Synthetic depth: flat floor at R_FLOOR for all (beam, ray)
    depth       = np.full((N_BEAMS, N_RAYS), R_FLOOR, dtype=np.float32)
    reflectivity = np.full((N_BEAMS, N_RAYS), MU_DEFAULT, dtype=np.float32)
    # Surface normal pointing toward sensor (y-up in sensor frame)
    normals      = np.zeros((N_BEAMS, N_RAYS, 3), dtype=np.float32)
    normals[:, :, 1] = 1.0

    # Frequency grid (DC-centred, matches backscatter.wgsl convention)
    delta_f = BANDWIDTH / N_FREQ
    if N_FREQ % 2 == 0:
        freqs = delta_f * (-N_FREQ + 2 * (np.arange(N_FREQ) + 1)) / 2.0
    else:
        freqs = delta_f * (-(N_FREQ - 1) + 2 * (np.arange(N_FREQ) + 1)) / 2.0

    spectrum = np.zeros((N_BEAMS, N_FREQ), dtype=complex)

    for b in range(N_BEAMS):
        for ray in range(N_RAYS):
            r = depth[b, ray]
            if r <= 0.001 or r > MAX_RANGE:
                continue

            # Ray direction (sensor frame)
            ba = beam_angles[b];  ra = ray_angles[ray]
            rd = np.array([np.sin(ba)*np.cos(ra), np.sin(ra), np.cos(ba)*np.cos(ra)])
            rd = rd / np.linalg.norm(rd)

            # Incidence angle cosine: |ray_dir · surface_normal|  (Eq.14)
            n_vec  = normals[b, ray]
            cos_inc = abs(np.dot(rd, n_vec))

            # Projected area: dA = r^2 * dtheta_h * dtheta_v  (Eq.14)
            dth = H_FOV / max(N_BEAMS - 1, 1)
            dphi = V_FOV / max(N_RAYS  - 1, 1)
            dA = r**2 * dth * dphi

            # Transmission loss (one-way spherical spreading, no absorption)
            TL = 1.0 / r

            # Eq.14 deterministic amplitude
            mu = reflectivity[b, ray]
            A_det = np.sqrt(mu) * cos_inc * np.sqrt(dA) * TL

            if add_speckle:
                # CN(0,1) speckle — each ray is an unresolved scatterer
                xi = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                amp = A_det * xi
            else:
                amp = A_det  # deterministic: remove speckle for clean peak

            # Eq.8: phase sweep across frequency bins
            k    = 2 * np.pi * freqs / SOUND_SPEED   # wavenumber array
            phi  = 2 * r * k                          # two-way phase
            spectrum[b] += amp * np.exp(1j * phi)

    # IFFT → time domain (range domain)
    # p_j(t) = IFFT{ P_j(f) }
    # Note: numpy.fft.ifft applies 1/N normalisation; multiply back for amplitude
    p_time = np.fft.ifft(spectrum, axis=1) * N_FREQ

    # Intensity: I_j(t) = |p_j(t)|^2
    intensity = np.abs(p_time) ** 2
    return intensity


# =============================================================================
# Range axis helper
# =============================================================================
def range_axis():
    """Range in metres for each FFT output bin."""
    return np.arange(N_FREQ) * SOUND_SPEED / (2.0 * BANDWIDTH)


def expected_bin(range_m: float) -> int:
    """FFT bin index corresponding to a given range."""
    return int(range_m * 2 * BANDWIDTH / SOUND_SPEED) % N_FREQ


# =============================================================================
# Main
# =============================================================================
def main():
    print("=== numpy reference (Choi et al. 2021) ===\n")

    # Run reference (no speckle — clean peak for validation)
    print("Running numpy reference (deterministic, no speckle)...")
    I_ref = run_reference(add_speckle=False)

    print("Running numpy reference (with speckle)...")
    I_speckle = run_reference(add_speckle=True)

    r_axis = range_axis()
    range_res = SOUND_SPEED / (2 * BANDWIDTH)
    exp_bin   = expected_bin(R_FLOOR)

    print(f"\nRange resolution: {range_res:.6f} m/bin  (c=1500, B={BANDWIDTH/1e6:.1f}MHz)")
    print(f"Floor at {R_FLOOR} m  → expected bin: {exp_bin}")
    print(f"Observed peak bin (no speckle): {np.argmax(I_ref.sum(axis=0))}")
    print(f"Observed peak bin (speckle):    {np.argmax(I_speckle.sum(axis=0))}")

    # ---- Load GPU output if available ----
    gpu_file = "output/sonar_intensity.bin"
    I_gpu = None
    if os.path.exists(gpu_file):
        print(f"\nLoading GPU output from {gpu_file}...")
        I_gpu = np.fromfile(gpu_file, dtype=np.float32).reshape(N_BEAMS, N_FREQ)
        gpu_peak = np.argmax(I_gpu.sum(axis=0))
        print(f"GPU peak bin: {gpu_peak}  (expected: {exp_bin})")
        diff = abs(gpu_peak - exp_bin)
        if diff <= 1:
            print(f"✓ PASS: GPU peak within 1 bin of expected (diff={diff})")
        else:
            print(f"✗ FAIL: GPU peak {diff} bins off — check frequency grid convention")
    else:
        print(f"\n[info] GPU output not found at {gpu_file}")
        print("       Run: cargo run --release -- --export-bin output/sonar_intensity.bin")
        print("       (Showing numpy reference only)\n")

    # ---- Plot ----
    ncols = 3 if I_gpu is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    # Plot 1 — reference no speckle
    ax = axes[0]
    im = ax.imshow(I_ref.T, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
    ax.set_title('Reference (numpy, no speckle)\nEq.14 + Eq.8 + IFFT')
    ax.set_xlabel('Beam index'); ax.set_ylabel('Range [m]')
    ax.axhline(R_FLOOR, color='cyan', lw=1.5, ls='--', label=f'floor r={R_FLOOR}m')
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot 2 — reference with speckle
    ax = axes[1]
    im = ax.imshow(I_speckle.T, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
    ax.set_title('Reference (numpy, with CN(0,1) speckle)\nCoherent Rayleigh fading')
    ax.set_xlabel('Beam index')
    ax.axhline(R_FLOOR, color='cyan', lw=1.5, ls='--', label=f'floor r={R_FLOOR}m')
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='Intensity')

    # Plot 3 — GPU output (if available)
    if I_gpu is not None:
        ax = axes[2]
        im = ax.imshow(I_gpu.T, aspect='auto', cmap='hot', origin='lower',
                       extent=[0, N_BEAMS, r_axis[0], r_axis[-1]])
        ax.set_title('GPU output (wgpu / WGSL)\nPhilox4x32-10 speckle')
        ax.set_xlabel('Beam index')
        ax.axhline(R_FLOOR, color='cyan', lw=1.5, ls='--', label=f'floor r={R_FLOOR}m')
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label='Intensity')

        # Numerical comparison
        # Speckle makes exact match impossible — compare beam-summed intensity profile
        ref_profile = I_ref.sum(axis=0)
        gpu_profile = I_gpu.sum(axis=0)
        # Normalise both
        ref_n = ref_profile / (ref_profile.max() + 1e-12)
        gpu_n = gpu_profile / (gpu_profile.max() + 1e-12)
        max_err = np.max(np.abs(ref_n - gpu_n))
        print(f"\nMax normalised error (beam-summed profile): {max_err:.4f}")
        print("(Speckle causes per-pixel differences; profile comparison checks physics)")

    plt.suptitle(
        f'Choi et al. (2021)  |  {N_BEAMS} beams × {N_RAYS} rays × {N_FREQ} freq bins\n'
        f'c={SOUND_SPEED} m/s  B={BANDWIDTH/1e6:.1f} MHz  range_res={range_res*1e6:.1f} μm/bin',
        fontsize=9
    )
    plt.tight_layout()

    os.makedirs("validate", exist_ok=True)
    out_path = "validate/comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()