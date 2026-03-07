/**
 * sonar_engine.h — C-compatible header for the Rust wgpu compute engine.
 *
 * Include this in any C or C++ file (e.g. a Gazebo plugin) that needs to
 * call the GPU-accelerated sonar ray processor.
 *
 * Usage:
 *   #include "sonar_engine.h"
 *   process_sonar_data(ray_ptr, ray_count);
 */

#pragma once

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * process_sonar_data
 *
 * Applies simulated underwater acoustic backscatter noise to an array of
 * sonar ray distances using a wgpu GPU compute shader.
 *
 * The operation is IN-PLACE: values in `data` are modified directly.
 * wgpu automatically selects the best available GPU backend
 * (Vulkan, Metal, DX12, or WebGPU) — no vendor lock-in.
 *
 * Thread safety: NOT thread-safe. Call from a single thread per wgpu device.
 *
 * @param data  Non-null pointer to an array of `len` f32 ray distances (metres).
 * @param len   Number of elements. Must be > 0.
 */
void process_sonar_data(float* data, size_t len);

/**
 * sonar_backend_name
 *
 * Returns a null-terminated string identifying the GPU adapter that wgpu
 * selected (e.g. "NVIDIA GeForce RTX 3060", "Intel Iris Xe Graphics").
 *
 * The returned pointer is valid for the lifetime of the process.
 * The caller must NOT free it.
 */
const char* sonar_backend_name(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif
