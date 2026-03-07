// cpp_host/main.cpp
// Simulates a Gazebo plugin calling into the Rust wgpu compute engine via FFI.
//
// Compile after building the Rust library:
//   cargo build --release
//   g++ -std=c++17 -o sonar_host cpp_host/main.cpp \
//       -L./target/release -lsonar_engine \
//       -Wl,-rpath,./target/release -ldl -lpthread
//
// Then run:  ./sonar_host

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>

// ── FFI declarations matching Rust's #[no_mangle] exports ────────────────────
extern "C" {
    /**
     * process_sonar_data — GPU-accelerated acoustic ray processor.
     *
     * @param data  Pointer to mutable array of ray distances (metres).
     * @param len   Number of elements in the array.
     *
     * The function modifies the array IN-PLACE, adding simulated underwater
     * backscatter noise via a wgpu compute shader.
     */
    void        process_sonar_data(float* data, std::size_t len);

    /** Returns the GPU adapter name string (e.g. "NVIDIA GeForce RTX 3060"). */
    const char* sonar_backend_name();
}
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n"
              << "║  sonar_bridge_poc — C++ → Rust FFI smoke test   ║\n"
              << "╚══════════════════════════════════════════════════╝\n\n";

    // Query which GPU wgpu selected (proves vendor-agnostic backend detection)
    const char* gpu_name = sonar_backend_name();
    std::cout << "[wgpu]  Active GPU adapter : " << gpu_name << "\n\n";

    // ── Simulate 100 sonar rays from a Gazebo SDF sensor ─────────────────────
    constexpr std::size_t RAY_COUNT = 100;
    std::vector<float> rays(RAY_COUNT);
    for (std::size_t i = 0; i < RAY_COUNT; ++i) {
        rays[i] = static_cast<float>(i + 1); // 1.0 m … 100.0 m
    }

    // ── Print a few samples BEFORE processing ────────────────────────────────
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[C++ → Rust]  Calling process_sonar_data() with "
              << RAY_COUNT << " rays ...\n\n";

    std::cout << "  BEFORE  rays[ 0] = " << rays[0]
              << "   rays[49] = " << rays[49]
              << "   rays[99] = " << rays[99] << "\n";

    float before[3] = { rays[0], rays[49], rays[99] };

    // ── THE FFI CALL — crosses the C++/Rust boundary ─────────────────────────
    process_sonar_data(rays.data(), rays.size());
    // ─────────────────────────────────────────────────────────────────────────

    std::cout << "  AFTER   rays[ 0] = " << rays[0]
              << "   rays[49] = " << rays[49]
              << "   rays[99] = " << rays[99] << "\n\n";

    // ── Verify noise was actually applied ────────────────────────────────────
    bool ok = (rays[0] != before[0]) && (rays[49] != before[1]) && (rays[99] != before[2]);

    if (ok) {
        std::cout << "✔  Backscatter noise confirmed (values changed after GPU pass)\n";
        std::cout << "✔  C++/Rust FFI bridge is working correctly\n";
        std::cout << "✔  wgpu vendor-agnostic compute: SUCCESS\n\n";
        return 0;
    } else {
        std::cerr << "✘  ERROR: ray values unchanged — compute shader may have failed\n\n";
        return 1;
    }
}
