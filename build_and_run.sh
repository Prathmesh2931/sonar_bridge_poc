#!/usr/bin/env bash
# build_and_run.sh — Builds Rust lib + C++ host, then runs both demos
# Usage: bash build_and_run.sh

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        sonar_bridge_poc — build & run script        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Build Rust (release) ─────────────────────────────────────────────
echo "▶ Step 1/4  Building Rust library (cargo build --release) ..."
cargo build --release
echo "  ✔ Rust library built: target/release/libsonar_engine.a"
echo ""

# ── Step 2: Run Rust standalone demo ─────────────────────────────────────────
echo "▶ Step 2/4  Running Rust standalone demo ..."
cargo run --release
echo ""

# ── Step 3: Compile C++ host ──────────────────────────────────────────────────
echo "▶ Step 3/4  Compiling C++ host (g++) ..."
g++ -std=c++17 -O2 \
    -o sonar_host \
    cpp_host/main.cpp \
    -L./target/release \
    -lsonar_engine \
    -Wl,-rpath,./target/release \
    -ldl -lpthread -lm
echo "  ✔ C++ host compiled: ./sonar_host"
echo ""

# ── Step 4: Run C++ host ──────────────────────────────────────────────────────
echo "▶ Step 4/4  Running C++ host (FFI bridge test) ..."
./sonar_host
echo ""

echo "══════════════════════════════════════════════════════"
echo "  ALL CHECKS PASSED — sonar_bridge_poc is working!"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  • Screenshot the GPU name printed above for your proposal"
echo "  • Run on NVIDIA:  WGPU_BACKEND=vulkan ./sonar_host"
echo "  • Run on Intel:   WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME='Intel' ./sonar_host"
echo "  • Record 30s screen capture of this terminal for Arjo's email"
echo ""
