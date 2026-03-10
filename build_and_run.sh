#!/usr/bin/env bash
set -euo pipefail

echo "==== sonar_bridge_poc build ===="

# dependency checks
command -v cargo >/dev/null || { echo "Rust missing → https://rustup.rs"; exit 1; }
command -v g++   >/dev/null || { echo "g++ missing → sudo apt install g++"; exit 1; }

# build rust
echo "[1] Building Rust library"
cargo build --release

# run rust benchmark
echo "[2] Running Rust benchmark"
cargo run --release

# compile C++ host
echo "[3] Building C++ host"
g++ -std=c++17 -O2 \
    -o sonar_host \
    cpp_host/main.cpp \
    -L./target/release \
    -lsonar_engine \
    -Wl,-rpath,./target/release \
    -ldl -lpthread -lm

# run host
echo "[4] Running C++ host"
./sonar_host

# compile gazebo stub
echo "[5] Building Gazebo plugin stub"
g++ -std=c++17 -O2 \
    -o gazebo_plugin_test \
    gazebo_plugin/SonarPlugin.cpp \
    -L./target/release \
    -lsonar_engine \
    -Wl,-rpath,./target/release \
    -ldl -lpthread -lm

./gazebo_plugin_test

echo ""
echo "✓ ALL STEPS PASSED"
