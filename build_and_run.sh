#!/usr/bin/env bash
# build_and_run.sh
# builds the rust library and runs both demos

set -e

echo "building rust project..."
cargo build --release

echo ""
echo "running rust benchmark..."
cargo run --release

echo ""
echo "compiling c++ host..."
g++ -std=c++17 -O2 \
    cpp_host/main.cpp \
    -L./target/release \
    -lsonar_engine \
    -Wl,-rpath,./target/release \
    -ldl -lpthread -lm \
    -o gazebo_plugin_test

echo ""
echo "running c++ lifecycle test..."
./gazebo_plugin_test