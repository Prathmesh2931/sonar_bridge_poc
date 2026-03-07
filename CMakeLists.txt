cmake_minimum_required(VERSION 3.20)
project(sonar_bridge_poc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 1. Build the Rust library via Cargo ──────────────────────────────────────
# Detect build type → choose Cargo profile
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CARGO_BUILD_FLAG "--release")
    set(CARGO_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
else()
    set(CARGO_BUILD_FLAG "")
    set(CARGO_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/debug")
endif()

# Custom target that runs `cargo build` as part of the CMake build
add_custom_target(rust_sonar_engine ALL
    COMMAND cargo build ${CARGO_BUILD_FLAG}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Rust sonar_engine library via Cargo ..."
    VERBATIM
)

# ── 2. Import the compiled Rust library as a CMake imported target ────────────
add_library(sonar_engine STATIC IMPORTED GLOBAL)
set_target_properties(sonar_engine PROPERTIES
    IMPORTED_LOCATION "${CARGO_TARGET_DIR}/libsonar_engine.a"
)
add_dependencies(sonar_engine rust_sonar_engine)

# ── 3. Build the C++ host executable ─────────────────────────────────────────
add_executable(sonar_host cpp_host/main.cpp)

target_link_libraries(sonar_host
    PRIVATE
        sonar_engine        # Our Rust FFI library
        dl                  # dlopen — required by wgpu/Vulkan loader
        pthread             # Rust std threads
        m                   # libm (math)
)

# wgpu pulls in Vulkan at runtime (via dlopen), so no compile-time -lvulkan needed.
# On macOS, add: "-framework Metal -framework CoreFoundation -framework QuartzCore"
if(APPLE)
    target_link_libraries(sonar_host PRIVATE
        "-framework Metal"
        "-framework CoreFoundation"
        "-framework QuartzCore"
    )
endif()

# ── 4. ROS 2 / Gazebo integration hints ──────────────────────────────────────
# When integrating into a ROS 2 package, replace the add_executable block with:
#
#   find_package(ament_cmake REQUIRED)
#   find_package(rclcpp REQUIRED)
#
#   add_library(gz_sonar_plugin SHARED
#       gz_plugin/SonarPlugin.cpp
#   )
#   target_link_libraries(gz_sonar_plugin
#       PRIVATE sonar_engine dl pthread
#       PUBLIC  rclcpp::rclcpp
#   )
#   ament_target_dependencies(gz_sonar_plugin rclcpp)
#   ament_export_libraries(gz_sonar_plugin)
#   ament_package()

# ── 5. Optional: install target for colcon build ─────────────────────────────
install(TARGETS sonar_host DESTINATION lib/${PROJECT_NAME})

message(STATUS "sonar_bridge_poc configured. Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Rust lib expected at: ${CARGO_TARGET_DIR}/libsonar_engine.a")
