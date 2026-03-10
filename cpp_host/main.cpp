#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

extern "C" {
    void* sonar_engine_init();
    void  sonar_engine_update(void* engine, float* data, size_t len);
    void  sonar_engine_destroy(void* engine);
    const char* sonar_backend_name(void* engine);
}

int main() {

    std::cout << "sonar engine test\n";

    // init
    auto t0 = std::chrono::high_resolution_clock::now();
    void* engine = sonar_engine_init();
    auto t1 = std::chrono::high_resolution_clock::now();

    double init_ms =
        std::chrono::duration<double,std::milli>(t1 - t0).count();

    std::cout << "gpu: " << sonar_backend_name(engine) << "\n";
    std::cout << "init: " << std::fixed << std::setprecision(2)
              << init_ms << " ms\n\n";

    const size_t ray_count = 1024;
    std::vector<float> rays(ray_count, 0.0f);

    for (int tick = 0; tick < 5; tick++) {

        auto ta = std::chrono::high_resolution_clock::now();

        sonar_engine_update(engine, rays.data(), rays.size());

        auto tb = std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration<double,std::milli>(tb - ta).count();

        std::cout << "tick " << tick + 1 << " | " << std::setprecision(2) << ms << " ms"
                  << " | r0=" << std::setprecision(4) << rays[0]<< " r512=" << rays[512]<< "\n";
    }

    // shutdown
    sonar_engine_destroy(engine);

    std::cout << "\nengine destroyed\n";

    return 0;
}