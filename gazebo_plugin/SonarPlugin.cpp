#include "../cpp_host/sonar_engine.h"
#include <iostream>
#include <vector>
#include <chrono>

class SonarPlugin {
public:

    // Called when plugin is loaded
    void Load() {
        std::cout << "[SonarPlugin] init sonar engine\n";

        engine_ = sonar_engine_init();

        std::cout << "GPU: "
                  << sonar_backend_name(engine_) << "\n";

        num_rays_ = 1024;
        ray_buffer_.resize(num_rays_, 0.0f);

        std::cout << "rays: " << num_rays_ << "\n\n";
    }

    // Called each simulation tick
    void Update() {
        auto t0 = std::chrono::high_resolution_clock::now();

        sonar_engine_update(engine_, ray_buffer_.data(), ray_buffer_.size());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "tick " << ms << " ms | "
                  << "r0="   << ray_buffer_[0]   << " "
                  << "r511=" << ray_buffer_[511] << " "
                  << "r1023="<< ray_buffer_[1023] << "\n";
    }

    void Unload() {
        std::cout << "\nfree sonar engine\n";
        sonar_engine_destroy(engine_);
        engine_ = nullptr;
    }

private:
    void* engine_ = nullptr;
    int num_rays_ = 0;
    std::vector<float> ray_buffer_;
};


// simple lifecycle test (no Gazebo)
int main() {
    std::cout << "SonarPlugin lifecycle test\n";

    SonarPlugin plugin;
    plugin.Load();

    for (int i = 0; i < 5; i++) {
        plugin.Update();
    }

    plugin.Unload();

    return 0;
}