#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// create engine (call once)
void* sonar_engine_init(void);

// run one sonar update
void sonar_engine_update(void* engine, float* data, size_t len);

// destroy engine
void sonar_engine_destroy(void* engine);

// return active GPU name
const char* sonar_backend_name(void* engine);

#ifdef __cplusplus
}
#endif