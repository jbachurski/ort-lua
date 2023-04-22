#pragma once
#include <onnxruntime_c_api.h>

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
