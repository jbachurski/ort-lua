#pragma once
#include <onnxruntime_cxx_api.h>

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);