#include <onnxruntime_cxx_api.h>
#include "custom_op.h"

template <typename T>
void GroupNormKernel<T>::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
  const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
  const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));
  const OrtValue* input_scale = ort_.KernelContext_GetInput(context, 2);
  const T* scale_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_scale));
  const OrtValue* input_B = ort_.KernelContext_GetInput(context, 3);
  const T* B_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_B));

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);
  const int64_t N = dimensions[0];
  const int64_t C = dimensions[1] / num_groups[0];  // assume [N C*num_groups H W]  per the spec

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);


}