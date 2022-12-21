#include <onnxruntime_cxx_api.h>

struct LuaKernel {
private:
  Ort::CustomOpApi ort_;

public:
  LuaKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {}

  void Compute(OrtKernelContext* context);
};


struct LuaOp : Ort::CustomOpBase<LuaOp, LuaKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new LuaKernel(api, info); };
  const char* GetName() const { return "Lua"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };
};
