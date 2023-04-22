#pragma once
#include "onnxruntime_api.hpp"

constexpr size_t OP_IO_SLOTS = 16;

struct LuaKernel {
private:
  std::string code_;
  Ort::CustomOpApi ort_;

public:
  LuaKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
    code_ = ort_.KernelInfoGetAttribute<std::string>(info, "code");
  }
  void Compute(OrtKernelContext* context);
};

struct LuaOp : Ort::CustomOpBase<LuaOp, LuaKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new LuaKernel(api, info); };
  const char* GetName() const { return "Lua"; };

  size_t GetInputTypeCount() const { return OP_IO_SLOTS; };
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return INPUT_OUTPUT_OPTIONAL; }

  size_t GetOutputTypeCount() const { return OP_IO_SLOTS; };
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; };
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_OPTIONAL; }
};
