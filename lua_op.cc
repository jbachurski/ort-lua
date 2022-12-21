  #include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <lua.hpp>
#include "lua_op.h"


void LuaKernel::Compute(OrtKernelContext* context) {
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const double* X_data = reinterpret_cast<const double*>(ort_.GetTensorData<double>(input_X));

  auto lua_state = luaL_newstate();
  luaL_openlibs(lua_state);

  lua_setglobal(lua_state, "inputs");
  luaL_dostring(lua_state, "function");
  lua_close(lua_state);

  std::vector<int64_t> shape;
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, shape.data(), shape.size());
  double* out = ort_.GetTensorMutableData<double>(output);



  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

}
