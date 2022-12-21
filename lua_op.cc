#include <type_traits>

#include <onnxruntime_cxx_api.h>
#include <lua.hpp>
#include "lua_op.h"

void LuaKernel::Compute(OrtKernelContext* context) {
  static_assert(std::is_same<double, lua_Number>::value, "Assumes that lua_Number is a double-precision float.");

  // Access ORT input value
  const OrtValue* X_value = ort_.KernelContext_GetInput(context, 0);
  OrtTensorTypeAndShapeInfo* X_info = ort_.GetTensorTypeAndShape(X_value);
  auto X_shape = ort_.GetTensorShape(X_info);
  ort_.ReleaseTensorTypeAndShapeInfo(X_info);
  const double* X_data = reinterpret_cast<const double*>(ort_.GetTensorData<double>(X_value));
  
  size_t X_size = 1;
  for(auto x : X_shape)
    X_size *= x;

  // Construct Lua state
  auto lua_state = luaL_newstate();
  luaL_openlibs(lua_state);

  // Define the Lua compute function and check for possible errors or lack of return.
  if(luaL_dostring(lua_state, code_.data()) != LUA_OK) {
    std::string message(lua_tostring(lua_state, -1));
    throw std::runtime_error(message);
  }
  if(!lua_gettop(lua_state)) {
    throw std::runtime_error("Lua code must return a function, but nothing was returned.");
  }
  if(!lua_isfunction(lua_state, -1)) {
    throw std::runtime_error("Lua code must return a function to be called with operator inputs, but a different type was found.");
  }
  // Bottom of stack is now the compute function

  // Parse input data and place on Lua stack as tables
  // TODO: Non-flattened input.
  // TODO: More than one input.
  // TODO: Non-number input.
  lua_createtable(lua_state, X_size, 0);
  for(size_t i = 0; i < X_size; i++) {
    lua_pushnumber(lua_state, i + 1);
    lua_pushnumber(lua_state, X_data[i]);
    lua_settable(lua_state, -3);
  }

  // Execute function with arguments on the stack same as in operator input order, check error
  if(lua_pcall(lua_state, 1, 1, 0) != LUA_OK) {
    std::string message(lua_tostring(lua_state, -1));
    throw std::runtime_error(message);
  }

  // Access the result as number
  // TODO: More than one output.
  // TODO: Non-scalar output.
  // TODO: Non-number output.
  auto result = lua_tonumber(lua_state, -1);

  // Create ORT output value
  std::vector<int64_t> out_shape;
  OrtValue* out_value = ort_.KernelContext_GetOutput(context, 0, out_shape.data(), out_shape.size());
  double* out = ort_.GetTensorMutableData<double>(out_value);
  *out = result;
}
