#include <iostream>
#include <type_traits>
#include <onnxruntime_cxx_api.h>
#include <lua.hpp>
#include "lua_op.h"


void LuaKernel::Compute(OrtKernelContext* context) {
  static_assert(std::is_same<double, lua_Number>::value, "Assumes that lua_Number is a double-precision float.");

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
  // TODO: Non-number input.
  for(size_t k = 0; k < OP_IO_SLOTS; k++) {
    const OrtValue* value = ort_.KernelContext_GetInput(context, 0);
    OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(value);
    auto shape = ort_.GetTensorShape(info);
    ort_.ReleaseTensorTypeAndShapeInfo(info);
    const double* data = reinterpret_cast<const double*>(ort_.GetTensorData<double>(value));

    size_t X_size = 1;
    for(auto x : shape)
      X_size *= x;
    lua_createtable(lua_state, X_size, 0);
    for(size_t i = 0; i < X_size; i++) {
      lua_pushnumber(lua_state, i + 1);
      lua_pushnumber(lua_state, data[i]);
      lua_settable(lua_state, -3);
    }
  }

  // Execute function with arguments on the stack same as in operator input order, check error
  if(lua_pcall(lua_state, OP_IO_SLOTS, OP_IO_SLOTS, 0) != LUA_OK) {
    std::string message(lua_tostring(lua_state, -1));
    throw std::runtime_error(message);
  }

  // Access the result as number
  // TODO: Non-scalar output.
  // TODO: Non-number output.
  for(size_t k = OP_IO_SLOTS; k --> 0; lua_pop(lua_state, 1)) {
    if(lua_isnil(lua_state, -1)) {
      std::vector<int64_t> shape = {};
      ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size());
      continue;
    }
    std::vector<int64_t> shape;
    OrtValue* value = ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size());
    if(!value) {
      throw std::runtime_error("Output " + std::to_string(k) + " was not nil, but it has no slot and would be implicitly ignored.");
    }
    if(!lua_isnumber(lua_state, -1)) {
      throw std::runtime_error("The only currently supported result type is a number, but another type was returned.");
    }
    double* out = ort_.GetTensorMutableData<double>(value);
    *out = lua_tonumber(lua_state, -1);
  }
  std::cout << std::endl;
  auto result = lua_tonumber(lua_state, -1);

  // Create ORT output value

}
