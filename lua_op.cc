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
  if(lua_gettop(lua_state) != 1) {
    throw std::runtime_error("Lua code must return a single function, but " + std::to_string(lua_gettop(lua_state)) +  " values were returned.");
  }
  if(!lua_isfunction(lua_state, -1)) {
    throw std::runtime_error("Lua code must return a function to be called with operator inputs, but a different type was found.");
  }
  // Bottom of stack is now the compute function

  lua_checkstack(lua_state, 2 * OP_IO_SLOTS);
  // Parse input data and place on Lua stack as tables
  // TODO: Non-number input.
  std::vector<int64_t> shapes[OP_IO_SLOTS];
  for(size_t k = 0; k < OP_IO_SLOTS; k++) {
    const OrtValue* value = ort_.KernelContext_GetInput(context, k);
    if(value == nullptr) {
      lua_pushnil(lua_state);
      continue;
    }
    auto& shape = shapes[k];
    OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(value);
    shape = ort_.GetTensorShape(info);
    ort_.ReleaseTensorTypeAndShapeInfo(info);
    const double* data = reinterpret_cast<const double*>(ort_.GetTensorData<double>(value));

    size_t size = 1;
    for(auto d : shape) size *= d;

    lua_createtable(lua_state, 4, 0); {
      lua_pushstring(lua_state, "index");
      lua_pushnumber(lua_state, k);
      lua_settable(lua_state, -3);

      lua_pushstring(lua_state, "ptr");
      lua_pushlightuserdata(lua_state, (void*)data);
      lua_settable(lua_state, -3);

      lua_pushstring(lua_state, "shape");
      lua_createtable(lua_state, shape.size(), 0);
      for(size_t i = 0; i < shape.size(); i++) {
        lua_pushnumber(lua_state, i + 1);
        lua_pushnumber(lua_state, shape[i]);
        lua_settable(lua_state, -3);
      }
      lua_settable(lua_state, -3);

      lua_pushstring(lua_state, "get");
      lua_pushlightuserdata(lua_state, (void*)data);
      lua_pushinteger(lua_state, size);
      lua_pushlightuserdata(lua_state, (void*)&shape);
      lua_pushcclosure(lua_state, [](lua_State *L1) {
        double* data = reinterpret_cast<double*>(lua_touserdata(L1, lua_upvalueindex(1)));
        size_t size = lua_tonumber(L1, lua_upvalueindex(2));
        std::vector<int64_t>& shape = *reinterpret_cast<std::vector<int64_t>*>(lua_touserdata(L1, lua_upvalueindex(3)));
        if (lua_gettop(L1) != shape.size()) {
          return luaL_error(L1, "Tensor get: expected number of arguments to be equal to rank.");
        }
        size_t index = 0;
        for(size_t i = 0; i < shape.size(); i++) {
          index *= shape[i];
          index += luaL_checkinteger(L1, 1 + i);
        }
        if(index >= size) {
          return luaL_error(L1, "Tensor get: index out of bounds.");
        }
        lua_pushnumber(L1, data[index]);
        return 1;
      }, 3);
      lua_settable(lua_state, -3);

      lua_pushstring(lua_state, "memget");
      lua_pushlightuserdata(lua_state, (void*)data);
      lua_pushinteger(lua_state, size);
      lua_pushcclosure(lua_state, [](lua_State *L1) {
        double* data = reinterpret_cast<double*>(lua_touserdata(L1, lua_upvalueindex(1)));
        size_t size = lua_tonumber(L1, lua_upvalueindex(2));
        auto index = luaL_checkinteger(L1, 1);
        if(index >= size) {
          return luaL_error(L1, "Tensor get: index out of bounds.");
        }
        lua_pushnumber(L1, data[index]);
        return 1;
      }, 2);
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
