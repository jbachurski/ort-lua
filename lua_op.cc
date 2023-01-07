#include <iostream>
#include <type_traits>
#include <onnxruntime_cxx_api.h>
#include <lua.hpp>
#include "lua_op.h"


void LuaKernel::Compute(OrtKernelContext* context) {
  static_assert(std::is_same<double, lua_Number>::value, "Assumes that lua_Number is a double-precision float.");

  // Construct Lua state
  auto L = luaL_newstate();
  luaL_openlibs(L);

  // Define the Lua compute function and check for possible errors or lack of return.
  if(luaL_dostring(L, code_.data()) != LUA_OK) {
    std::string message(lua_tostring(L, -1));
    throw std::runtime_error(message);
  }
  if(lua_gettop(L) != 1) {
    throw std::runtime_error("Lua code must return a single function, but " + std::to_string(lua_gettop(L)) +  " values were returned.");
  }
  if(!lua_isfunction(L, -1)) {
    throw std::runtime_error("Lua code must return a function to be called with operator inputs, but a different type was found.");
  }
  // Bottom of stack is now the compute function

  lua_checkstack(L, 2 * OP_IO_SLOTS);
  // Parse input data and place on Lua stack as tables
  // TODO: Non-number input.
  std::vector<int64_t> shapes[OP_IO_SLOTS];
  for(size_t k = 0; k < OP_IO_SLOTS; k++) {
    const OrtValue* value = ort_.KernelContext_GetInput(context, k);
    if(value == nullptr) {
      lua_pushnil(L);
      continue;
    }
    auto& shape = shapes[k];
    OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(value);
    shape = ort_.GetTensorShape(info);
    ort_.ReleaseTensorTypeAndShapeInfo(info);
    const double* data = reinterpret_cast<const double*>(ort_.GetTensorData<double>(value));

    size_t size = 1;
    for(auto d : shape) size *= d;

    lua_createtable(L, 4, 0); {
      lua_pushstring(L, "index");
      lua_pushnumber(L, k);
      lua_settable(L, -3);

      lua_pushstring(L, "ptr");
      lua_pushlightuserdata(L, (void*)data);
      lua_settable(L, -3);

      lua_pushstring(L, "shape");
      lua_createtable(L, shape.size(), 0);
      for(size_t i = 0; i < shape.size(); i++) {
        lua_pushnumber(L, i + 1);
        lua_pushnumber(L, shape[i]);
        lua_settable(L, -3);
      }
      lua_settable(L, -3);

      lua_pushstring(L, "get");
      lua_pushlightuserdata(L, (void*)data);
      lua_pushinteger(L, size);
      lua_pushlightuserdata(L, (void*)&shape);
      lua_pushcclosure(L, [](lua_State *L1) {
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
      lua_settable(L, -3);

      lua_pushstring(L, "memget");
      lua_pushlightuserdata(L, (void*)data);
      lua_pushinteger(L, size);
      lua_pushcclosure(L, [](lua_State *L1) {
        double* data = reinterpret_cast<double*>(lua_touserdata(L1, lua_upvalueindex(1)));
        size_t size = lua_tonumber(L1, lua_upvalueindex(2));
        auto index = luaL_checkinteger(L1, 1);
        if(index >= size) {
          return luaL_error(L1, "Tensor get: index out of bounds.");
        }
        lua_pushnumber(L1, data[index]);
        return 1;
      }, 2);
      lua_settable(L, -3);
    }
  }

  // Execute function with arguments on the stack same as in operator input order, check error
  if(lua_pcall(L, OP_IO_SLOTS, OP_IO_SLOTS, 0) != LUA_OK) {
    std::string message(lua_tostring(L, -1));
    throw std::runtime_error(message);
  }

  // Access the result as number
  // TODO: Non-scalar output.
  // TODO: Non-number output.
  for(size_t k = OP_IO_SLOTS; k --> 0; lua_pop(L, 1)) {
    if(lua_isnil(L, -1)) {
      std::vector<int64_t> shape = {};
      ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size());
      continue;
    }
    if(!lua_istable(L, -1)) {
      throw std::runtime_error("The returned value must be a (tensor) table.");
    }
    if(lua_pushstring(L, "shape"); lua_gettable(L, -2) != LUA_TTABLE) {
      throw std::runtime_error("The returned tensor-table is missing the 'shape' field.");
    }
    std::vector<int64_t> shape;
    lua_pushnil(L);
    while(lua_next(L, -2)) {
      // lua_tointeger(lua_state, )
      lua_pop(L, 1);
    }
    lua_pop(L, 1);
    OrtValue* value = ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size());
    if(!value) {
      throw std::runtime_error("Output " + std::to_string(k) + " was not nil, but it has no slot and would be implicitly ignored.");
    }
    double* out = ort_.GetTensorMutableData<double>(value);
    *out = 0;
  }
  std::cout << std::endl;
  auto result = lua_tonumber(L, -1);

  // Create ORT output value

}
