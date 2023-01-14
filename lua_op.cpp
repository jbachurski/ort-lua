#include <iostream>
#include <type_traits>
#include <onnxruntime_cxx_api.h>
#include <lua.hpp>
#include "lua_op.hpp"
#define bail(__msg) do { throw std::runtime_error(__msg); } while(0)


struct LuaState
{
  lua_State *L;

  LuaState() : L(luaL_newstate()) {
    luaL_openlibs(L);
  }

  ~LuaState() {
    lua_close(L);
  }

  operator lua_State*() {
    return L;
  }
};


template<typename T> std::vector<T> get_tensor_data(Ort::CustomOpApi&, const OrtValue*);
template<> std::vector<double> get_tensor_data<double>(Ort::CustomOpApi& ort, const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
  const size_t count = ort.GetTensorShapeElementCount(info);
  ort.ReleaseTensorTypeAndShapeInfo(info);
  std::vector<double> target(count);
  auto data = ort.GetTensorData<double>(value);
  copy(data, data + count, target.begin());
  return target;
}

template<typename T> void set_tensor_data(Ort::CustomOpApi&, OrtValue*, const std::vector<T>&);

template<> void set_tensor_data<double>(Ort::CustomOpApi& ort, OrtValue* value, const std::vector<double>& data) {
  copy(data.begin(), data.end(), ort.GetTensorMutableData<double>(value));
}


template<typename T> void push_tensor_element(lua_State*, const T&);

template<> void push_tensor_element<double>(lua_State *L1, const double& value) {
  lua_pushnumber(L1, value);
}

template<typename T>
void push_tensor_table(LuaState& L, const std::vector<int64_t>& shape, const std::vector<T>& data) {
    size_t size = 1;
    for(auto d : shape) size *= d;
    lua_createtable(L, 2, 0);

    // Store the shape as an array table -  {1: shape[0], 2: shape[1], ...}
    lua_pushstring(L, "shape");
    lua_createtable(L, shape.size(), 0);
    for(size_t i = 0; i < shape.size(); i++) {
      lua_pushinteger(L, i + 1);
      lua_pushinteger(L, shape[i]);
      lua_settable(L, -3);
    }
    lua_settable(L, -3);

    // Set up the element getter as a closure with access to data, size and shape
    lua_pushstring(L, "get");
    lua_pushlightuserdata(L, (void*)&data);
    lua_pushlightuserdata(L, (void*)&shape);
    lua_pushcclosure(L, [](lua_State *L1) {
      const std::vector<T>& data = *reinterpret_cast<const std::vector<T>*>(lua_touserdata(L1, lua_upvalueindex(1)));
      const std::vector<int64_t>& shape = *reinterpret_cast<const std::vector<int64_t>*>(lua_touserdata(L1, lua_upvalueindex(2)));
      if (lua_gettop(L1) != shape.size()) {
        return luaL_error(L1, "Tensor get: expected number of arguments to be equal to rank.");
      }
      // Get the row-major index
      size_t index = 0;
      for(size_t i = 0; i < shape.size(); i++) {
        index *= shape[i];
        index += luaL_checkinteger(L1, 1 + i);
      }
      if(index >= data.size()) {
        return luaL_error(L1, "Tensor get: index out of bounds.");
      }
      push_tensor_element<T>(L1, data[index]);
      return 1; // Number of results
    }, 2);
    lua_settable(L, -3);
}


template<typename T> T pop_tensor_element(lua_State*, int);

template<> double pop_tensor_element<double>(lua_State *L1, int i) {
  return lua_tonumber(L1, i);
}

std::vector<int64_t> top_tensor_table_shape(LuaState& L) {
  // Access the shape table
  if(lua_pushstring(L, "shape"); lua_gettable(L, -2) != LUA_TTABLE) {
    bail("The returned tensor-table does not have an (array) table 'shape' field.");
  }
  // Access the table by iterating, expecting {1: shape[0], 2: shape[1], ..., shape.size(): shape.back()}
  std::vector<int64_t> shape;
  lua_pushnil(L);
  while(lua_next(L, -2)) {
    lua_pushvalue(L, -2);
    if(lua_tointeger(L, -1) != 1 + shape.size()) {
      bail("The returned shape is not an array table.");
    }
    shape.push_back(lua_tointeger(L, -2));
    lua_pop(L, 2);
  }
  lua_pop(L, 1);
  return shape;
}

template<typename T>
std::vector<T> pop_tensor_table(LuaState& L, const std::vector<int64_t>& shape) {
    // Access the element function and put it on the top of the stack
    if(lua_pushstring(L, "get"); lua_gettable(L, -2) != LUA_TFUNCTION) {
      bail("The returned tensor-table does not have a function 'get' field.");
    }

    size_t length = 1;
    for(auto dim : shape)
      length *= dim > 0 ? dim : 0;
    const size_t rank = shape.size();

    std::vector<T> data(length);
    std::vector<size_t> index(rank);
    for(size_t i = 0; i < length; i++) {
      for(size_t d = rank, j = i; d --> 0; ) {
        index[d] = j % shape[d];
        j /= shape[d];
      }
      // The stack has the getter, copy it so we retain it after pcall
      lua_pushvalue(L, -1);
      for(size_t d = 0; d < rank; d++)
        lua_pushinteger(L, index[d]);
      // The stack is the getter twice and then the 'rank' x dimension indices
      if(lua_pcall(L, rank, 1, 0) != LUA_OK) {
        std::string message(lua_tostring(L, -1));
        bail(message);
      }
      // Stack is now just the getter and resulting element at 'index', take and pop it
      data[i] = pop_tensor_element<T>(L, -1);
      lua_pop(L, 1);
    }
    // Pop the getter from the stack
    lua_pop(L, 1);

    return data;
}

void LuaKernel::Compute(OrtKernelContext* context) {
  static_assert(std::is_same<double, lua_Number>::value, "Assumes that lua_Number is a double-precision float.");

  LuaState L;

  // Define the Lua compute function and check for possible errors or lack of return.
  if(luaL_dostring(L, code_.data()) != LUA_OK) {
    std::string message(lua_tostring(L, -1));
    bail(message);
  }
  if(lua_gettop(L) != 1) {
    bail("Lua code must return a single function, but " + std::to_string(lua_gettop(L)) +  " values were returned.");
  }
  if(!lua_isfunction(L, -1)) {
    bail("Lua code must return a function to be called with operator inputs, but a different type was found.");
  }
  // Stack: just compute function

  lua_checkstack(L, 2 * OP_IO_SLOTS);
  // Parse input data and place on Lua stack as tables
  std::vector<double> input_data[OP_IO_SLOTS];
  std::vector<int64_t> shapes[OP_IO_SLOTS];
  for(size_t k = 0; k < OP_IO_SLOTS; k++) {
    // Push all the input tensor tables in sequential order (if the tensors exist)
    const OrtValue* value = ort_.KernelContext_GetInput(context, k);
    if(value == nullptr) {
      lua_pushnil(L);
      continue;
    }
    auto& shape = shapes[k];
    OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(value);
    shape = ort_.GetTensorShape(info);
    ort_.ReleaseTensorTypeAndShapeInfo(info);
    push_tensor_table(L, shape, input_data[k] = get_tensor_data<double>(ort_, value));
  }
  // Stack: compute function, then OP_IO_SLOTS x tensor tables (or nil for missing inputs)

  // Execute function with arguments on the stack same as in operator input order, check error
  if(lua_pcall(L, OP_IO_SLOTS, OP_IO_SLOTS, 0) != LUA_OK) {
    std::string message(lua_tostring(L, -1));
    bail(message);
  }
  // Stack: OP_IO_SLOTS x tensor tables (or nil for missing outputs)

  for(size_t k = OP_IO_SLOTS; k --> 0; lua_pop(L, 1)) {
    // Pop output tensors in reverse order (if they exist)
    if(lua_isnil(L, -1)) {
      std::vector<int64_t> shape = {};
      if(ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size())) {
        bail("Output " + std::to_string(k) + " was nil, but it has a slot and would be missing.");
      }
      continue;
    }
    if(!lua_istable(L, -1)) {
      bail("The returned value must be a (tensor) table.");
    }

    auto shape = top_tensor_table_shape(L);
    OrtValue* value = ort_.KernelContext_GetOutput(context, k, shape.data(), shape.size());
    if(!value) {
      bail("Output " + std::to_string(k) + " was not nil, but it has no slot and would be implicitly ignored.");
    }
    auto data = pop_tensor_table<double>(L, shape);
    set_tensor_data(ort_, value, data);
  }
}

#undef bail
