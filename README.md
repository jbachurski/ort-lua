# Lua in ONNX Runtime

## Usage

This project implements a simple ONNX Runtime custom operator, `lang.lua:Lua`, for running Lua scripts.

Due to the current limitation in ONNX Runtime (though support is now added in [#13946](https://github.com/microsoft/onnxruntime/pull/13946)), inputs/outputs of custom operators may not be variadic. Instead, this operator takes 'any' number of optional inputs/outputs of `tensor(float64)`. In the future, nonhomogenous and truly variadic inputs/outputs could be implemented.

The operator has a single required attribute: `code`. It should be a Lua snippet which returns a single function, for instance:

```lua
function run(xs, ys)
    local x = 0
    for i = 0, xs.shape[1] - 1 do
        x = x + xs.get(i) * (ys.get(0, i) + ys.get(1, i))
    end
    return {
        shape={3},
        get=function (i) return i * x end
    }
end
return run
```

Here, the function `run` is defined and returned. At runtime, the inputs of the operator will be passed as parameters to the function in the form special Lua tables. Similarly, results from this function should follow the same table pattern as well. The number of parameters is the same as the number of operator inputs, and likewise for results and operator outputs.

The passed tables implement 'tensors': they should have a `shape` table (an array of integers indexed from `1` to `#shape`) and an element getter function, `get`, which takes `#shape` parameters. The getter function is indexed from `0` in every axis and the order of parameters is the same as the order of respective dimensions in `shape`.

The example code assumes that `xs` is of shape `N` and `ys` of `N x 2`. It computes the dot product `d` of `xs[:]` and `ys[0,:] + ys[1,:]`, and returns a tensor of shape `3` and elements `(0, d, 2*d)`.

## Why Lua

Because it's easy to embed and should be reasonably efficient.

> On the topic of efficiency, I don't think I picked the best interface. The function calls seem to be the main overhead - both to the C closure for input `get` and later the Lua function to output `get`.

Also, an embedded Python operator ([PyOp](https://github.com/microsoft/onnxruntime-extensions/tree/main/pyop)) already existed and seemed to be really hard to get right. There were plenty of issues with runtime handling and with the specifics of how it was implemented. As such, maybe a Lua approach is easier to get fully right.

I tried to handle *most* errors, but probably missed wrong type accesses C-side and so on (but those fail graciously in Lua).

## Building from source

**Warning:** ONNX Runtime's `CustomOpApi`, used here, is now deprecated since 1.14 (the library was originally developed at 1.13). Some refactoring is necessary to update it.

You can build the shared library with CMake. It downloads and builds Lua from source, and also downloads the necessary ONNX Runtime headers from GitHub at a configurable version (tag).

Begin by configuring the project, which also grabs the dependencies:

```
cmake -Bbuild
```

Afterwards you can build:

```
cmake --build build --config Debug --target all
```

You can also use the different configs (like `Release`). The built library is available under `build/liblangops.so`.

You might want to use a virtual environment and install requirements (`numpy`, `onnx`, `onnxruntime`) with:

```py
pip install -r requirements.txt
```

Then you should be able to successfully run `python test.py`.
