# Lua in ONNX Runtime

## Building from source

You can build the shared library with CMake. It downloads and builds Lua from source, and also downloads an ONNX Runtime binary release.

Begin by configuring the project, which also grabs the dependencies:

```
cmake -Bbuild
```

Afterwards you can build:

```
cmake --build build --config Debug --target all
```

You can also use the different configs (like `Release`). The built library is available under `build/liblangops.so`.

If you are building on a different architecture you might have to modify `ONNXRUNTIME_ARCH`, as the library is downloaded only for a set architecture. Refer to the ORT download page. The shared library itself should always build for your architecture.

You might want to use a virtual environment and install requirements (`numpy`, `onnx`, `onnxruntime`) with:

```py
pip install -r requirements.txt
```

Then you should be able to successfully run `python test.py`.
