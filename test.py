import numpy
import onnx
import onnx.helper
import onnxruntime

print(f'{onnx.__version__ = }')
print(f'{onnxruntime.__version__ = }')

test_code = """
function run(xs, ys)
    local x = 0
    for k, v in ipairs(xs) do
        x = x + v * ys[k]
    end
    return x, -x
end
return run
"""

model = onnx.helper.make_model(
    onnx.helper.make_graph([
        onnx.helper.make_node(
            'Lua',
            ['x', 'y'],
            ['r', 's'],
            name='luanode',
            domain='lang.lua',
            code=test_code
        )],
        'graph', [
            onnx.helper.make_tensor_value_info('x', onnx.TensorProto.DOUBLE, ('N')),
            onnx.helper.make_tensor_value_info('y', onnx.TensorProto.DOUBLE, ('N'))
        ], [
            onnx.helper.make_tensor_value_info('r', onnx.TensorProto.DOUBLE, ()),
            onnx.helper.make_tensor_value_info('s', onnx.TensorProto.DOUBLE, ())
        ]
    ),
    opset_imports=[
        onnx.helper.make_operatorsetid('', 17),
        onnx.helper.make_operatorsetid('lang.lua', 1)
    ]
)

options = onnxruntime.SessionOptions()
options.register_custom_ops_library('build/liblangops.so')
session = onnxruntime.InferenceSession(model.SerializeToString(), options)

print(session.run(None, {
    'x': numpy.array([1.0, 2.0, 3.0]),
    'y': numpy.array([-1.0, 0.5, 1.0])
}))
