import time
import numpy
import onnx
import onnx.helper
import onnxruntime

print(f'{onnx.__version__ = }')
print(f'{onnxruntime.__version__ = }')

test_code = """
function run(xs, ys)
    local x = 0
    local zs = {};
    for i = 0, xs.shape[1] - 1 do
        x = x + xs.get(i) * (ys.get(0, i) + ys.get(1, i))
        zs[#zs+1] = 2 * xs.get(i)
    end
    return {
        shape={1},
        get=function (_) return -x end
    }, {
        shape={#zs},
        get=function (i) return zs[i+1] end
    }
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
            onnx.helper.make_tensor_value_info('y', onnx.TensorProto.DOUBLE, (2, 'N'))
        ], [
            onnx.helper.make_tensor_value_info('r', onnx.TensorProto.DOUBLE, (1,)),
            onnx.helper.make_tensor_value_info('s', onnx.TensorProto.DOUBLE, (None,))
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
    'y': numpy.array([[-1.0, 0.5, 1.0], [-1.0, 0.5, 1.0]])
}))
