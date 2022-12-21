import onnx
import onnx.helper
import onnxruntime


model = onnx.helper.make_model(
    onnx.helper.make_graph(
        [onnx.helper.make_node(
            'Add',
            ['x', 'y'],
            ['z']
        )],
        'graph',
        [
            onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ('N')),
            onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ('N'))
        ],
        [
            onnx.helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, ('N'))
        ]
    )
)

options = onnxruntime.SessionOptions()
options.register_custom_ops_library('build/libluaop.dylib')

session = onnxruntime.InferenceSession(model, options)
