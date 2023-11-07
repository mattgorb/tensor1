#include "ptq.h"
#include "tensor1.h"
#include "util.h"
#include "kernels.h"
#include "layers.h"
#include "mnist_mlp_ptq_256.h"

TensorInt161D* ptq_forward(TensorInt161D** test_img){
    TensorInt161D* fc1_out=linear1(test_img);
    TensorInt161D* fc2_out=linear2(&fc1_out);
    return fc2_out;
}

TensorInt161D* linear1(TensorInt161D** input){
    struct QInt8Linear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_int8_2d(fc1_weight_dtype, fc1_weight_dim, fc1_weight_shape, fc1_weight_data, fc1_weight_size);
    layer.bias=create_tensor_float_1d(fc1_bias_dtype,  fc1_bias_dim, fc1_bias_shape,fc1_bias_data, fc1_bias_size);
    layer.scale = *((double*)fc1_weight_scale_data);
    layer.zero_point= *((double*)fc1_weight_zero_point_data);

    bool fuse_relu=true;
    struct QInt8Linear* layer_tr = &layer;

    TensorInt161D* output=create_empty_int16_tensor_1d(layer.weight.rows);

    output = qint8_linear(layer_tr, input,  output, fuse_relu );
    destroy_ptr((void**)input);

    return output;
}



TensorInt161D* linear2(TensorInt161D** input){
    struct QInt8Linear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_int8_2d(fc2_weight_dtype, fc2_weight_dim, fc2_weight_shape, fc2_weight_data, fc2_weight_size);
    layer.bias=create_tensor_float_1d(fc2_bias_dtype,  fc2_bias_dim, fc2_bias_shape,fc2_bias_data, fc2_bias_size);
    layer.scale = *((double*)fc2_weight_scale_data);
    layer.zero_point= *((double*)fc2_weight_zero_point_data);

    bool fuse_relu=true;
    struct QInt8Linear* layer_tr = &layer;

    TensorInt161D* output=create_empty_int16_tensor_1d(layer.weight.rows);

    output = qint8_linear(layer_tr, input,  output, fuse_relu );
    
    destroy_ptr((void**)input);
    return output;
}
