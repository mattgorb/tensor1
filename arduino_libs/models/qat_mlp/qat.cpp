#include "tensor1.h"
#include "util.h"
#include "kernels.h"
#include "layers.h"
#include "qat.h"
#include "mnist_mlp_qat_256.h"

TensorFloat1D* qat_forward(TensorUInt81D** test_img){

    //torch qat adjusts the input with quant.scale and quant.zero_point 
    float scale;
    t1_memcpy(&scale, quant_scale_data, sizeof(float));
    double zero_point;
    t1_memcpy(&zero_point, quant_zero_point_data, sizeof(double));
    TensorFloat1D* image_fp32=uint_to_float32_scale(test_img, scale,zero_point);
    destroy_ptr((void**)test_img);


    TensorFloat1D* fc1_out=linear1(&image_fp32);

    TensorFloat1D* fc2_out=linear2(&fc1_out);
    return fc2_out;
}



TensorFloat1D* linear1(TensorFloat1D** input){
    struct QInt8Linear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_int8_2d(fc1_weight_dtype, fc1_weight_dim, fc1_weight_shape, fc1_weight_data, fc1_weight_size);
    layer.bias=create_tensor_float_1d(fc1_bias_dtype,  fc1_bias_dim, fc1_bias_shape,fc1_bias_data, fc1_bias_size);
    layer.scale_weight = *((double*)fc1_weight_scale_data);
    layer.zero_point_weight= *((double*)fc1_weight_zero_point_data);

    layer.scale = *((float*)fc1_scale_data);
    layer.zero_point= *((double*)fc1_zero_point_data);

    bool fuse_relu=true;
    struct QInt8Linear* layer_tr = &layer;

    TensorFloat1D* output=create_empty_float_tensor_1d(layer.weight.rows);
    
    output = qint8_float_linear_qat(layer_tr, input ,  output, fuse_relu );
    destroy_ptr((void**)input);
        

    return output;
}



TensorFloat1D* linear2(TensorFloat1D** input){
    struct QInt8Linear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_int8_2d(fc2_weight_dtype, fc2_weight_dim, fc2_weight_shape, fc2_weight_data, fc2_weight_size);
    layer.bias=create_tensor_float_1d(fc2_bias_dtype,  fc2_bias_dim, fc2_bias_shape,fc2_bias_data, fc2_bias_size);
    layer.scale_weight = *((double*)fc2_weight_scale_data);
    layer.zero_point_weight= *((double*)fc2_weight_zero_point_data);

    layer.scale = *((float*)fc2_scale_data);
    layer.zero_point= *((double*)fc2_zero_point_data);

    bool fuse_relu=false;
    struct QInt8Linear* layer_tr = &layer;

    TensorFloat1D* output=create_empty_float_tensor_1d(layer.weight.rows);    

    output = qint8_float_linear_qat(layer_tr, input,  output, fuse_relu );
    destroy_ptr((void**)input);
    return output;
}
