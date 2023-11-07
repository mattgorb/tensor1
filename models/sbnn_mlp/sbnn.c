#include "sbnn.h"
#include "tensor1.h"
#include "util.h"
#include "kernels.h"
#include "layers.h"
#include "sbnn_256_weights.h"

TensorFloat1D* sbnn_forward(TensorFloat1D** test_img){
    TensorFloat1D* fc1_out=linear1(test_img);
    TensorFloat1D* fc2_out=linear2(&fc1_out);
    return fc2_out;
}

TensorFloat1D* linear1(TensorFloat1D** input){
    struct SBLinear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_uint8_2d(fc1_weight_dtype, fc1_weight_dim, fc1_weight_shape, fc1_weight_data, fc1_weight_size);
    layer.mask = create_tensor_uint8_2d(fc1_mask_dtype, fc1_mask_dim, fc1_mask_shape, fc1_mask_data, fc1_mask_size);
    layer.alpha = *((float*)fc1_alpha_data);

    bool fuse_relu=true;
    struct SBLinear* layer_tr = &layer;

    TensorFloat1D* output=create_empty_float_tensor_1d(layer.weight.rows);
    output = sb_linear(layer_tr, input,  output, fuse_relu );
    destroy_ptr((void**)input);

    return output;
}



TensorFloat1D* linear2(TensorFloat1D** input){
    struct SBLinear layer;

    // Initialize the components (allocate memory or set to appropriate values)
    layer.weight = create_tensor_uint8_2d(fc2_weight_dtype, fc2_weight_dim, fc2_weight_shape, fc2_weight_data, fc2_weight_size);
    layer.mask = create_tensor_uint8_2d(fc2_mask_dtype, fc2_mask_dim, fc2_mask_shape, fc2_mask_data, fc2_mask_size);
    layer.alpha = *((float*)fc2_alpha_data);

    bool fuse_relu=false;
    struct SBLinear* layer_tr = &layer;

    TensorFloat1D* output=create_empty_float_tensor_1d(layer.weight.rows);

    output = sb_linear(layer_tr, input,  output, fuse_relu );
    
    destroy_ptr((void**)input);
    return output;
}
