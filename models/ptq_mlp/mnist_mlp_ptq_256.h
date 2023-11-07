
#include <stdint.h>
#include "tensor1.h"
#ifndef MNIST_MLP_PTQ_256_H_
#define MNIST_MLP_PTQ_256_H_
extern const char *fc1_weight_dtype; 
extern const char *fc1_bias_dtype; 
extern const char *fc1_weight_scale_dtype; 
extern const char *fc1_weight_zero_point_dtype; 
extern const char *fc2_weight_dtype; 
extern const char *fc2_bias_dtype; 
extern const char *fc2_weight_scale_dtype; 
extern const char *fc2_weight_zero_point_dtype; 
extern const char fc1_weight_dim;
extern const char fc1_bias_dim;
extern const char fc1_weight_scale_dim;
extern const char fc1_weight_zero_point_dim;
extern const char fc2_weight_dim;
extern const char fc2_bias_dim;
extern const char fc2_weight_scale_dim;
extern const char fc2_weight_zero_point_dim;
extern const uint32_t fc1_weight_shape[] ;
extern const uint32_t fc1_bias_shape[] ;
extern const uint32_t fc1_weight_scale_shape[] ;
extern const uint32_t fc1_weight_zero_point_shape[] ;
extern const uint32_t fc2_weight_shape[] ;
extern const uint32_t fc2_bias_shape[] ;
extern const uint32_t fc2_weight_scale_shape[] ;
extern const uint32_t fc2_weight_zero_point_shape[] ;
extern const uint64_t fc1_weight_size ;
extern const uint64_t fc1_bias_size ;
extern const uint64_t fc1_weight_scale_size ;
extern const uint64_t fc1_weight_zero_point_size ;
extern const uint64_t fc2_weight_size ;
extern const uint64_t fc2_bias_size ;
extern const uint64_t fc2_weight_scale_size ;
extern const uint64_t fc2_weight_zero_point_size ;
extern const uint8_t fc1_weight_data[]; 
extern const uint8_t fc1_bias_data[]; 
extern const uint8_t fc1_weight_scale_data[]; 
extern const uint8_t fc1_weight_zero_point_data[]; 
extern const uint8_t fc2_weight_data[]; 
extern const uint8_t fc2_bias_data[]; 
extern const uint8_t fc2_weight_scale_data[]; 
extern const uint8_t fc2_weight_zero_point_data[]; 
#endif  // MNIST_MLP_PTQ_256_H_
    