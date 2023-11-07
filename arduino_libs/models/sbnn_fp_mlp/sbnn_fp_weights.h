
#include <stdint.h>
#include "tensor1.h"

#ifndef SBNN_FP_WEIGHTS_H_
#define SBNN_FP_WEIGHTS_H_


extern const char *fc1_weight_dtype; 
extern const char *fc1_mask_dtype; 
extern const char *fc1_alpha_dtype; 
extern const char *fc2_weight_dtype; 
extern const char *fc2_mask_dtype; 
extern const char *fc2_alpha_dtype; 
extern const char fc1_weight_dim;
extern const char fc1_mask_dim;
extern const char fc1_alpha_dim;
extern const char fc2_weight_dim;
extern const char fc2_mask_dim;
extern const char fc2_alpha_dim;
extern const uint32_t fc1_weight_shape[] ;
extern const uint32_t fc1_mask_shape[] ;
extern const uint32_t fc1_alpha_shape[] ;
extern const uint32_t fc2_weight_shape[] ;
extern const uint32_t fc2_mask_shape[] ;
extern const uint32_t fc2_alpha_shape[] ;
extern const uint64_t fc1_weight_size ;
extern const uint64_t fc1_mask_size ;
extern const uint64_t fc1_alpha_size ;
extern const uint64_t fc2_weight_size ;
extern const uint64_t fc2_mask_size ;
extern const uint64_t fc2_alpha_size ;
extern const uint8_t fc1_weight_data[]; 
extern const uint8_t fc1_mask_data[]; 
extern const uint8_t fc1_alpha_data[]; 
extern const uint8_t fc2_weight_data[]; 
extern const uint8_t fc2_mask_data[]; 
extern const uint8_t fc2_alpha_data[]; 


#endif  // SBNN_FP_WEIGHTS_H_
    