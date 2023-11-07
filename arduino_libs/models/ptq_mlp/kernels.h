#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include "tensor1.h"
#include "layers.h"

TensorFloat1D* fp_linear(struct QInt8Linear* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu);
TensorInt161D* qint8_linear(struct QInt8Linear* layer, TensorInt161D** input, TensorInt161D* output, bool fuse_relu);
TensorFloat1D* qint8_float_linear_qat(struct QInt8Linear* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu);

TensorFloat1D* sb_fp_linear(struct SBLinear_FP* layer,  TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu);
TensorFloat1D* sb_linear(struct SBLinear* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu);

#endif 