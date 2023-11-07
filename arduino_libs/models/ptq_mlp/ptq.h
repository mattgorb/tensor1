#ifndef QAT_H
#define QAT_H


#include <stdio.h>
#include "tensor1.h"


//SBLinear_FP init_linear(int rows, int cols);
TensorInt161D* ptq_forward(TensorInt161D** input);
TensorInt161D* linear1(TensorInt161D** input);
TensorInt161D* linear2(TensorInt161D** input);
#endif 