#pragma once

#include "tensor.h"


/* Layers (Operations) */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Softmax(Tensor *inout);
void Scaling(Tensor *inout, float s);
void Add(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out);

/* Example of using CUDA kernel */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ConvBlock_Stream_CUDA(Tensor *in, 
    Tensor *conv0_w, Tensor *conv0_b, Tensor *conv0_a,
    Tensor *conv1_w, Tensor *conv1_b, Tensor *conv1_a,
    Tensor *conv2_w, Tensor *conv2_b, Tensor *conv2_a,
    Tensor *conv3_w, Tensor *conv3_b, Tensor *conv3_a);
void GetMax_Stream_CUDA(
    Tensor *conv0_a, Tensor *pool0_a, 
    Tensor *conv1_a, Tensor *pool1_a, 
    Tensor *conv2_a, Tensor *pool2_a, 
    Tensor *conv3_a, Tensor *pool3_a);
void Linear_Stream_CUDA(Tensor *in, 
    Tensor *exp0_w, Tensor *exp0_b, Tensor *expert0_a,
    Tensor *exp1_w, Tensor *exp1_b, Tensor *expert1_a,
    Tensor *exp2_w, Tensor *exp2_b, Tensor *expert2_a,
    Tensor *exp3_w, Tensor *exp3_b, Tensor *expert3_a);
void ReLU_CUDA(Tensor *inout);
void GetMax_CUDA(Tensor *in, Tensor *out);
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Softmax_CUDA(Tensor *inout);
void Scaling_CUDA(Tensor *inout, float s);
void Scaling_Stream_CUDA(
    Tensor *expert0_a, 
    Tensor *expert1_a, 
    Tensor *expert2_a, 
    Tensor *expert3_a, 
    Tensor *gate_a);
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out);

/* [Advanced] Example of using half-precision on CPU */
void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out);    