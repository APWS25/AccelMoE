#include "layer.h"


/* Conv1D 
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *         = (s - K + 2 * 0) / 1 + 1
 *         = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  
  size_t os = s - K + 1;

  for (size_t i = 0; i < OC; i++) {
    for (size_t j = 0; j < os; j++) {
      float val = 0.f;
      for (size_t k = 0; k < C; k++) {
        for (size_t l = 0; l < K; l++) {
          val += in->buf[k * s + j + l] * 
                  w->buf[i * C * K + k * K + l];
        }
      }
      out->buf[i * os + j] = val + b->buf[i];
    }
  }
}

__global__ void Conv1D_Kernel(float *in, float *w, float *b, float *out, 
  size_t C, size_t s, size_t OC, size_t K) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t os = s - K + 1;

  if (i < OC * os) {
    size_t oc = i / os;
    size_t j = i % os;
    float val = 0.f;

    for (size_t k = 0; k < C; k++) {
      for (size_t l = 0; l < K; l++) {
        val += in[k * s + j + l] * w[oc * C * K + k * K + l];
      }
    }
    out[oc * os + j] = val + b[oc];
  }
}

//MARK: Conv1D_CUDA
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim = 256;
  dim3 gridDim = ((OC * os) + 256 - 1) / 256;
  Conv1D_Kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, b->gbuf, out->gbuf, C, s, OC, K);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = inout->buf[i] > 0 ? inout->buf[i] : 0;
  }
}

__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}

//MARK: ReLU_CUDA
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();
  dim3 blockDim = 256;
  dim3 gridDim = (N + 255) / 256;
  ReLU_Kernel<<<gridDim, blockDim>>>(inout->gbuf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  for (size_t i = 0; i < C; i++) {
    out->buf[i] = in->buf[i * s];
    for (size_t j = 1; j < s; j++) {
      out->buf[i] = in->buf[i * s + j] > out->buf[i] ? 
        in->buf[i * s + j] : out->buf[i];
    }
  }
}

__global__ void GetMax_Kernel(float *in, float *out, size_t C, size_t s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C) {
    float max_val = in[i * s];
    for (size_t j = 1; j < s; j++) {
      max_val = fmaxf(max_val, in[i * s + j]);
    }
    out[i] = max_val;
  }
}

//MARK: GetMax_CUDA
void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  dim3 blockDim = 256;
  dim3 gridDim = (C + 256 - 1) / 256;
  GetMax_Kernel<<<gridDim, blockDim>>>(in->gbuf, out->gbuf, C, s);
  CHECK_CUDA(cudaDeviceSynchronize());
}


/* Concat
 * @param [in1] in1: [N1]
 * @param [in2] in2: [N2]
 * @param [in3] in3: [N3]
 * @param [in4] in4: [N4]
 * @param [out] out: [N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  for (size_t i = 0; i < N1; i++) {
    out->buf[i] = in1->buf[i];
  }
  for (size_t i = 0; i < N2; i++) {
    out->buf[N1 + i] = in2->buf[i];
  }
  for (size_t i = 0; i < N3; i++) {
    out->buf[N1 + N2 + i] = in3->buf[i];
  }
  for (size_t i = 0; i < N4; i++) {
    out->buf[N1 + N2 + N3 + i] = in4->buf[i];
  }
}

__global__ void Concat_Kernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t N1, size_t N2, size_t N3, size_t N4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N1) out[i] = in1[i];
  else if (i < N1 + N2) out[i] = in2[i - N1];
  else if (i < N1 + N2 + N3) out[i] = in3[i - (N1 + N2)];
  else if (i < N1 + N2 + N3 + N4) out[i] = in4[i - (N1 + N2 + N3)];
}

//MARK: Concat_CUDA
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *out) {
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  dim3 blockDim = 256;
  dim3 gridDim = (N1 + N2 + N3 + N4 + 256 - 1) / 256;
  Concat_Kernel<<<gridDim, blockDim>>>(in1->gbuf, in2->gbuf, in3->gbuf, in4->gbuf, out->gbuf, N1, N2, N3, N4);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  for (size_t i = 0; i < M; i++) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in->buf[j] * w->buf[i * N + j];
    }
    out->buf[i] = val + b->buf[i];
  }
}

__global__ void Linear_Kernel(float *in, float *w, float *b, float *out, size_t N, size_t M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in[j] * w[i * N + j];
    }
    out[i] = val + b[i];
  }
}

//MARK: Linear_CUDA
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  dim3 blockDim = 256;
  dim3 gridDim = (M + 256 - 1) / 256;
  Linear_Kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, b->gbuf, out->gbuf, N, M);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void Softmax(Tensor *inout) {
  size_t N = inout->shape[0];

  float max_val = -INFINITY;
  for (size_t i = 0; i < N; i++) {
    max_val = inout->buf[i] > max_val ? inout->buf[i] : max_val;
  }

  float sum = 0.f;
  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = exp(inout->buf[i] - max_val);
    sum += inout->buf[i];
  }

  for (size_t i = 0; i < N; i++) { inout->buf[i] /= sum; }
}

// __global__ void Softmax_Kernel(float *buf, size_t N) {
//   __shared__ float max_val;
//   __shared__ float sum;

//   int tid = threadIdx.x;
//   if (tid == 0) {
//       max_val = -INFINITY;
//       for (size_t i = 0; i < N; i++) {
//           max_val = fmaxf(max_val, buf[i]);
//       }
//   }
//   __syncthreads();
//   if (tid == 0) {
//       sum = 0.0f;
//       for (size_t i = 0; i < N; i++) {
//           buf[i] = expf(buf[i] - max_val);
//           sum += buf[i];
//       }
//   }
//   __syncthreads();

//   // Step 3: Normalize
//   if (tid == 0) {
//       for (size_t i = 0; i < N; i++) {
//           buf[i] /= sum;
//       }
//   }
// }

__global__ void Softmax_Kernel(float *gbuf, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  // 1. Find max value (to prevent overflow)
  float max_val = -INFINITY;
  for (size_t i = 0; i < N; i++) {
    max_val = fmaxf(max_val, gbuf[i]);
  }

  // 2. Compute exponentials and sum
  float sum = 0.0f;
  for (size_t i = 0; i < N; i++) {
    gbuf[i] = expf(gbuf[i] - max_val);
    sum += gbuf[i];
  }

  // 3. Normalize
  for (size_t i = 0; i < N; i++) {
    gbuf[i] /= sum;
  }
}

//MARK: Softmax_CUDA
void Softmax_CUDA(Tensor *inout) {
  size_t N = inout->shape[0];

  dim3 blockDim = 256;
  dim3 gridDim = (N + 256 - 1) / 256;
  Softmax_Kernel<<<gridDim, blockDim>>>(inout->gbuf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* (Elemwise) Scaling
 * @param [in & out] inout: [N]
 * @param [in]           s: [1]
 * 'N' is the number of elements in the tensor.
 */
void Scaling(Tensor *inout, float s) {
  size_t N = inout->shape[0];

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] *= s;
  }
}

__global__ void Scaling_Kernel(float *gbuf, size_t N, float s) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
      gbuf[idx] *= s;
  }
}

//MARK: Scaling_CUDA
void Scaling_CUDA(Tensor *inout, float s) {
  size_t N = inout->shape[0];
  
  dim3 blockDim = 256;
  dim3 gridDim = (N + 256 - 1) / 256;
  Scaling_Kernel<<<gridDim, blockDim>>>(inout->gbuf, N, s);
  cudaDeviceSynchronize();
}

/* (Elemwise) Addition
 * @param [in1] in1: [N]
 * @param [in2] in2: [N]
 * @param [in3] in3: [N]
 * @param [in4] in4: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the input tensor.
 */
void Add(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out) {
  size_t N = in1->shape[0];

  for (size_t i = 0; i < N; i++) {
    out->buf[i] = in1->buf[i] + in2->buf[i] + in3->buf[i] + in4->buf[i];
  }
}

__global__ void Add_Kernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
      out[idx] = in1[idx] + in2[idx] + in3[idx] + in4[idx];
  }
}


//MARK: Add_CUDA
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *out) {
  size_t N = in1->shape[0]; // (2048, 1, 1, 1)

  dim3 blockDim = 256;
  dim3 gridDim = (N + 256 - 1) / 256;
  Add_Kernel<<<gridDim, blockDim>>>(in1->gbuf, in2->gbuf, in3->gbuf, in4->gbuf, out->gbuf, N);
  cudaDeviceSynchronize();
}