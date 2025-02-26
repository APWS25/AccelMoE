#include "layer.h"
//#include <nvToolsExt.h>

#define BLOCK_SIZE 32

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

//MARK: ReLU_Kernel
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  inout[i] = inout[i] > 0 ? inout[i] : 0;
}

//MARK: ReLU_CUDA
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();
  dim3 blockDim = 32;
  dim3 gridDim = (N + 32) / 32;
  ReLU_Kernel<<<gridDim, blockDim>>>(inout->gbuf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

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

//MARK: Conv1D_Kernel
__global__ void Conv1D_Kernel(float *in, float *w, float *b, float *out, 
  size_t C, size_t s, size_t OC, size_t K) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t os = s - K + 1;
  
  if (i >= OC * os) return;

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

//MARK: Conv1D_CUDA
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim = 32;
  dim3 gridDim = ((OC * os) + 32 - 1) / 32;
  Conv1D_Kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, b->gbuf, out->gbuf, C, s, OC, K);
  CHECK_CUDA(cudaDeviceSynchronize());
}

//MARK: C_ReLU_Kernel
__global__ void Conv1D_ReLU_Kernel(float *in, float *w, float *b, float *out, 
  size_t C, size_t s, size_t OC, size_t K) {
  
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t os = s - K + 1;

  if (i >= OC * os) return;

  size_t oc = i / os;
  size_t j = i % os;
  float val = 0.f;

  for (size_t k = 0; k < C; k++) {
    for (size_t l = 0; l < K; l++) {
      val += in[k * s + j + l] * w[oc * C * K + k * K + l];
    }
  }
  out[oc * os + j] = fmaxf(val + b[oc], 0.0f);
}

//MARK: C_R_Stream_CUDA
void Conv1D_ReLU_Stream_CUDA(Tensor *in, 
  Tensor *conv0_w, Tensor *conv0_b, Tensor *conv0_a,
  Tensor *conv1_w, Tensor *conv1_b, Tensor *conv1_a,
  Tensor *conv2_w, Tensor *conv2_b, Tensor *conv2_a,
  Tensor *conv3_w, Tensor *conv3_b, Tensor *conv3_a) {
  
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  size_t c0_OC = conv0_w->shape[0];
  size_t c0_K = conv0_w->shape[2];
  size_t c0_os = s - c0_K + 1;

  size_t c1_OC = conv1_w->shape[0];
  size_t c1_K = conv1_w->shape[2];
  size_t c1_os = s - c1_K + 1;

  size_t c2_OC = conv2_w->shape[0];
  size_t c2_K = conv2_w->shape[2];
  size_t c2_os = s - c2_K + 1;

  size_t c3_OC = conv3_w->shape[0];
  size_t c3_K = conv3_w->shape[2];
  size_t c3_os = s - c3_K + 1;

  cudaStream_t s0, s1, s2, s3;
  cudaStreamCreate(&s0);
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaStreamCreate(&s3);

  dim3 blockDim = 32;

  Conv1D_ReLU_Kernel<<<((c0_OC * c0_os) + 32 - 1) / 32, blockDim, 0, s0>>>(in->gbuf, conv0_w->gbuf, conv0_b->gbuf, conv0_a->gbuf, C, s, c0_OC, c0_K);
  Conv1D_ReLU_Kernel<<<((c1_OC * c1_os) + 32 - 1) / 32, blockDim, 0, s1>>>(in->gbuf, conv1_w->gbuf, conv1_b->gbuf, conv1_a->gbuf, C, s, c1_OC, c1_K);
  Conv1D_ReLU_Kernel<<<((c2_OC * c2_os) + 32 - 1) / 32, blockDim, 0, s2>>>(in->gbuf, conv2_w->gbuf, conv2_b->gbuf, conv2_a->gbuf, C, s, c2_OC, c2_K);
  Conv1D_ReLU_Kernel<<<((c3_OC * c3_os) + 32 - 1) / 32, blockDim, 0, s3>>>(in->gbuf, conv3_w->gbuf, conv3_b->gbuf, conv3_a->gbuf, C, s, c3_OC, c3_K);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
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

//MARK: GetMax_Kernel
__global__ void GetMax_Kernel(float *in, float *out, size_t C, size_t s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= C) return;

  float max_val = in[i * s];
  for (size_t j = 1; j < s; j++) {
    max_val = fmaxf(max_val, in[i * s + j]);
  }
  out[i] = max_val;
}

//MARK: GetMax_CUDA
void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  dim3 blockDim = 32;
  dim3 gridDim = (C + 32 - 1) / 32;
  GetMax_Kernel<<<gridDim, blockDim>>>(in->gbuf, out->gbuf, C, s);
  CHECK_CUDA(cudaDeviceSynchronize());
}

//MARK: G_Stream_CUDA
void GetMax_Stream_CUDA(
  Tensor *conv0_a, Tensor *pool0_a, 
  Tensor *conv1_a, Tensor *pool1_a, 
  Tensor *conv2_a, Tensor *pool2_a, 
  Tensor *conv3_a, Tensor *pool3_a){

  size_t c0_C = conv0_a->shape[0];
  size_t c0_s = conv0_a->shape[1];

  size_t c1_C = conv1_a->shape[0];
  size_t c1_s = conv1_a->shape[1];

  size_t c2_C = conv2_a->shape[0];
  size_t c2_s = conv2_a->shape[1];

  size_t c3_C = conv3_a->shape[0];
  size_t c3_s = conv3_a->shape[1];

  cudaStream_t s0, s1, s2, s3;
  cudaStreamCreate(&s0);
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaStreamCreate(&s3);

  dim3 blockDim = 32;

  GetMax_Kernel<<<(c0_C + 32 - 1) / 32, blockDim, 0, s0>>>(conv0_a->gbuf, pool0_a->gbuf, c0_C, c0_s);
  GetMax_Kernel<<<(c1_C + 32 - 1) / 32, blockDim, 1, s1>>>(conv1_a->gbuf, pool1_a->gbuf, c1_C, c1_s);
  GetMax_Kernel<<<(c2_C + 32 - 1) / 32, blockDim, 2, s2>>>(conv2_a->gbuf, pool2_a->gbuf, c2_C, c2_s);
  GetMax_Kernel<<<(c3_C + 32 - 1) / 32, blockDim, 3, s3>>>(conv3_a->gbuf, pool3_a->gbuf, c3_C, c3_s);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
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

//MARK: Concat_Kernel
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

  dim3 blockDim = 32;
  dim3 gridDim = (N1 + N2 + N3 + N4 + 32 - 1) / 32;
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

//MARK: L_Kernel
__global__ void Linear_Kernel(float *in, float *w, float *b, float *out, size_t N, size_t M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;

  float val = 0.f;
  for (size_t j = 0; j < N; j++) {
    val += in[j] * w[i * N + j];
  }
  out[i] = val + b[i];
}

//MARK: L_CUDA
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  dim3 blockDim = 32;
  dim3 gridDim = (M + 32 - 1) / 32;

  // dim3 blockDim(32, 32);
  // dim3 gridDim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  Linear_Kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, b->gbuf, out->gbuf, N, M);
  CHECK_CUDA(cudaDeviceSynchronize());
}

//MARK: L_ReLU_Kernel
__global__ void Linear_ReLU_Kernel(float *in, float *w, float *b, float *out, size_t N, size_t M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;

  float val = 0.f;
  for (size_t j = 0; j < N; j++) {
    val += in[j] * w[i * N + j];
  }
  out[i] = fmaxf(val + b[i], 0.0f);
}

//MARK: L_ReLU_Shared_Kernel
//NOTE: 1 by는 아무런 의미가 없다...
__global__ void Linear_ReLU_Shared_Kernel(float *in, float *we, float *b, float *out, int N, int M) {
  // 입력: in (1xN), we (M×N; NxM의 전치), 출력: out (1xM)
  // 원본 코드의 M, K, N == 여기에서 1, N, M
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int gj = blockIdx.x;
  if (gj * BLOCK_SIZE >= M) return;
  int lj = threadIdx.x;

  // 쓰레드 블록끼리는 공유 가능한 데이터
  __shared__ float inLocal[BLOCK_SIZE];
  __shared__ float weLocal[BLOCK_SIZE][BLOCK_SIZE];

  float val = 0.f;    // 쓰레드마다 개별적으로 저장

  for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
      // step 1: 타일 내의 값 공유 메모리 로드
      int in_col_idx = bk + lj;
      //int we_col_idx = bk + lj;
      inLocal[lj] = (in_col_idx < N) ? in[in_col_idx] : 0.f;
      for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
        weLocal[lj][kk] = we[j * N + bk + kk]; 
      }
      __syncthreads();
      // 각 스레드가 모두 로드 완료

      // step 2: 타일 내의 matmul
      for (int lk = 0; lk < BLOCK_SIZE; ++lk) {
          val += inLocal[lk] * weLocal[lj][lk];
      }
      __syncthreads();
  }
  // 각 스레드가 자신이 담당한 output 원소 하나 계산 완료
  out[j] = fmaxf(val + b[j], 0.0f);
}

//MARK: L_ReLU_CUDA
void Linear_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  dim3 blockDim = 32;
  dim3 gridDim = (M + 32 - 1) / 32;

  // dim3 blockDim(32, 32);
  // dim3 gridDim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  Linear_ReLU_Kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, b->gbuf, out->gbuf, N, M);
  CHECK_CUDA(cudaDeviceSynchronize());
}

//MARK: L_Stream_CUDA
void Linear_Stream_CUDA(Tensor *in, 
  Tensor *gate_w, Tensor *gate_b, Tensor *gate_a,
  Tensor *exp0_w, Tensor *exp0_b, Tensor *expert0_a,
  Tensor *exp1_w, Tensor *exp1_b, Tensor *expert1_a,
  Tensor *exp2_w, Tensor *exp2_b, Tensor *expert2_a,
  Tensor *exp3_w, Tensor *exp3_b, Tensor *expert3_a) {

  cudaStream_t s0, s1, s2, s3, s4;
  cudaStreamCreate(&s0);
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaStreamCreate(&s3);
  cudaStreamCreate(&s4);

  dim3 blockDim = 32;
  dim3 gridDim = (expert0_a->shape[0] + 32 - 1) / 32;
  
  Linear_Kernel<<<gridDim, blockDim, 0, s0>>>(in->gbuf, gate_w->gbuf, gate_b->gbuf, gate_a->gbuf, in->shape[0], gate_w->shape[0]);
  Linear_Kernel<<<gridDim, blockDim, 0, s1>>>(in->gbuf, exp0_w->gbuf, exp0_b->gbuf, expert0_a->gbuf, in->shape[0], exp0_w->shape[0]);
  Linear_Kernel<<<gridDim, blockDim, 0, s2>>>(in->gbuf, exp1_w->gbuf, exp1_b->gbuf, expert1_a->gbuf, in->shape[0], exp1_w->shape[0]);
  Linear_Kernel<<<gridDim, blockDim, 0, s3>>>(in->gbuf, exp2_w->gbuf, exp2_b->gbuf, expert2_a->gbuf, in->shape[0], exp2_w->shape[0]);
  Linear_Kernel<<<gridDim, blockDim, 0, s4>>>(in->gbuf, exp3_w->gbuf, exp3_b->gbuf, expert3_a->gbuf, in->shape[0], exp3_w->shape[0]);

  CHECK_CUDA(cudaDeviceSynchronize());
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
  cudaStreamDestroy(s4);
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


//MARK: Softmax_Kernel
__global__ void Softmax_Kernel(float *gbuf, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float max_val = -INFINITY;
  for (size_t i = 0; i < N; i++) {
    max_val = fmaxf(max_val, gbuf[i]);
  }

  float sum = 0.0f;
  for (size_t i = 0; i < N; i++) {
    gbuf[i] = expf(gbuf[i] - max_val);
    sum += gbuf[i];
  }

  for (size_t i = 0; i < N; i++) {
    gbuf[i] /= sum;
  }
}

//MARK: Softmax_CUDA
void Softmax_CUDA(Tensor *inout) {
  size_t N = inout->shape[0];

  dim3 blockDim = 32;
  dim3 gridDim = (N + 32 - 1) / 32;
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

//MARK: Scaling_Kernel
__global__ void Scaling_Kernel(float *gbuf, size_t N, float s) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N) return;

  gbuf[idx] *= s;
}

//MARK: Scaling_CUDA
void Scaling_CUDA(Tensor *inout, float s) {
  size_t N = inout->shape[0];
  
  dim3 blockDim = 32;
  dim3 gridDim = (N + 32 - 1) / 32;
  Scaling_Kernel<<<gridDim, blockDim>>>(inout->gbuf, N, s);
  cudaDeviceSynchronize();
}

//MARK: S_Stream_CUDA
void Scaling_Stream_CUDA(Tensor *expert0_a, Tensor *expert1_a, Tensor *expert2_a, Tensor *expert3_a, Tensor *gate_a) {
  cudaStream_t s0, s1, s2, s3;
  cudaStreamCreate(&s0);
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  cudaStreamCreate(&s3);

  dim3 blockDim = 32;
  dim3 gridDim = (expert0_a->shape[0] + 32 - 1) / 32;

  Scaling_Kernel<<<gridDim, blockDim, 0, s0>>>(expert0_a->gbuf, expert0_a->shape[0], gate_a->buf[0]);
  Scaling_Kernel<<<gridDim, blockDim, 0, s1>>>(expert1_a->gbuf, expert1_a->shape[0], gate_a->buf[1]);
  Scaling_Kernel<<<gridDim, blockDim, 0, s2>>>(expert2_a->gbuf, expert2_a->shape[0], gate_a->buf[2]);
  Scaling_Kernel<<<gridDim, blockDim, 0, s3>>>(expert3_a->gbuf, expert3_a->shape[0], gate_a->buf[3]);

  CHECK_CUDA(cudaDeviceSynchronize());
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
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

//MARK: Add_Kernel
__global__ void Add_Kernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
      out[idx] = in1[idx] + in2[idx] + in3[idx] + in4[idx];
  }
}

//MARK: Add_CUDA
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *out) {
  size_t N = in1->shape[0]; // (2048, 1, 1, 1)

  dim3 blockDim = 32;
  dim3 gridDim = (N + 32 - 1) / 32;
  Add_Kernel<<<gridDim, blockDim>>>(in1->gbuf, in2->gbuf, in3->gbuf, in4->gbuf, out->gbuf, N);
  cudaDeviceSynchronize();
}