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

/* [Conv1D CUDA kernel] */
__global__ void Conv1D_Kernel(float *in, float *w, float *b, float *out, 
  size_t C, size_t s, size_t OC, size_t K) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t os = s - K + 1;

  if (i < OC * os) {
    size_t oc = i / os;  // Output channel index
    size_t j = i % os;   // Output spatial index
    float val = 0.f;

    for (size_t k = 0; k < C; k++) {
      for (size_t l = 0; l < K; l++) {
        val += in[k * s + j + l] * w[oc * C * K + k * K + l];
      }
    }
    out[oc * os + j] = val + b[oc];
  }
}

/* [Conv1D using CUDA] */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  float *d_in, *d_w, *d_b, *d_out;

  // GPU 메모리 할당
  CHECK_CUDA(cudaMalloc(&d_in, C * s * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_w, OC * C * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, OC * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, OC * os * sizeof(float)));

  // GPU로 데이터 복사
  CHECK_CUDA(cudaMemcpy(d_in, in->buf, C * s * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, w->buf, OC * C * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b->buf, OC * sizeof(float), cudaMemcpyHostToDevice));

  // 블록 및 스레드 설정
  size_t total_threads = OC * os;
  size_t block_size = 256;
  size_t grid_size = (total_threads + block_size - 1) / block_size;

  // CUDA 커널 실행
  Conv1D_Kernel<<<grid_size, block_size>>>(d_in, d_w, d_b, d_out, C, s, OC, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 결과를 CPU로 복사
  CHECK_CUDA(cudaMemcpy(out->buf, d_out, OC * os * sizeof(float), cudaMemcpyDeviceToHost));

  // GPU 메모리 해제
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
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

/* [Example] ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}

/* [Example] ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  // GPU memory 할당 및 CPU memory에 있던 데이터를 GPU로 옮기기
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), 
                        cudaMemcpyHostToDevice));

  // 커널 함수를 사용해서 GPU에서 병렬처리
  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  // CPU memory로 다시 옮기기
  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), 
                        cudaMemcpyDeviceToHost));

  // 할당했던 GPU memory 해제하기          
  CHECK_CUDA(cudaFree(d_inout));
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
      max_val = fmaxf(max_val, in[i * s + j]); // CUDA 내장 함수 사용
    }
    out[i] = max_val;
  }
}

void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * C * s));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * C));
  CHECK_CUDA(cudaMemcpy(d_in, in->buf, sizeof(float) * C * s, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (C + blockSize - 1) / blockSize;
  GetMax_Kernel<<<gridSize, blockSize>>>(d_in, d_out, C, s);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, sizeof(float) * C, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
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

void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *out) {
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  float *d_in1, *d_in2, *d_in3, *d_in4, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in1, sizeof(float) * N1));
  CHECK_CUDA(cudaMalloc(&d_in2, sizeof(float) * N2));
  CHECK_CUDA(cudaMalloc(&d_in3, sizeof(float) * N3));
  CHECK_CUDA(cudaMalloc(&d_in4, sizeof(float) * N4));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * (N1 + N2 + N3 + N4)));

  CHECK_CUDA(cudaMemcpy(d_in1, in1->buf, sizeof(float) * N1, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in2, in2->buf, sizeof(float) * N2, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in3, in3->buf, sizeof(float) * N3, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in4, in4->buf, sizeof(float) * N4, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N1 + N2 + N3 + N4 + blockSize - 1) / blockSize;
  Concat_Kernel<<<gridSize, blockSize>>>(d_in1, d_in2, d_in3, d_in4, d_out, N1, N2, N3, N4);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, sizeof(float) * (N1 + N2 + N3 + N4), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_in1));
  CHECK_CUDA(cudaFree(d_in2));
  CHECK_CUDA(cudaFree(d_in3));
  CHECK_CUDA(cudaFree(d_in4));
  CHECK_CUDA(cudaFree(d_out));
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 출력 뉴런 인덱스
  if (i < M) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in[j] * w[i * N + j];  // 행렬 곱셈
    }
    out[i] = val + b[i];  // 편향 추가
  }
}

void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  float *d_in, *d_w, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(float) * N));
  CHECK_CUDA(cudaMalloc((void**)&d_w, sizeof(float) * M * N));
  CHECK_CUDA(cudaMalloc((void**)&d_b, sizeof(float) * M));
  CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float) * M));

  CHECK_CUDA(cudaMemcpy(d_in, in->buf, sizeof(float) * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, w->buf, sizeof(float) * M * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b->buf, sizeof(float) * M, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (M + blockSize - 1) / blockSize;

  Linear_Kernel<<<gridSize, blockSize>>>(d_in, d_w, d_b, d_out, N, M);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, sizeof(float) * M, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
}

/* [Advanced Example] Linear in Half precision on CPU */
void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  for (size_t i = 0; i < M; i++) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += static_cast<float>(half_cpu(in->buf[j]) * 
        half_cpu(w->buf[i * N + j]));
    }
    out->buf[i] = val + b->buf[i];
  }
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

__global__ void Softmax_Kernel(float *buf, size_t N) {
  __shared__ float max_val;
  __shared__ float sum;

  int tid = threadIdx.x;
  if (tid == 0) {
      max_val = -INFINITY;
      for (size_t i = 0; i < N; i++) {
          max_val = fmaxf(max_val, buf[i]);
      }
  }
  __syncthreads();
  if (tid == 0) {
      sum = 0.0f;
      for (size_t i = 0; i < N; i++) {
          buf[i] = expf(buf[i] - max_val);
          sum += buf[i];
      }
  }
  __syncthreads();

  // Step 3: Normalize
  if (tid == 0) {
      for (size_t i = 0; i < N; i++) {
          buf[i] /= sum;
      }
  }
}

void Softmax_CUDA(Tensor *inout) {
  size_t N = inout->shape[0];
  float *d_buf;
  cudaMalloc(&d_buf, N * sizeof(float));
  cudaMemcpy(d_buf, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  Softmax_Kernel<<<1, 1>>>(d_buf, N);
  cudaDeviceSynchronize();
  cudaMemcpy(inout->buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_buf);
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

__global__ void Scaling_Kernel(float *buf, size_t N, float s) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
      buf[idx] *= s;
  }
}

void Scaling_CUDA(Tensor *inout, float s) {
  size_t N = inout->shape[0];
  float *d_buf;
  
  cudaMalloc(&d_buf, N * sizeof(float));
  cudaMemcpy(d_buf, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  Scaling_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_buf, N, s);
  cudaDeviceSynchronize();
  
  cudaMemcpy(inout->buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_buf);
}

__global__ void Add_Kernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
      out[idx] = in1[idx] + in2[idx] + in3[idx] + in4[idx];
  }
}

void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *out) {
  size_t N = in1->shape[0];
  float *d_in1, *d_in2, *d_in3, *d_in4, *d_out;
  
  cudaMalloc(&d_in1, N * sizeof(float));
  cudaMalloc(&d_in2, N * sizeof(float));
  cudaMalloc(&d_in3, N * sizeof(float));
  cudaMalloc(&d_in4, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  
  cudaMemcpy(d_in1, in1->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in3, in3->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in4, in4->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  Add_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in1, d_in2, d_in3, d_in4, d_out, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(out->buf, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_in3);
  cudaFree(d_in4);
  cudaFree(d_out);
}