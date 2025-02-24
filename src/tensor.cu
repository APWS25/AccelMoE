#include "model.h"


/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

 // 생성자: 기본 버전
Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //buf = (float *) calloc(N_, sizeof(float)); // 기존
  CHECK_CUDA(cudaMallocHost((void**)&buf, sizeof(float) * N_)); // 변경
  memset(buf, 0, sizeof(float) * N_); // 변경
}

// 생성자: 기존 버퍼 사용
Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //buf = (float *) malloc(N_ * sizeof(float)); // 기존
  CHECK_CUDA(cudaMallocHost((void**)&buf, sizeof(float) * N_)); // 변경
  memcpy(buf, buf_, N_ * sizeof(float));
}

// Tensor 소멸자
Tensor::~Tensor() {
  //if (buf != nullptr) free(buf); // 기존
  if (buf != nullptr) CHECK_CUDA(cudaFreeHost(buf)); // 변경
}

// 텐서의 총 원소 개수를 반환하는 함수
size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}