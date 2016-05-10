#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.gpu_data(), 0., top_data);  // summer
  if (bias_term_) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, 1, (Dtype)1., bias_multiplier_.gpu_data(),
        this->blobs_[0]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1,
        top_diff, sum_multiplier_.gpu_data(), 0., bottom_diff);
  if (bias_term_ && this->param_propagate_down_[0]) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasNoTrans, 1, num, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[0]->mutable_gpu_diff());
  }
  
}


INSTANTIATE_LAYER_GPU_FUNCS(SumLayer);


}  // namespace caffe
