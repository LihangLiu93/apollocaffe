#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void DotMultiplyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, (Dtype)1.,
  //     weights_multiplier_.gpu_data(), weight, (Dtype)0., weights_multiplier2_.mutable_gpu_data());
  // caffe_gpu_mul<Dtype>(num_*dim_, bottom_data, weights_multiplier2_.gpu_data(), top_data);
  for (int n = 0; n < num_; ++n) {
      caffe_gpu_mul<Dtype>(dim_, bottom_data+bottom[0]->offset(n),
                    weight, top_data+top[0]->offset(n));
  }
  
}

template <typename Dtype>
void DotMultiplyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    // caffe_gpu_mul<Dtype>(num_*dim_, top_diff, bottom_data, temp_.mutable_gpu_data())
    for (int n = 0; n < num_; ++n) {
        caffe_gpu_mul<Dtype>(dim_, top_diff+top[0]->offset(n),
                    bottom_data+bottom[0]->offset(n), temp_.mutable_gpu_data()+temp_.offset(n));
    }
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_, 1, num_, (Dtype)1.,
        temp_.mutable_gpu_data(), weights_multiplier_.gpu_data(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    // Gradient with respect to bottom data
    // caffe_gpu_mul<Dtype>(num_*dim_, top_diff, weights_multiplier2_.gpu_data(), bottom[0]->mutable_gpu_diff());
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_mul<Dtype>(dim_, top_diff+top[0]->offset(n),
                    weight, bottom[0]->mutable_gpu_diff()+bottom[0]->offset(n));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotMultiplyLayer);

}  // namespace caffe
