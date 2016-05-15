#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  num_ = bottom[0]->num();
  dim_ = bottom[0]->count() / num_;

  vector<int> top_shape(1, 1);
  top[0]->Reshape(top_shape);

  vector<int> multiplier_shape(1, num_);
  bottom_multiplier_.Reshape(multiplier_shape);
  caffe_set(num_, Dtype(1), bottom_multiplier_.mutable_cpu_data());

  vector<int> mean_shape(1, dim_);
  bottom_mean0_.Reshape(mean_shape);
  bottom_mean1_.Reshape(mean_shape);
  diff_.Reshape(mean_shape);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // calculate bottom mean 0
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, dim_, num_, (Dtype)1.0/num_,  
                bottom_multiplier_.cpu_data(), bottom_data0, (Dtype)0., bottom_mean0_.mutable_cpu_data());

  // calculate bottom mean 1
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, dim_, num_, (Dtype)1.0/num_,  
                bottom_multiplier_.cpu_data(), bottom_data1, (Dtype)0., bottom_mean1_.mutable_cpu_data());

  // calculate diff of mean0 and mean1
  caffe_sub<Dtype>(dim_, bottom_mean0_.cpu_data(),
            bottom_mean1_.cpu_data(), diff_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot<Dtype>(dim_, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / dim_ / Dtype(2);
  top_data[0] = loss;
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1.0 : -1.0;
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, sign*top_diff[0]/dim_,
        bottom_multiplier_.cpu_data(), diff_.cpu_data(), (Dtype)0., bottom[i]->mutable_cpu_diff());

    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MMDLossLayer);
#endif

INSTANTIATE_CLASS(MMDLossLayer);
REGISTER_LAYER_CLASS(MMDLoss);

}  // namespace caffe
