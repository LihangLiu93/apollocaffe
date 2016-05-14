#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


// (n,c) .*(c) = (n,c)

namespace caffe {

template <typename Dtype>
void DotMultiplyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  dim_ = bottom[0]->count() / num_;

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(1, dim_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.dot_multiply_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DotMultiplyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);

  vector<int> multiplier_shape(1, num_);
  weights_multiplier_.Reshape(multiplier_shape);
  caffe_set(num_, Dtype(1), weights_multiplier_.mutable_cpu_data());

  vector<int> multiplier2_shape(1, num_);
  multiplier2_shape.push_back(dim_);
  temp_.Reshape(multiplier2_shape);
}

template <typename Dtype>
void DotMultiplyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, (Dtype)1.,
  //     weights_multiplier_.cpu_data(), weight, (Dtype)0., weights_multiplier2_.mutable_cpu_data());
  // caffe_mul<Dtype>(num_*dim_, bottom_data, weights_multiplier2_.cpu_data(), top_data);
  for (int n = 0; n < num_; ++n) {
      caffe_mul<Dtype>(dim_, bottom_data+bottom[0]->offset(n),
                    weight, top_data+top[0]->offset(n));
  }
  
}

template <typename Dtype>
void DotMultiplyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    // caffe_mul<Dtype>(num_*dim_, top_diff, bottom_data, temp_.mutable_cpu_data())
    for (int n = 0; n < num_; ++n) {
        caffe_mul<Dtype>(dim_, top_diff+top[0]->offset(n),
                    bottom_data+bottom[0]->offset(n), temp_.mutable_cpu_data()+temp_.offset(n));
    }
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_, 1, num_, (Dtype)1.,
        temp_.mutable_cpu_data(), weights_multiplier_.cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    // Gradient with respect to bottom data
    // caffe_mul<Dtype>(num_*dim_, top_diff, weights_multiplier2_.cpu_data(), bottom[0]->mutable_cpu_diff());
    for (int n = 0; n < num_; ++n) {
      caffe_mul<Dtype>(dim_, top_diff+top[0]->offset(n),
                    weight, bottom[0]->mutable_cpu_diff()+bottom[0]->offset(n));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotMultiplyLayer);
#endif

INSTANTIATE_CLASS(DotMultiplyLayer);
REGISTER_LAYER_CLASS(DotMultiply);

}  // namespace caffe
