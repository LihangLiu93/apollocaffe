#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bias_term_ = this->layer_param_.sum_param().bias_term();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(1);
      vector<int> bias_shape(1,1);
      this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.sum_param().bias_filler()));
      bias_filler->Fill(this->blobs_[0].get());
    } 
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);

  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, bottom[0]->num());
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(bottom[0]->num(), Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data =top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, bottom_data,
        sum_multiplier_.cpu_data(), 0., top_data);  // summer
  if (bias_term_) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, 1, (Dtype)1., bias_multiplier_.cpu_data(),
        this->blobs_[0]->cpu_data(), (Dtype)1., top_data);
  }
}


template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, (Dtype)1.,
        top_diff, sum_multiplier_.cpu_data(), (Dtype)0., bottom_diff);
  if (bias_term_ && this->param_propagate_down_[0]) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasNoTrans, 1, num, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[0]->mutable_cpu_diff());
  }
}


#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);


}  // namespace caffe
