#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data0 = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data();

  // calculate bottom mean 0
  // caffe_gpu_gemv<Dtype>(CblasTrans, dim_, num_, (Dtype)1.0/num_, bottom_data0,       // wrong calculation ?
  //               bottom_multiplier_.gpu_data(), (Dtype)0., bottom_mean0_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, dim_, num_, (Dtype)1.0/num_,  
                bottom_multiplier_.gpu_data(), bottom_data0, (Dtype)0., bottom_mean0_.mutable_gpu_data());

  // calculate bottom mean 1
  // caffe_gpu_gemv<Dtype>(CblasTrans, dim_, num_, (Dtype)1.0/num_, bottom_data1, 
  //               bottom_multiplier_.gpu_data(), (Dtype)0., bottom_mean1_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, dim_, num_, (Dtype)1.0/num_,  
                bottom_multiplier_.gpu_data(), bottom_data1, (Dtype)0., bottom_mean1_.mutable_gpu_data());

  // calculate diff of mean0 and mean1
  caffe_gpu_sub<Dtype>(dim_, bottom_mean0_.gpu_data(),
            bottom_mean1_.gpu_data(), diff_.mutable_gpu_data());

  // std::cout << "bottom ";
  // for (int i=0;i<5;++i)
  //   std::cout <<bottom[0]->cpu_data()[i] << " ";
  // std::cout<<std::endl;
  
  Dtype dot;
  caffe_gpu_dot<Dtype>(dim_, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / dim_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;                       // !!!!!!! cpu_data instead of gpu_data
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1.0 : -1.0;
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, dim_, 1, sign*top_diff[0]/dim_,
        bottom_multiplier_.gpu_data(), diff_.gpu_data(), (Dtype)0., bottom[i]->mutable_gpu_diff());

    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDLossLayer);

}  // namespace caffe
