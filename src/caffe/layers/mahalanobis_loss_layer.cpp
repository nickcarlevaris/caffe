#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const MahalanobisLossParameter& param =
      this->layer_param_.mahalanobis_loss_param();
  is_angle_.clear();
  std::copy(param.is_angle().begin(), param.is_angle().end(),
      std::back_inserter(is_angle_));
}

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  if (bottom.size() >= 3) {
    CHECK_EQ(top.size(), 2);
    U_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->channels(),
        1);
    // will only modify the upper triangle, set rest to zero
    memset(U_.mutable_cpu_data(), 0.0, U_.count() * sizeof(Dtype));
    // setup second top blob, regularization
    top[1]->ReshapeLike(*top[0]);
  } else {
    U_.Reshape(0, 0, 0, 0);
  }
}

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // pack the information matrices
  // TODO(NCB) is there a more efficient way to do this?
  if (bottom.size() >= 3) {
    for (int n = 0; n < U_.num(); ++n) {
      int ii = 0;
      for (size_t i = 0; i < U_.channels(); ++i) {
        for (size_t j = i; j < U_.height(); ++j) {
          Dtype val = bottom[2]->cpu_data()[(n*bottom[2]->channels() + ii)];
          if (i == j)
            val = fabs(val);
          U_.mutable_cpu_data()[(n*U_.channels() + i)*U_.height() + j] = val;
          ++ii;
        }
      }
    }
  }
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  for (int n = 0; n < diff_.num(); ++n) {
    for (size_t i = 0; i < is_angle_.size(); ++i) {
      int ii = n*diff_.channels() + is_angle_[i];
      diff_.mutable_cpu_data()[ii] = caffe_cpu_min_angle(diff_.cpu_data()[ii]);
    }
    // if there is an information matrix apply it
    // diff = U*diff
// std::cout << "diff_before ";
// for (size_t j = 0; j < diff_.channels(); ++j) {
//   std::cout << diff_.data_at(n, j, 0, 0) << " ";
// }
// std::cout << std::endl;
// std::cout << "U ";
// for (size_t i = 0; i < diff_.channels(); ++i) {
//   std::cout << std::endl;
//   for (size_t j = 0; j < diff_.channels(); ++j) {
//     std::cout << U_.data_at(n, i, j, 0) << " ";
//   }
// }
// std::cout << std::endl;
    if (bottom.size() >= 3) {
      caffe_cpu_gemv(CblasNoTrans, U_.channels(), U_.height(), Dtype(1.0),
          U_.cpu_data(), diff_.cpu_data() + n*diff_.channels(), Dtype(0.0),
          diff_.mutable_cpu_data() + n*diff_.channels());
    }
// std::cout << "diff_after ";
// for (size_t j = 0; j < diff_.channels(); ++j) {
//   std::cout << diff_.data_at(n, j, 0, 0) << " ";
// }
// std::cout << std::endl;
  }
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
  // compute regularizer
  if (bottom.size() >= 3) {
    Dtype reg = Dtype(0.0);
    for (int n = 0; n < U_.num(); ++n) {
      Dtype det = Dtype(1.0);
      for (size_t i = 0; i < U_.channels(); ++i) {
          det *= U_.cpu_data()[(n*U_.channels() + i)*U_.height() + i];
      }
      reg += Dtype(1.0) / (det*det + 1e-6);
    }
    reg = reg / bottom[0]->num();
    top[1]->mutable_cpu_data()[0] = reg;
  }
}

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.cpu_data(),                // a
          Dtype(0),                        // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MahalanobisLossLayer);
#endif

INSTANTIATE_CLASS(MahalanobisLossLayer);
REGISTER_LAYER_CLASS(MahalanobisLoss);
}  // namespace caffe
