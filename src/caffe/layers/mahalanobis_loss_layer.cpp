#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

const double EPS = 1e-3;

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
    Udiff_.ReshapeLike(diff_);
    UtUdiff_.ReshapeLike(diff_);
    // setup second top blob, regularization
    top[1]->ReshapeLike(*top[0]);
  } else {
    U_.Reshape(0, 0, 0, 0);
    Udiff_.Reshape(0, 0, 0, 0);
    UtUdiff_.Reshape(0, 0, 0, 0);
  }
}

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // compute the difference
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  // account for angular roll over
  for (int n = 0; n < diff_.num(); ++n) {
    for (size_t i = 0; i < is_angle_.size(); ++i) {
      int ii = n*diff_.channels() + is_angle_[i];
      diff_.mutable_cpu_data()[ii] = caffe_cpu_min_angle(diff_.cpu_data()[ii]);
    }
  }
  if (bottom.size() >= 3) {  // weighted distance
    // TODO(NCB) is there a more efficient way to do this?
    Dtype reg(0);
    for (int n = 0; n < U_.num(); ++n) {
      // pack the upper-triangular weight matrix (Cholesky factor of information
      // matrix)
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
      // Udiff
      caffe_cpu_gemv(CblasNoTrans, U_.channels(), U_.height(), Dtype(1.0),
          U_.cpu_data() + n*U_.channels()*U_.height(),
          diff_.cpu_data() + n*diff_.channels(), Dtype(0.0),
          Udiff_.mutable_cpu_data() + n*Udiff_.channels());
      // UtUdiff
      caffe_cpu_gemv(CblasTrans, U_.channels(), U_.height(), Dtype(1.0),
          U_.cpu_data() + n*U_.channels()*U_.height(),
          Udiff_.cpu_data() + n*Udiff_.channels(),
          Dtype(0.0), UtUdiff_.mutable_cpu_data() + n*UtUdiff_.channels());
      // compute regularizer
      for (size_t i = 0; i < U_.channels(); ++i) {
          reg += log(U_.data_at(n,i,i,0) + EPS);
      }
    }
    // difftUtUdiff
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), UtUdiff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
    top[1]->mutable_cpu_data()[0] = Dtype(-2.0) * reg / bottom[0]->num();
  } else {  // unweighted distance
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void MahalanobisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      if (bottom.size() >= 3) {
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            UtUdiff_.cpu_data(),             // a
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // b
      } else {
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                           // alpha
            diff_.cpu_data(),                // a
            Dtype(0),                        // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
    }
  }
  if (bottom.size() >= 3 && propagate_down[2]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    int dim = bottom[0]->channels();
    Dtype tmp[dim*dim];
    for (int n = 0; n < U_.num(); ++n) {
      caffe_cpu_gemm(CblasNoTrans,   // trans A
                     CblasNoTrans,   // trans B,
                     dim, dim, 1,  // Dimensions of A and B
                     alpha,
                     Udiff_.cpu_data() + n*dim,
                     diff_.cpu_data() + n*dim,
                     Dtype(0),
                     tmp);
      int ii = 0;
      for (size_t i = 0; i < U_.channels(); ++i) {
        for (size_t j = i; j < U_.height(); ++j) {
          Dtype d_loss = tmp[i*dim+j]; // the contribution from the loss
          // the diagonal elements contribute to the regularizer and have an abs
          // non linearity
          Dtype d_reg(0);
          if (i == j) {
            d_reg = top[1]->cpu_diff()[0] / bottom[0]->num() ;
            d_reg *= Dtype(-2) / (fabs(bottom[2]->data_at(n, ii, 0, 0)) + EPS);
            if (bottom[2]->data_at(n, ii, 0, 0) < 0) {
              d_loss *= -1;
              d_reg *= -1;
            }
          }
          bottom[2]->mutable_cpu_diff()[n*bottom[2]->channels() + ii] =
              d_loss + d_reg;
          ++ii;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MahalanobisLossLayer);
#endif

INSTANTIATE_CLASS(MahalanobisLossLayer);
REGISTER_LAYER_CLASS(MahalanobisLoss);
}  // namespace caffe
