#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MahalanobisLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MahalanobisLossLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(128, 6, 1, 1)),
        blob_bottom_1_(new Blob<Dtype>(128, 6, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(3.14);  // +- 3PI to test roll over
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_loss_);

    // mark the last 3 dims as angular
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(3);
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(4);
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(5);
  }
  virtual ~MahalanobisLossLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(MahalanobisLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MahalanobisLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  MahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  const int channels = this->blob_bottom_0_->channels();
  Dtype loss(0);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_0_->cpu_data()[i*channels+j] -
          this->blob_bottom_1_->cpu_data()[i*channels+j];
      if (j > 2) {
        diff = caffe_cpu_min_angle(diff);
      }
      loss += diff*diff;
    }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-4);
}

TYPED_TEST(MahalanobisLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  MahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


template <typename TypeParam>
class MahalanobisLossLayerWeightedTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MahalanobisLossLayerWeightedTest()
      : blob_bottom_0_(new Blob<Dtype>(128, 6, 1, 1)),
        blob_bottom_1_(new Blob<Dtype>(128, 6, 1, 1)),
        blob_bottom_2_(new Blob<Dtype>(128, 21, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_reg_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(3.14);  // +- 3PI to test roll over
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
//     filler_param.set_mean(0.0);
//     filler_param.set_std(5.0);
//     filler = GaussianFiller<Dtype>(filler_param);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_2_);

    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_.push_back(blob_top_reg_);

    // mark the last 3 dims as angular
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(3);
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(4);
    layer_param_.mutable_mahalanobis_loss_param()->add_is_angle(5);
    layer_param_.add_loss_weight(1.0);
    layer_param_.add_loss_weight(1.0);

  }
  virtual ~MahalanobisLossLayerWeightedTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_loss_;
    delete blob_top_reg_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_top_reg_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
};

TYPED_TEST_CASE(MahalanobisLossLayerWeightedTest, TestDtypesAndDevices);

TYPED_TEST(MahalanobisLossLayerWeightedTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  MahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const int num = this->blob_bottom_0_->num();
  const int dim = this->blob_bottom_0_->channels();
  Dtype loss(0);
  Dtype reg(0);
  Dtype *diff = new Dtype[dim];
  Dtype *wdiff = new Dtype[dim];
  Dtype *U = new Dtype[dim*dim];
  // will only modify the upper triangle, set rest to zero
  memset(U, 0.0, dim * dim * sizeof(Dtype));
  for (int n = 0; n < num; ++n) {
    // compute the difference
    for (int i = 0; i < dim; ++i) {
      diff[i] = this->blob_bottom_0_->cpu_data()[n*dim+i] -
          this->blob_bottom_1_->cpu_data()[n*dim+i];
      if (i > 2) {
        diff[i] = caffe_cpu_min_angle(diff[i]);
      }
    }
    // build U
    int ii = 0;
    Dtype eps(1e-3);
    for (int i = 0; i < dim; ++i) {
      for (int j = i; j < dim; ++j) {
        if (i == j) {
          U[i*dim+j] = fabs(this->blob_bottom_2_->data_at(n, ii, 0, 0));
          reg += log(U[i*dim+j] + eps);
        } else {
          U[i*dim+j] = this->blob_bottom_2_->data_at(n, ii, 0, 0);
        }
        ++ii;
      }
    }
    // apply diff = U*diff
    caffe_cpu_gemv(CblasNoTrans, dim, dim, Dtype(1.0), U, diff, Dtype(0.0),
        wdiff);
    // compute loss
    for (int i = 0; i < dim; ++i) {
      loss += wdiff[i]*wdiff[i];
    }
  }
  delete [] diff;
  delete [] wdiff;
  delete [] U;
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-2);
  reg *= Dtype(-2) / static_cast<Dtype>(num);
  EXPECT_NEAR(this->blob_top_reg_->cpu_data()[0], reg, 1e-2);
}

TYPED_TEST(MahalanobisLossLayerWeightedTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  MahalanobisLossLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701, 0.0, 0.1);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 0);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
}

}  // namespace caffe
