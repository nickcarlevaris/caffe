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
  // check the gradient for the first bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
