/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>

#include <numeric>
#include <stdexcept>

#include "lidar-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace pointpillar {
namespace lidar {

class BackboneImplement : public Backbone {
public:
    virtual ~BackboneImplement() {
        if (cls_) checkRuntime(cudaFree(cls_));
        if (box_) checkRuntime(cudaFree(box_));
        if (dir_) checkRuntime(cudaFree(dir_));
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        // Native CUDA preprocessor/decoder contracts must match this engine.
        if (engine_->num_bindings() != 6 || engine_->has_dynamic_dim()
            || engine_->static_dims("voxels") != std::vector<int>({10000, 32, 10})
            || engine_->static_dims("voxel_idxs") != std::vector<int>({10000, 4})
            || engine_->static_dims("voxel_num") != std::vector<int>({1})
            || engine_->static_dims("cls_preds") != std::vector<int>({1, 248, 216, 18})
            || engine_->static_dims("box_preds") != std::vector<int>({1, 248, 216, 42})
            || engine_->static_dims("dir_cls_preds") != std::vector<int>({1, 248, 216, 12})
            || engine_->dtype("voxels") != TensorRT::DType::HALF
            || engine_->dtype("voxel_idxs") != TensorRT::DType::INT32
            || engine_->dtype("voxel_num") != TensorRT::DType::INT32
            || engine_->dtype("cls_preds") != TensorRT::DType::FLOAT
            || engine_->dtype("box_preds") != TensorRT::DType::FLOAT
            || engine_->dtype("dir_cls_preds") != TensorRT::DType::FLOAT) {
            throw std::runtime_error("PointPillars engine I/O shape or dtype mismatch");
        }

        cls_dims_ = engine_->static_dims("cls_preds");
        box_dims_ = engine_->static_dims("box_preds");
        dir_dims_ = engine_->static_dims("dir_cls_preds");

        int32_t volumn = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&cls_, volumn * sizeof(float)));

        volumn = std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&box_, volumn * sizeof(float)));

        volumn = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&dir_, volumn * sizeof(float)));
        return true;
    }

    virtual void print() override { engine_->print("Lidar Backbone"); }

    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) override {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
        std::vector<const void *> bindings(engine_->num_bindings());
        bindings[engine_->index("voxels")] = voxels;
        bindings[engine_->index("voxel_idxs")] = voxel_idxs;
        bindings[engine_->index("voxel_num")] = params;
        bindings[engine_->index("cls_preds")] = cls_;
        bindings[engine_->index("box_preds")] = box_;
        bindings[engine_->index("dir_cls_preds")] = dir_;
        if (!engine_->forward(bindings, static_cast<cudaStream_t>(_stream))) throw std::runtime_error("PointPillars enqueueV3 failed");
    }

    virtual float* cls() override { return cls_; }
    virtual float* box() override { return box_; }
    virtual float* dir() override { return dir_; }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    float *cls_ = nullptr;
    float *box_ = nullptr;
    float *dir_ = nullptr;
    std::vector<int> cls_dims_, box_dims_, dir_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar
