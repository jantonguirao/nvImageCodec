/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <vector>
#include <future>
#include "nvimgcodec.h"
#include <iostream>

struct StreamCtx;

/**
 * @brief Context for a single sample
 */
struct SampleCtx {
    StreamCtx* stream;
    int batch_idx;
    nvimgcodecImageDesc_t* image;
    nvimgcodecProcessingStatus_t processing_status;
    const nvimgcodecDecodeParams_t* params;
};

/** 
 * @brief Context for a single encoded stream
 */
struct StreamCtx {
    // Code stream pointer;
    nvimgcodecCodeStreamDesc_t* code_stream_;
    // Unique stream id
    uint64_t code_stream_id_;

    std::vector<SampleCtx*> samples_;

    // Pointer and size of encoded stream (could point to buffer if the io stream can't be mapped)
    void* encoded_stream_data_ = nullptr;
    size_t encoded_stream_data_size_ = 0;

    // Local copy of the stream, in case map is not supported
    std::vector<unsigned char> buffer_;

    size_t size() const {
        return samples_.size();
    }
    
    void reset() {
        setStream(nullptr);
    }

    void clearSamples() {
        samples_.clear();
    }

    void setStream(nvimgcodecCodeStreamDesc_t* code_stream) {
        bool same_stream = code_stream == code_stream_ && code_stream->id == code_stream_->id;
        if (!same_stream) {
            clearSamples();
            code_stream_ = code_stream;
            code_stream_id_ = code_stream ? code_stream->id : 0;
            encoded_stream_data_ = nullptr;
            encoded_stream_data_size_ = 0;
        
            buffer_.clear();
        }
    }

    bool load() {
        if (encoded_stream_data_ == nullptr) {
            nvimgcodecIoStreamDesc_t* io_stream = code_stream_->io_stream;
            io_stream->size(io_stream->instance, &encoded_stream_data_size_);
            void* mapped_encoded_stream_data = nullptr;
            buffer_.clear();
            io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size_);
            if (!mapped_encoded_stream_data) {
                nvtx3::scoped_range marker{"buffer read"};
                buffer_.resize(encoded_stream_data_size_);
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                size_t read_nbytes = 0;
                io_stream->read(io_stream->instance, &read_nbytes, &buffer_[0], encoded_stream_data_size_);
                if (read_nbytes != encoded_stream_data_size_)
                    return false;
                encoded_stream_data_ = &buffer_[0];
            } else {
                encoded_stream_data_ = mapped_encoded_stream_data;
            }
        }
        return true;
    }

    void addSample(SampleCtx* sample)
    {
        samples_.push_back(sample);
    }
};


struct StreamCtxManager {
    using StreamCtxPtr = std::shared_ptr<StreamCtx>;

    StreamCtxPtr getFreeCtx() {
        if (free_ctx_.empty())
            return std::make_shared<StreamCtx>();

        auto ret = std::move(free_ctx_.back());
        free_ctx_.pop_back();
        return ret;
    }

    void releaseCtx(StreamCtxPtr&& ctx) {
        assert(ctx);
        ctx->reset();
        free_ctx_.push_back(std::move(ctx));
    }

    void feedSamples(nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images,
        int batch_size, const nvimgcodecDecodeParams_t* params)
    {
        samples_.clear();
        samples_.resize(batch_size);

        std::map<uint64_t, StreamCtxPtr> new_stream_ctx;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            auto *cs = code_streams[sample_idx];
            auto &ctx = new_stream_ctx[cs->id];
            if (!ctx) {  // if not seen in this iteration
                auto it = stream_ctx_.find(cs->id);  // look for it in the last iteration ctx
                if (it != stream_ctx_.end()) {
                    ctx = std::move(it->second);
                } else {
                    ctx = getFreeCtx();
                }
                ctx->clearSamples();
            }

            // Set stream if needed
            ctx->setStream(cs);

            // Append current sample to the stream context
            auto &sample = samples_[sample_idx];
            sample.stream = ctx.get();
            sample.batch_idx = sample_idx;
            sample.image = images[sample_idx];
            sample.processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
            sample.params = params;
            ctx->addSample(&sample);
        }

        // we used the ones we were interested in, we can release the rest to the pool
        for (auto& [id, ctx] : stream_ctx_) {
            if (ctx)
                releaseCtx(std::move(ctx));
        }

        // only remember stream ctxs for one iteration back
        std::swap(stream_ctx_, new_stream_ctx);

        stream_ctx_view_.clear();
        stream_ctx_view_.reserve(stream_ctx_.size());
        for (auto& [id, ctx] : stream_ctx_)
            stream_ctx_view_.push_back(ctx);
    }

    size_t size() const {
        return stream_ctx_view_.size();
    }

    StreamCtxPtr& operator[](size_t index) {
        return stream_ctx_view_[index];
    }

    StreamCtxPtr& get_by_stream_id(uint64_t stream_id) {
        return stream_ctx_[stream_id];
    }

    SampleCtx& get_sample(int sample_idx) {
        return samples_[sample_idx];
    }

  private:
    std::map<uint64_t, StreamCtxPtr> stream_ctx_;
    std::vector<StreamCtxPtr> free_ctx_;
    std::vector<StreamCtxPtr> stream_ctx_view_;
    std::vector<SampleCtx> samples_;
};
