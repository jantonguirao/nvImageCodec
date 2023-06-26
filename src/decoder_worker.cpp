/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "decoder_worker.h"

#include <cassert>

#include "device_guard.h"
#include "icodec.h"
#include "iimage_decoder_factory.h"
#include "log.h"

namespace nvimgcdcs {

DecoderWorker::DecoderWorker(IWorkManager<nvimgcdcsDecodeParams_t>* work_manager, int device_id, const std::vector<nvimgcdcsBackend_t>& backends,
    const std::string& options, const ICodec* codec, int index)
    : work_manager_(work_manager)
    , codec_(codec)
    , index_(index)
    , device_id_(device_id)
    , backends_(backends)
    , options_(options)
{
}

DecoderWorker::~DecoderWorker()
{
    stop();
}

DecoderWorker* DecoderWorker::getFallback()
{
    if (!fallback_) {
        int n = codec_->getDecodersNum();
        if (index_ + 1 < n) {
            fallback_ = std::make_unique<DecoderWorker>(work_manager_, device_id_, backends_, options_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageDecoder* DecoderWorker::getDecoder()
{
    while (!decoder_ && (index_ < codec_->getDecodersNum())) {
        auto decoder_factory = codec_->getDecoderFactory(index_);
        if (decoder_factory) {
            auto backend_kind = decoder_factory->getBackendKind();
            auto backend_it =
                find_if(backends_.begin(), backends_.end(), [backend_kind](const nvimgcdcsBackend_t& b) { return b.kind == backend_kind; });
            if (backends_.size() == 0 || backend_it != backends_.end()) {
                auto backend_params = backend_it != backends_.end() ? &backend_it->params : nullptr;
                decoder_ = decoder_factory->createDecoder(device_id_, backend_params, options_.c_str());
                if (decoder_) {
                    decode_state_batch_ = decoder_->createDecodeStateBatch();
                    is_device_output_ = backend_kind != NVIMGCDCS_BACKEND_KIND_CPU_ONLY;
                }
            } else {
                index_++;
            }
        } else {
            index_++;
        }
    }
    return decoder_.get();
}

void DecoderWorker::start()
{
    std::call_once(started_, [&]() { worker_ = std::thread(&DecoderWorker::run, this); });
}

void DecoderWorker::stop()
{
    if (worker_.joinable()) {
        {
            std::lock_guard lock(mtx_);
            stop_requested_ = true;
            work_.reset();
        }
        cv_.notify_all();
        worker_.join();
        worker_ = {};
    }
}

void DecoderWorker::run()
{
    DeviceGuard dg(device_id_);
    std::unique_lock lock(mtx_, std::defer_lock);
    while (!stop_requested_) {
        lock.lock();
        cv_.wait(lock, [&]() { return stop_requested_ || work_ != nullptr; });
        if (stop_requested_)
            break;
        assert(work_ != nullptr);
        auto w = std::move(work_);
        lock.unlock();
        processBatch(std::move(w));
    }
}

void DecoderWorker::addWork(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work)
{
    assert(work->getSamplesNum() > 0);
    {
        std::lock_guard guard(mtx_);
        assert((work->images_.size() == work->code_streams_.size()));
        if (work_) {
            work_manager_->combineWork(work_.get(), std::move(work));
            // no need to notify - a work item was already there, so it will be picked up regardless
        } else {
            work_ = std::move(work);
            cv_.notify_one();
        }
    }
    start();
}

static void move_work_to_fallback(Work<nvimgcdcsDecodeParams_t>* fb, Work<nvimgcdcsDecodeParams_t>* work, const std::vector<bool>& keep)
{
    int moved = 0;
    size_t n = work->code_streams_.size();
    for (size_t i = 0; i < n; i++) {
        if (keep[i]) {
            if (moved) {
                // compact
                if (!work->images_.empty())
                    work->images_[i - moved] = work->images_[i];
                if (!work->host_temp_buffers_.empty())
                    work->host_temp_buffers_[i - moved] = std::move(work->host_temp_buffers_[i]);
                if (!work->device_temp_buffers_.empty())
                    work->device_temp_buffers_[i - moved] = std::move(work->device_temp_buffers_[i]);
                if (!work->idx2orig_buffer_.empty())
                    work->idx2orig_buffer_[i - moved] = std::move(work->idx2orig_buffer_[i]);
                if (!work->code_streams_.empty())
                    work->code_streams_[i - moved] = work->code_streams_[i];
                work->indices_[i - moved] = work->indices_[i];
            }
        } else {
            if (fb)
                fb->moveEntry(work, i);
            moved++;
        }
    }
    if (moved)
        work->resize(n - moved);
}

static void filter_work(Work<nvimgcdcsDecodeParams_t>* work, const std::vector<bool>& keep)
{
    move_work_to_fallback(nullptr, work, keep);
}

void DecoderWorker::processBatch(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work) noexcept
{
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageDecoder* decoder = getDecoder();
    std::vector<bool> mask(work->code_streams_.size());
    std::vector<nvimgcdcsProcessingStatus_t> status(work->code_streams_.size());
    if (decoder) {
        NVIMGCDCS_LOG_DEBUG("code streams: " << work->code_streams_.size());
        decoder->canDecode(work->code_streams_, work->images_, work->params_, &mask, &status);
        for (size_t i = 0; i < work->code_streams_.size(); i++) {
            NVIMGCDCS_LOG_DEBUG("canDecode status #" << i << " : " << status[i]);
        }
    } else {
        NVIMGCDCS_LOG_ERROR("Could not create decoder");
        work->results_.setAll(ProcessingResult::failure(NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED));
        return;
    }
    std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> fallback_work;
    auto fallback_worker = getFallback();
    if (fallback_worker) {
        fallback_work = work_manager_->createNewWork(work->results_, work->params_);
        move_work_to_fallback(fallback_work.get(), work.get(), mask);
        if (!fallback_work->empty())
            fallback_worker->addWork(std::move(fallback_work));
    } else {
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i]) {
                work->results_.set(work->indices_[i], ProcessingResult::failure(status[i]));
            }
        }
        filter_work(work.get(), mask);
    }

    if (!work->code_streams_.empty()) {
        work->ensure_expected_buffer_for_decode_each_image(is_device_output_);
        auto future = decoder_->decode(decode_state_batch_.get(), work->code_streams_, work->images_, work->params_);

        for (;;) {
            auto indices = future->waitForNew();
            if (indices.second == 0)
                break; // if wait_new returns with an empty result, it means that everything is ready

            for (size_t i = 0; i < indices.second; ++i) {
                int sub_idx = indices.first[i];
                ProcessingResult r = future->getOne(sub_idx);
                if (r.isSuccess()) {
                    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                    work->images_[i]->getImageInfo(&image_info);
                    work->copy_buffer_if_necessary(is_device_output_, sub_idx, image_info.cuda_stream, &r);
                    work->results_.set(work->indices_[sub_idx], r);
                } else { // failed to decode
                    if (fallback_worker) {
                        // if there's fallback, we don't set the result, but try to use the fallback first
                        if (!fallback_work)
                            fallback_work = work_manager_->createNewWork(work->results_, work->params_);
                        fallback_work->moveEntry(work.get(), sub_idx);
                    } else {
                        // no fallback - just propagate the result to the original promise
                        work->results_.set(work->indices_[sub_idx], r);
                    }
                }
            }

            if (fallback_work && !fallback_work->empty())
                fallback_worker->addWork(std::move(fallback_work));
        }
    }
    work_manager_->recycleWork(std::move(work));
}

} // namespace nvimgcdcs
