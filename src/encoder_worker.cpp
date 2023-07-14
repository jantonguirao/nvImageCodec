/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "encoder_worker.h"

#include <cassert>

#include "device_guard.h"
#include "icodec.h"
#include "iimage_encoder_factory.h"
#include "log.h"

namespace nvimgcdcs {

EncoderWorker::EncoderWorker(ILogger* logger, IWorkManager<nvimgcdcsEncodeParams_t>* work_manager, int device_id,
    const std::vector<nvimgcdcsBackend_t>& backends, const std::string& options, const ICodec* codec, int index)
    : logger_(logger)
    , work_manager_(work_manager)
    , codec_(codec)
    , index_(index)
    , device_id_(device_id)
    , backends_(backends)
    , options_(options)
{
}

EncoderWorker::~EncoderWorker()
{
    stop();
}

EncoderWorker* EncoderWorker::getFallback()
{
    if (!fallback_) {
        int n = codec_->getEncodersNum();
        if (index_ + 1 < n) {
            fallback_ = std::make_unique<EncoderWorker>(logger_, work_manager_, device_id_, backends_, options_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageEncoder* EncoderWorker::getEncoder()
{
    while (!encoder_ && (index_ < codec_->getEncodersNum())) {
        auto encoder_factory = codec_->getEncoderFactory(index_);
        if (encoder_factory) {
            auto backend_kind = encoder_factory->getBackendKind();
            auto backend_it =
                find_if(backends_.begin(), backends_.end(), [backend_kind](const nvimgcdcsBackend_t& b) { return b.kind == backend_kind; });
            if (backends_.size() == 0 || backend_it != backends_.end()) {
                auto backend_params = backend_it != backends_.end() ? &backend_it->params : nullptr;
                encoder_ = encoder_factory->createEncoder(device_id_, backend_params, options_.c_str());
                if (encoder_) {
                    encode_state_batch_ = encoder_->createEncodeStateBatch();
                    is_input_expected_in_device_ = backend_kind != NVIMGCDCS_BACKEND_KIND_CPU_ONLY;
                }
            } else {
                index_++;
            }
        } else {
            index_++;
        }
    }
    return encoder_.get();
}

void EncoderWorker::start()
{
    std::call_once(started_, [&]() { worker_ = std::thread(&EncoderWorker::run, this); });
}

void EncoderWorker::stop()
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

void EncoderWorker::run()
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

void EncoderWorker::addWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work)
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

static void move_work_to_fallback(Work<nvimgcdcsEncodeParams_t>* fb, Work<nvimgcdcsEncodeParams_t>* work, const std::vector<bool>& keep)
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

static void filter_work(Work<nvimgcdcsEncodeParams_t>* work, const std::vector<bool>& keep)
{
    move_work_to_fallback(nullptr, work, keep);
}

void EncoderWorker::processBatch(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work) noexcept
{
    NVIMGCDCS_LOG_TRACE(logger_, "processBatch");
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageEncoder* encoder = getEncoder();
    std::vector<bool> mask(work->code_streams_.size());
    std::vector<nvimgcdcsProcessingStatus_t> status(work->code_streams_.size());
    if (encoder) {
        NVIMGCDCS_LOG_DEBUG(logger_, "code streams: " << work->code_streams_.size());
        encoder->canEncode(work->images_, work->code_streams_, work->params_, &mask, &status);
    } else {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not create encoder");
        work->results_.setAll(ProcessingResult::failure(NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED));
        return;
    }
    std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> fallback_work;
    auto fallback_worker = getFallback();
    if (fallback_worker) {
        fallback_work = work_manager_->createNewWork(work->results_, work->params_);
        move_work_to_fallback(fallback_work.get(), work.get(), mask);
        if (!fallback_work->empty())
            fallback_worker->addWork(std::move(fallback_work));
    } else {
        filter_work(work.get(), mask);
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i])
                work->results_.set(work->indices_[i], ProcessingResult::failure(status[i]));
        }
    }

    if (!work->code_streams_.empty()) {
        work->ensure_expected_buffer_for_encode_each_image(is_input_expected_in_device_);
        auto future = encoder_->encode(encode_state_batch_.get(), work->images_, work->code_streams_, work->params_);

        for (;;) {
            auto indices = future->waitForNew();
            if (indices.second == 0)
                break; // if wait_new returns with an empty result, it means that everything is ready

            for (size_t i = 0; i < indices.second; ++i) {
                int sub_idx = indices.first[i];
                ProcessingResult r = future->getOne(sub_idx);
                if (r.success) {
                    work->clean_after_encoding(is_input_expected_in_device_, sub_idx, &r);
                    work->results_.set(work->indices_[sub_idx], r);
                } else { // failed to encode
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
