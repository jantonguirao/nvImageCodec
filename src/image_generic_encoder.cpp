/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_generic_encoder.h"
#include <cassert>
#include <condition_variable>
#include <memory>
#include <thread>
#include "device_guard.h"
#include "encode_state_batch.h"
#include "exception.h"
#include "icode_stream.h"
#include "icodec.h"
#include "iimage.h"
#include "iimage_encoder.h"
#include "log.h"
#include "processing_results.h"

namespace nvimgcdcs {

/**
 * @brief A worker that processes sub-batches of work to be processed by a particular encoder.
 *
 * A Worker waits for incoming Work objects and processes them by running
 * `encoder_->ScheduleEncode` and waiting for partial results, scheduling the failed
 * samples to a fallback encoder, if present.
 *
 * When a sample is successfully encoded, it is marked as a success in the parent
 * EncodeResultsPromise. If it fails, it goes to fallback and only if all fallbacks fail, it is
 * marked in the EncodeResultsPromise as a failure.
 */
class ImageGenericEncoder::Worker
{
  public:
    /**
   * @brief Constructs a encoder worker for a given encoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the encoder for this worker
   * @param start   - if true, the encoder is immediately instantiated and the worker thread
   *                  is launched; otherwise a call to `start` is delayed until the first
   *                  work that's relevant for this encoder.
   */
    Worker(IWorkManager* work_manager, int device_id, std::string options, const ICodec* codec, int index)
    {
        work_manager_ = work_manager;
        codec_ = codec;
        index_ = index;
        device_id_ = device_id;
        options_ = std::move(options);
    }
    ~Worker();

    void start();
    void stop();
    void addWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work);

    ImageGenericEncoder::Worker* getFallback();
    IImageEncoder* getEncoder();

  private:
    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback encoder is present.
   */
    void processBatch(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work) noexcept;

    /**
   * @brief The main loop of the worker thread.
   */
    void run();

    int device_id_;
    std::string options_;
    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    IWorkManager* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    std::unique_ptr<IImageEncoder> encoder_;
    bool is_input_expected_in_device_ = false;
    std::unique_ptr<IEncodeState> encode_state_batch_;
    std::unique_ptr<Worker> fallback_ = nullptr;
};

ImageGenericEncoder::Worker::~Worker()
{
    stop();
}

ImageGenericEncoder::Worker* ImageGenericEncoder::Worker::getFallback()
{
    if (!fallback_) {
        int n = codec_->getEncodersNum();
        if (index_ + 1 < n) {
            fallback_ = std::make_unique<ImageGenericEncoder::Worker>(work_manager_, device_id_, options_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageEncoder* ImageGenericEncoder::Worker::getEncoder()
{
    if (!encoder_) {
        encoder_ = codec_->createEncoder(index_, device_id_, options_.c_str());
        if (encoder_) {
            encode_state_batch_ = encoder_->createEncodeStateBatch();
            size_t capabilities_size;
            const nvimgcdcsCapability_t* capabilities_ptr;
            encoder_->getCapabilities(&capabilities_ptr, &capabilities_size);
            if (capabilities_size)
                is_input_expected_in_device_ =
                    std::find(capabilities_ptr, capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                        NVIMGCDCS_CAPABILITY_DEVICE_INPUT) != capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
            else
                is_input_expected_in_device_ = false;
        }
    }
    return encoder_.get();
}

void ImageGenericEncoder::Worker::start()
{
    std::call_once(started_, [&]() { worker_ = std::thread(&Worker::run, this); });
}

void ImageGenericEncoder::Worker::stop()
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

void ImageGenericEncoder::Worker::run()
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

void ImageGenericEncoder::Worker::addWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work)
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

void ImageGenericEncoder::Worker::processBatch(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work) noexcept
{
    NVIMGCDCS_LOG_TRACE("processBatch");
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageEncoder* encoder = getEncoder();
    std::vector<bool> mask(work->code_streams_.size());
    std::vector<nvimgcdcsProcessingStatus_t> status(work->code_streams_.size());
    if (encoder) {
        NVIMGCDCS_LOG_DEBUG("code streams: " << work->code_streams_.size());
        encoder->canEncode(work->images_, work->code_streams_, work->params_, &mask, &status);
    } else {
        NVIMGCDCS_LOG_ERROR("Could not create encoder");
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

//ImageGenericEncoder

ImageGenericEncoder::ImageGenericEncoder(ICodecRegistry* codec_registry, int device_id, const char* options)
    : capabilities_{NVIMGCDCS_CAPABILITY_HOST_OUTPUT, NVIMGCDCS_CAPABILITY_DEVICE_INPUT, NVIMGCDCS_CAPABILITY_HOST_INPUT}
    , codec_registry_(codec_registry)
    , device_id_(device_id)
    , options_(options ? options : "")
{
}

ImageGenericEncoder::~ImageGenericEncoder()
{
}

void ImageGenericEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcdcsEncodeParams_t* params, nvimgcdcsProcessingStatus_t* processing_status, bool force_format)
{
    std::map<const ICodec*, std::vector<int>> codec2indices;
    for (size_t i = 0; i < code_streams.size(); i++) {
        ICodec* codec = code_streams[i]->getCodec();
        if (!codec || codec->getEncodersNum() == 0) {
            processing_status[i] = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        auto it = codec2indices.find(codec);
        if (it == codec2indices.end())
            it = codec2indices.insert(std::pair<const ICodec*, std::vector<int>>(codec, std::vector<int>())).first;
        it->second.push_back(i);
    }

    for (auto& entry : codec2indices) {
        std::vector<ICodeStream*> codec_code_streams(entry.second.size());
        std::vector<IImage*> codec_images(entry.second.size());
        std::vector<int> org_idx(entry.second.size());
        for (size_t i = 0; i < entry.second.size(); ++i) {
            org_idx[i] = entry.second[i];
            codec_code_streams[i] = code_streams[entry.second[i]];
            codec_images[i] = images[entry.second[i]];
        }
        auto worker = getWorker(entry.first, device_id_, options_);
        while (worker && codec_code_streams.size() != 0) {
            auto encoder = worker->getEncoder();
            std::vector<bool> mask(codec_code_streams.size());
            std::vector<nvimgcdcsProcessingStatus_t> status(codec_code_streams.size());
            encoder->canEncode(codec_images, codec_code_streams, params, &mask, &status);
            //filter out ready items
            int removed = 0;
            for (size_t i = 0; i < codec_code_streams.size() + removed; ++i) {
                processing_status[org_idx[i - removed]] = status[i];
                bool can_decode_with_other_format_or_params = (static_cast<unsigned int>(status[i]) & 0b11) == 0b01;
                if (status[i] == NVIMGCDCS_PROCESSING_STATUS_SUCCESS || (!force_format && can_decode_with_other_format_or_params)) {
                    codec_code_streams.erase(codec_code_streams.begin() + i - removed);
                    codec_images.erase(codec_images.begin() + i - removed);
                    org_idx.erase(org_idx.begin() + i - removed);
                    ++removed;
                }
            }
            worker = worker->getFallback();
        }
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericEncoder::encode(
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params)
{
    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();

    auto work = createNewWork(std::move(results), params);
    work->init(code_streams, images);

    distributeWork(std::move(work));

    return future;
}

std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> ImageGenericEncoder::createNewWork(
    const ProcessingResultsPromise& results, const void* params)
{
    if (free_work_items_) {
        std::lock_guard<std::mutex> g(work_mutex_);
        if (free_work_items_) {
            auto ptr = std::move(free_work_items_);
            free_work_items_ = std::move(ptr->next_);
            ptr->results_ = std::move(results);
            ptr->params_ = reinterpret_cast<const nvimgcdcsEncodeParams_t*>(params);
            return ptr;
        }
    }

    return std::make_unique<Work<nvimgcdcsEncodeParams_t>>(std::move(results), reinterpret_cast<const nvimgcdcsEncodeParams_t*>(params));
}

void ImageGenericEncoder::recycleWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work)
{
    std::lock_guard<std::mutex> g(work_mutex_);
    work->clear();
    work->next_ = std::move(free_work_items_);
    free_work_items_ = std::move(work);
}

void ImageGenericEncoder::combineWork(Work<nvimgcdcsEncodeParams_t>* target, std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> source)
{
    //if only one has temporary CPU  storage, allocate it in the other
    if (target->host_temp_buffers_.empty() && !source->host_temp_buffers_.empty())
        target->alloc_host_temps();
    else if (!target->host_temp_buffers_.empty() && source->host_temp_buffers_.empty())
        source->alloc_host_temps();
    //if only one has temporary GPU storage, allocate it in the other
    if (target->device_temp_buffers_.empty() && !source->device_temp_buffers_.empty())
        target->alloc_device_temps();
    else if (!target->device_temp_buffers_.empty() && source->device_temp_buffers_.empty())
        source->alloc_device_temps();

    auto move_append = [](auto& dst, auto& src) {
        dst.reserve(dst.size() + src.size());
        for (auto& x : src)
            dst.emplace_back(std::move(x));
    };

    move_append(target->images_, source->images_);
    move_append(target->code_streams_, source->code_streams_);
    move_append(target->indices_, source->indices_);
    move_append(target->host_temp_buffers_, source->host_temp_buffers_);
    move_append(target->device_temp_buffers_, source->device_temp_buffers_);
    std::move(source->idx2orig_buffer_.begin(), source->idx2orig_buffer_.end(),
        std::inserter(target->idx2orig_buffer_, std::end(target->idx2orig_buffer_)));
    recycleWork(std::move(source));
}

ImageGenericEncoder::Worker* ImageGenericEncoder::getWorker(const ICodec* codec, int device_id, const std::string& options)
{
    auto it = workers_.find(codec);
    if (it == workers_.end()) {
        it = workers_.emplace(codec, std::make_unique<Worker>(this, device_id, options, codec, 0)).first;
    }

    return it->second.get();
}

void ImageGenericEncoder::distributeWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work)
{
    NVIMGCDCS_LOG_TRACE("distributeWork");
    std::map<const ICodec*, std::unique_ptr<Work<nvimgcdcsEncodeParams_t>>> dist;
    for (int i = 0; i < work->getSamplesNum(); i++) {
        ICodec* codec = work->code_streams_[i]->getCodec();
        if (!codec) {
            work->results_.set(i, ProcessingResult::failure(NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED));
            continue;
        }
        auto& w = dist[codec];
        if (!w)
            w = createNewWork(work->results_, work->params_);
        w->moveEntry(work.get(), i);
    }

    for (auto& [codec, w] : dist) {
        auto worker = getWorker(codec, device_id_, options_);
        worker->addWork(std::move(w));
    }
}

} // namespace nvimgcdcs
