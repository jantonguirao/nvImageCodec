/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "image_generic_decoder.h"
#include <cassert>
#include <condition_variable>
#include <map>
#include <memory>
#include <thread>
#include "decode_state_batch.h"
#include "device_guard.h"
#include "exception.h"
#include "icode_stream.h"
#include "icodec.h"
#include "iimage.h"
#include "iimage_decoder.h"
#include "log.h"
#include "processing_results.h"

namespace nvimgcdcs {

// Work

/**
 * @brief Describes a sub-batch of work to be processed
 *
 * This object contains a (shared) promise from the original request containing the full batch
 * and a mapping of indices from this sub-batch to the full batch.
 * It also contains a subset of relevant sample views, sources, etc.
 */
struct IWorkManager::Work
{
    Work(const ProcessingResultsPromise& results, const nvimgcdcsDecodeParams_t* params)
        : results_(std::move(results))
        , params_(std::move(params))
    {
    }

    void clear()
    {
        indices_.clear();
        code_streams_.clear();
        images_.clear();
        host_temp_buffers_.clear();
        device_temp_buffers_.clear();
        idx2orig_buffer_.clear();
    }

    int getSamplesNum() const { return indices_.size(); }

    bool empty() const { return indices_.empty(); }

    void resize(int num_samples)
    {
        indices_.resize(num_samples);
        code_streams_.resize(num_samples);
        if (!images_.empty())
            images_.resize(num_samples);
    }

    void init(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
    {
        int N = images.size();

        indices_.reserve(N);
        for (int i = 0; i < N; i++)
            indices_.push_back(i);

        images_.reserve(N);
        for (auto o : images)
            images_.push_back(o);

        code_streams_.reserve(N);
        for (auto cs : code_streams)
            code_streams_.push_back(cs);
    }

    /**
   * @brief Moves one work entry from another work to this one
   */
    void moveEntry(Work* from, int which)
    {
        code_streams_.push_back(from->code_streams_[which]);
        indices_.push_back(from->indices_[which]);
        if (!from->images_.empty())
            images_.push_back(from->images_[which]);
        if (!from->host_temp_buffers_.empty())
            host_temp_buffers_.push_back(std::move(from->host_temp_buffers_[which]));
        if (!from->device_temp_buffers_.empty())
            device_temp_buffers_.push_back(std::move(from->device_temp_buffers_[which]));
        if (from->idx2orig_buffer_.find(which) != from->idx2orig_buffer_.end()) {
            auto entry = from->idx2orig_buffer_.extract(which);
            idx2orig_buffer_.insert(std::move(entry));
        }
    }

    /**
   * @brief Allocates temporary CPU outputs for this sub-batch
   *
   * This function is used when falling back from GPU to CPU decoder.
   */

    void alloc_host_temp_outputs()
    {
        host_temp_buffers_.clear();
        for (int i = 0, n = indices_.size(); i < n; i++) {
            nvimgcdcsImageInfo_t image_info;
            images_[i]->getImageInfo(&image_info);
            void* h_pinned = nullptr;
            CHECK_CUDA(cudaMallocHost(&h_pinned, image_info.buffer_size));
            std::unique_ptr<void, decltype(&cudaFreeHost)> h_ptr(h_pinned, &cudaFreeHost);
            host_temp_buffers_.push_back(std::move(h_ptr));
        }
    }

    void alloc_device_temp_outputs()
    {
        device_temp_buffers_.clear();
        for (int i = 0, n = indices_.size(); i < n; i++) {
            nvimgcdcsImageInfo_t image_info;
            images_[i]->getImageInfo(&image_info);
            void* device_buffer = nullptr;
            CHECK_CUDA(cudaMalloc(&device_buffer, image_info.buffer_size));
            std::unique_ptr<void, decltype(&cudaFree)> d_ptr(device_buffer, &cudaFree);
            device_temp_buffers_.push_back(std::move(d_ptr));
        }
    }

    void ensure_expected_buffer_for_each_image(bool is_device_output)
    {
        for (size_t i = 0; i < images_.size(); ++i) {
            nvimgcdcsImageInfo_t image_info;
            images_[i]->getImageInfo(&image_info);

            if (!is_device_output && image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (host_temp_buffers_.empty()) {
                    alloc_host_temp_outputs();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
                image_info.buffer = host_temp_buffers_[i].get();
                images_[i]->setImageInfo(&image_info);
            }
            if (is_device_output && image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST) {
                if (device_temp_buffers_.empty()) {
                    alloc_device_temp_outputs();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                image_info.buffer = device_temp_buffers_[i].get();
                images_[i]->setImageInfo(&image_info);
            }
        }
    }

    void copy_buffer_if_necessary(bool is_device_output, int sub_idx, cudaStream_t stream, ProcessingResult* r)
    {
        auto it = idx2orig_buffer_.find(sub_idx);
        try {
            if (it != idx2orig_buffer_.end()) {
                nvimgcdcsImageInfo_t image_info;
                images_[sub_idx]->getImageInfo(&image_info);
                auto copy_direction = is_device_output ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                CHECK_CUDA(cudaMemcpyAsync(it->second, image_info.buffer, image_info.buffer_size, copy_direction, stream));
                image_info.buffer = it->second;
                image_info.buffer_kind = image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE
                                             ? NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST
                                             : NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[sub_idx]->setImageInfo(&image_info);
                idx2orig_buffer_.erase(it);
            }
        } catch (...) {
            *r = ProcessingResult::failure(std::current_exception());
        }
    }

    // The original promise
    ProcessingResultsPromise results_;
    // The indices in the original request
    std::vector<int> indices_;
    std::vector<ICodeStream*> code_streams_;
    std::vector<IImage*> images_;
    std::vector<std::unique_ptr<void, decltype(&cudaFreeHost)>> host_temp_buffers_;
    std::vector<std::unique_ptr<void, decltype(&cudaFree)>> device_temp_buffers_;
    std::map<int, void*> idx2orig_buffer_;
    const nvimgcdcsDecodeParams_t* params_;
    std::unique_ptr<Work> next_;
};

// Worker

/**
 * @brief A worker that processes sub-batches of work to be processed by a particular decoder.
 *
 * A Worker waits for incoming Work objects and processes them by running
 * `decoder_->ScheduleDecode` and waiting for partial results, scheduling the failed
 * samples to a fallback decoder, if present.
 *
 * When a sample is successfully decoded, it is marked as a success in the parent
 * DecodeResultsPromise. If it fails, it goes to fallback and only if all fallbacks fail, it is
 * marked in the DecodeResultsPromise as a failure.
 */
class ImageGenericDecoder:: Worker
{
  public:
    /**
   * @brief Constructs a decoder worker for a given decoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the decoder for this worker
   * @param start   - if true, the decoder is immediately instantiated and the worker thread
   *                  is launched; otherwise a call to `start` is delayed until the first
   *                  work that's relevant for this decoder.
   */
    Worker(IWorkManager* work_manager, int device_id, const ICodec* codec, int index)
    {
        work_manager_ = work_manager;
        codec_ = codec;
        index_ = index;
        device_id_ = device_id;
    }
    ~Worker();

    void start();
    void stop();
    void addWork(std::unique_ptr<Work> work);

    ImageGenericDecoder::Worker* getFallback();
    IImageDecoder* getDecoder(const nvimgcdcsDecodeParams_t* params);

  private:
    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback decoder is present.
   */
    void processBatch(std::unique_ptr<Work> work) noexcept;

    /**
   * @brief The main loop of the worker thread.
   */
    void run();

    int device_id_;
    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    IWorkManager* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    std::unique_ptr<IImageDecoder> decoder_;
    bool is_device_output_ = false;
    std::vector<std::unique_ptr<IDecodeState>> decode_states_;
    std::unique_ptr<IDecodeState> decode_state_batch_;
    std::unique_ptr<Worker> fallback_ = nullptr;
};

ImageGenericDecoder::Worker::~Worker()
{
    stop();
}

ImageGenericDecoder::Worker* ImageGenericDecoder::Worker::getFallback()
{
    if (!fallback_) {
        int n = codec_->getDecodersNum();
        if (index_ + 1 < n) {
            fallback_ = std::make_unique<ImageGenericDecoder::Worker>(work_manager_, device_id_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageDecoder* ImageGenericDecoder::Worker::getDecoder(const nvimgcdcsDecodeParams_t* params)
{
    if (!decoder_) {
        decoder_ = codec_->createDecoder(index_, params);
        if (decoder_) {
            decode_state_batch_ = decoder_->createDecodeStateBatch();
            size_t capabilities_size;
            decoder_->getCapabilities(nullptr, &capabilities_size);
            const nvimgcdcsCapability_t* capabilities_ptr;
            decoder_->getCapabilities(&capabilities_ptr, &capabilities_size);
            is_device_output_ =
                std::find(capabilities_ptr, capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) != capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
        }
    }
    return decoder_.get();
}

void ImageGenericDecoder::Worker::start()
{
    std::call_once(started_, [&]() { worker_ = std::thread(&Worker::run, this); });
}

void ImageGenericDecoder::Worker::stop()
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

void ImageGenericDecoder::Worker::run()
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

void ImageGenericDecoder::Worker::addWork(std::unique_ptr<Work> work)
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

static void move_work_to_fallback(IWorkManager::Work* fb, IWorkManager::Work* work, const std::vector<bool>& keep)
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

static void filter_work(IWorkManager::Work* work, const std::vector<bool>& keep)
{
    move_work_to_fallback(nullptr, work, keep);
}

void ImageGenericDecoder::Worker::processBatch(std::unique_ptr<Work> work) noexcept
{
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageDecoder* decoder = getDecoder(work->params_);
    std::vector<bool> mask;
    if (decoder) {
        NVIMGCDCS_LOG_DEBUG("code streams: " << work->code_streams_.size());
        decoder->canDecode(work->code_streams_, work->images_, work->params_, &mask);
    } else {
        NVIMGCDCS_LOG_ERROR("Could not create decoder");
        return;
    }
    std::unique_ptr<IWorkManager::Work> fallback_work;
    auto fallback_worker = getFallback();
    if (fallback_worker) {
        fallback_work = work_manager_->createNewWork(work->results_, work->params_);
        move_work_to_fallback(fallback_work.get(), work.get(), mask);
        if (!fallback_work->empty())
            fallback_worker->addWork(std::move(fallback_work));
    } else {
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i])
            {
                work->results_.set(work->indices_[i], ProcessingResult::failure(nullptr));
            }
        }
        filter_work(work.get(), mask);
    }

    if (!work->code_streams_.empty()) {
        work->ensure_expected_buffer_for_each_image(is_device_output_);
        for (size_t i = 0; i < work->images_.size(); ++i) {
            if (decode_states_.size() == i) {
                decode_states_.push_back(decoder_->createDecodeState());
            }

            work->images_[i]->attachDecodeState(decode_states_[i].get());
        }
        auto future = decoder_->decode(decode_state_batch_.get(), work->code_streams_, work->images_, work->params_);

        for (;;) {
            auto indices = future->waitForNew();
            if (indices.second == 0)
                break; // if wait_new returns with an empty result, it means that everything is ready

            for (size_t i = 0; i < indices.second; ++i) {
                int sub_idx = indices.first[i];
                work->images_[i]->detachDecodeState();
                ProcessingResult r = future->getOne(sub_idx);
                if (r.success) {
                    nvimgcdcsImageInfo_t image_info;
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

//ImageGenericDecoder

ImageGenericDecoder::ImageGenericDecoder(ICodecRegistry* codec_registry)
    : capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT, NVIMGCDCS_CAPABILITY_HOST_OUTPUT, NVIMGCDCS_CAPABILITY_BATCH}
    , codec_registry_(codec_registry)
{
}

ImageGenericDecoder::~ImageGenericDecoder()
{
}

std::unique_ptr<IDecodeState> ImageGenericDecoder::createDecodeState() const
{
    return createDecodeStateBatch();
}

std::unique_ptr<IDecodeState> ImageGenericDecoder::createDecodeStateBatch() const
{
    return std::make_unique<DecodeStateBatch>(nullptr, nullptr);
}

void ImageGenericDecoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_LOG_TRACE("generic_get_capabilities");

    if (capabilities) {
        *capabilities = capabilities_.data();
    }

    if (size) {
        *size = capabilities_.size();
    } else {
        throw Exception(INVALID_PARAMETER, "Could not get decoder capabilities since size pointer is null", "");
    }
}

void ImageGenericDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, [[maybe_unused]] const std::vector<IImage*>& images,
    [[maybe_unused]] const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result) const
{
    result->resize(code_streams.size(), true);
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericDecoder::decode([[maybe_unused]] IDecodeState* decode_state_batch,
    const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params)
{
    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();
    for (size_t i = 0; i < images.size(); ++i) {
        images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
    }

    auto work = createNewWork(std::move(results), params);
    work->init(code_streams, images);

    distributeWork(std::move(work));

    return future;
}

std::unique_ptr<ImageGenericDecoder::Work> ImageGenericDecoder::createNewWork(const ProcessingResultsPromise& results, const void* params)
{
    if (free_work_items_) {
        std::lock_guard<std::mutex> g(work_mutex_);
        if (free_work_items_) {
            auto ptr = std::move(free_work_items_);
            free_work_items_ = std::move(ptr->next_);
            ptr->results_ = std::move(results);
            ptr->params_ = reinterpret_cast<const nvimgcdcsDecodeParams_t*>(params);

            return ptr;
        }
    }

    return std::make_unique<Work>(std::move(results), reinterpret_cast<const nvimgcdcsDecodeParams_t*>(params));
}

void ImageGenericDecoder::recycleWork(std::unique_ptr<IWorkManager::Work> work)
{
    std::lock_guard<std::mutex> g(work_mutex_);
    work->clear();
    work->next_ = std::move(free_work_items_);
    free_work_items_ = std::move(work);
}

void ImageGenericDecoder::combineWork(IWorkManager::Work* target, std::unique_ptr<IWorkManager::Work> source)
{
    //if only one has temporary CPU  storage, allocate it in the other
    if (target->host_temp_buffers_.empty() && !source->host_temp_buffers_.empty())
        target->alloc_host_temp_outputs();
    else if (!target->host_temp_buffers_.empty() && source->host_temp_buffers_.empty())
        source->alloc_host_temp_outputs();
    //if only one has temporary GPU storage, allocate it in the other
    if (target->device_temp_buffers_.empty() && !source->device_temp_buffers_.empty())
        target->alloc_device_temp_outputs();
    else if (!target->device_temp_buffers_.empty() && source->device_temp_buffers_.empty())
        source->alloc_device_temp_outputs();

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

ImageGenericDecoder::Worker* ImageGenericDecoder::getWorker(const ICodec* codec, int device_id)
{
    auto it = workers_.find(codec);
    if (it == workers_.end()) {
        it = workers_.emplace(codec, std::make_unique<Worker>(this, device_id, codec, 0)).first;
    }

    return it->second.get();
}

void ImageGenericDecoder::distributeWork(std::unique_ptr<IWorkManager::Work> work)
{
    std::map<const ICodec*, std::unique_ptr<Work>> dist;
    for (int i = 0; i < work->getSamplesNum(); i++) {
        ICodec* codec = work->code_streams_[i]->getCodec();
        if (!codec) {
            std::stringstream msg_ss;
            msg_ss << "Image #" << work->indices_[i] << " not supported";

            work->results_.set(i, ProcessingResult::failure(std::make_exception_ptr(std::runtime_error(msg_ss.str()))));
            continue;
        }
        auto& w = dist[codec];
        if (!w)
            w = createNewWork(work->results_, work->params_);
        w->moveEntry(work.get(), i);
    }

    int device_id = 0;
    for (int i = 0; i < work->params_->num_backends; ++i) {
        if (work->params_->backends->use_gpu) {
            device_id = work->params_->backends->cuda_device_id;
            break;
        }
    }

    for (auto& [codec, w] : dist) {
        auto worker = getWorker(codec, device_id);
        worker->addWork(std::move(w));
    }
}

} // namespace nvimgcdcs
