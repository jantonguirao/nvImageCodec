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
    Work(const ProcessingResultsPromise& results, const nvimgcdcsEncodeParams_t* params)
        : results_(std::move(results))
        , params_(std::move(params))
        , encode_state_batch_(nullptr)
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

    void init(IEncodeState* encode_state_batch, const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
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

        encode_state_batch_ = encode_state_batch;
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
   * This function is used when falling back from GPU to CPU encoder.
   */

    void alloc_host_temp_inputs()
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

    void alloc_device_temp_inputs()
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

    void ensure_expected_buffer_for_each_image(bool is_input_expected_in_device)
    {
        cudaEvent_t event;
        CHECK_CUDA(cudaEventCreate(&event));

        for (size_t i = 0; i < images_.size(); ++i) {
            nvimgcdcsImageInfo_t image_info;
            images_[i]->getImageInfo(&image_info);

            if (!is_input_expected_in_device && image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (host_temp_buffers_.empty()) {
                    alloc_host_temp_inputs();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer = host_temp_buffers_[i].get();
                image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
                images_[i]->setImageInfo(&image_info);

                CHECK_CUDA(cudaMemcpyAsync(image_info.buffer, idx2orig_buffer_[i], image_info.buffer_size, cudaMemcpyDeviceToHost));
            }
            if (is_input_expected_in_device && image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST) {
                if (device_temp_buffers_.empty()) {
                    alloc_device_temp_inputs();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer = device_temp_buffers_[i].get();
                image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[i]->setImageInfo(&image_info);

                CHECK_CUDA(cudaMemcpyAsync(image_info.buffer, idx2orig_buffer_[i], image_info.buffer_size, cudaMemcpyHostToDevice));
            }
            images_[i]->setImageInfo(&image_info);
        }

        CHECK_CUDA(cudaEventRecord(event));
        CHECK_CUDA(cudaEventSynchronize(event));
        CHECK_CUDA(cudaEventDestroy(event));
    }

    void clean_after_encoding(bool is_input_expected_in_device, int sub_idx, ProcessingResult* r)
    {
        auto it = idx2orig_buffer_.find(sub_idx);
        if (it != idx2orig_buffer_.end()) {
            try {
                nvimgcdcsImageInfo_t image_info;
                images_[sub_idx]->getImageInfo(&image_info);
                image_info.buffer = it->second;
                image_info.buffer_kind = image_info.buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE
                                             ? NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST
                                             : NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[sub_idx]->setImageInfo(&image_info);
                idx2orig_buffer_.erase(it);
            } catch (...) {
                *r = ProcessingResult::failure(std::current_exception());
            }
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
    const nvimgcdcsEncodeParams_t* params_;
    IEncodeState* encode_state_batch_;
    std::unique_ptr<Work> next_;
};

// Worker

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

    ImageGenericEncoder::Worker* getFallback();
    IImageEncoder* getEncoder(const nvimgcdcsEncodeParams_t* params);

  private:
    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback encoder is present.
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
            fallback_ = std::make_unique<ImageGenericEncoder::Worker>(work_manager_, device_id_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageEncoder* ImageGenericEncoder::Worker::getEncoder(const nvimgcdcsEncodeParams_t* params)
{
    if (!encoder_) {
        encoder_ = codec_->createEncoder(index_, params);
        if (encoder_) {
            encode_state_batch_ = encoder_->createEncodeStateBatch();
            size_t capabilities_size;
            encoder_->getCapabilities(nullptr, &capabilities_size);
            const nvimgcdcsCapability_t* capabilities_ptr;
            encoder_->getCapabilities(&capabilities_ptr, &capabilities_size);
            is_input_expected_in_device_ =
                std::find(capabilities_ptr, capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_DEVICE_INPUT) != capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
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

void ImageGenericEncoder::Worker::addWork(std::unique_ptr<Work> work)
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

void ImageGenericEncoder::Worker::processBatch(std::unique_ptr<Work> work) noexcept
{
    NVIMGCDCS_LOG_TRACE("processBatch");
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageEncoder* encoder = getEncoder(work->params_);
    std::vector<bool> mask;
    if (encoder) {
        NVIMGCDCS_LOG_DEBUG("code streams: " << work->code_streams_.size());
        encoder->canEncode(work->images_, work->code_streams_, work->params_, &mask);
    } else {
        NVIMGCDCS_LOG_ERROR("Could not create encoder");
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
        filter_work(work.get(), mask);
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i])
                work->results_.set(work->indices_[i], ProcessingResult::failure(nullptr));
        }
    }

    if (!work->code_streams_.empty()) {
        work->ensure_expected_buffer_for_each_image(is_input_expected_in_device_);
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

ImageGenericEncoder::ImageGenericEncoder(ICodecRegistry* codec_registry)
    : capabilities_{NVIMGCDCS_CAPABILITY_HOST_OUTPUT, NVIMGCDCS_CAPABILITY_DEVICE_INPUT, NVIMGCDCS_CAPABILITY_HOST_INPUT}
    , codec_registry_(codec_registry)
{
}

ImageGenericEncoder::~ImageGenericEncoder()
{
}

std::unique_ptr<IEncodeState> ImageGenericEncoder::createEncodeStateBatch() const
{
    return std::make_unique<EncodeStateBatch>();
}

void ImageGenericEncoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_LOG_TRACE("generic_get_capabilities");

    if (capabilities) {
        *capabilities = capabilities_.data();
    }

    if (size) {
        *size = capabilities_.size();
    } else {
        throw Exception(INVALID_PARAMETER, "Could not get encoder capabilities since size pointer is null", "");
    }
}

void ImageGenericEncoder::canEncode([[maybe_unused]] const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    [[maybe_unused]] const nvimgcdcsEncodeParams_t* params, std::vector<bool>* result) const
{
    result->resize(code_streams.size(), true);
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericEncoder::encode(IEncodeState* encode_state_batch,
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params)
{
    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();

    auto work = createNewWork(std::move(results), params);
    work->init(encode_state_batch, code_streams, images);

    distributeWork(std::move(work));

    return future;
}

std::unique_ptr<ImageGenericEncoder::Work> ImageGenericEncoder::createNewWork(const ProcessingResultsPromise& results, const void* params)
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

    return std::make_unique<Work>(std::move(results), reinterpret_cast<const nvimgcdcsEncodeParams_t*>(params));
}

void ImageGenericEncoder::recycleWork(std::unique_ptr<IWorkManager::Work> work)
{
    std::lock_guard<std::mutex> g(work_mutex_);
    work->clear();
    work->next_ = std::move(free_work_items_);
    free_work_items_ = std::move(work);
}

void ImageGenericEncoder::combineWork(IWorkManager::Work* target, std::unique_ptr<IWorkManager::Work> source)
{
    //if only one has temporary CPU  storage, allocate it in the other
    if (target->host_temp_buffers_.empty() && !source->host_temp_buffers_.empty())
        target->alloc_host_temp_inputs();
    else if (!target->host_temp_buffers_.empty() && source->host_temp_buffers_.empty())
        source->alloc_host_temp_inputs();
    //if only one has temporary GPU storage, allocate it in the other
    if (target->device_temp_buffers_.empty() && !source->device_temp_buffers_.empty())
        target->alloc_device_temp_inputs();
    else if (!target->device_temp_buffers_.empty() && source->device_temp_buffers_.empty())
        source->alloc_device_temp_inputs();

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

ImageGenericEncoder::Worker* ImageGenericEncoder::getWorker(const ICodec* codec, int device_id)
{
    auto it = workers_.find(codec);
    if (it == workers_.end()) {
        it = workers_.emplace(codec, std::make_unique<Worker>(this, device_id, codec, 0)).first;
    }

    return it->second.get();
}

void ImageGenericEncoder::distributeWork(std::unique_ptr<IWorkManager::Work> work)
{
    NVIMGCDCS_LOG_TRACE("distributeWork");
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
