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
    Work(std::unique_ptr<ProcessingResultsPromise> results, const nvimgcdcsDecodeParams_t* params)
        : results_(std::move(results))
        , params_(std::move(params))
    {
    }

    void clear()
    {
        indices_.clear();
        code_streams_.clear();
        images_.clear();
        //  temp_buffers.clear();
    }

    int getSamplesNum() const { return indices_.size(); }

    bool empty() const { return indices_.empty(); }

    void resize(int num_samples)
    {
        indices_.resize(num_samples);
        code_streams_.resize(num_samples);
        if (!images_.empty())
            images_.resize(num_samples);
        // if (!temp_buffers.empty())
        //     temp_buffers.resize(num_samples);
    }

    void init(IDecodeState* decode_state_batch, const std::vector<ICodeStream*>& code_streams,
        const std::vector<IImage*>& images)
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

        decode_state_batch_ = decode_state_batch;
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
        // if (!from.temp_buffers.empty())
        //    temp_buffers.push_back(std::move(from.temp_buffers[which]));
    }

    /**
   * @brief Allocates temporary CPU outputs for this sub-batch
   *
   * This function is used when falling back from GPU to CPU decoder.
   */
    //void alloc_temp_outputs();
    // void ImageGenericDecoder::Work::alloc_temp_outputs()
    // {
    //     outputs.resize(indices.size());
    //     temp_buffers.clear();
    //     for (int i = 0, n = indices.size(); i < n; i++) {
    //         SampleView<GPUBackend>& gpu_out = gpu_outputs[i];

    //         // TODO(michalz): Add missing utility functions to SampleView - or just use Tensor again...
    //         size_t size = volume(gpu_out.shape()) * TypeTable::GetTypeInfo(gpu_out.type()).size();
    //         constexpr int kTempBufferAlignment = 256;

    //         temp_buffers.emplace_back();
    //         auto& buf_ptr = temp_buffers.back();
    //         buf_ptr       = mm::alloc_raw_async_unique<char, mm::memory_kind::pinned>(
    //             size, mm::host_sync, ctx.stream, kTempBufferAlignment);

    //         SampleView<CPUBackend>& cpu_out = outputs[i];
    //         cpu_out = SampleView<CPUBackend>(buf_ptr.get(), gpu_out.shape(), gpu_out.type());
    //     }
    // }

    // The original promise
    std::unique_ptr<ProcessingResultsPromise> results_;
    // The indices in the original request
    std::vector<int> indices_;
    std::vector<ICodeStream*> code_streams_;
    std::vector<IImage*> images_;
    //std::vector<mm::async_uptr<void>> temp_buffers;
    const nvimgcdcsDecodeParams_t* params_;
    IDecodeState* decode_state_batch_;
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
class ImageGenericDecoder::Worker
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
    bool produces_gpu_output_ = false;
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
            fallback_ = std::make_unique<ImageGenericDecoder::Worker>(
                work_manager_, device_id_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageDecoder* ImageGenericDecoder::Worker::getDecoder(const nvimgcdcsDecodeParams_t* params)
{
    if (!decoder_) {
        decoder_ = codec_->createDecoder(index_, params);
        if (decoder_) {
            decode_state_batch_ = decoder_->createDecodeStateBatch(nullptr);
        } else {
            //TODO throw
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
        //assert(->rois.empty() || work->rois.size() == work->code_streams_.size());
        //assert(work->temp_buffers.empty() || work->temp_buffers.size() == work->cpu_outputs.size());
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

static void move_work_to_fallback(
    IWorkManager::Work* fb, IWorkManager::Work* work, const std::vector<bool>& keep)
{
    int moved = 0;
    size_t n = work->code_streams_.size();
    for (size_t i = 0; i < n; i++) {
        if (keep[i]) {
            if (moved) {
                // compact
                if (!work->images_.empty())
                    work->images_[i - moved] = work->images_[i];
                //if (!work.temp_buffers.empty())
                //    work.temp_buffers[i - moved] = std::move(work.temp_buffers[i]);
                //if (!work.rois[i])
                //   work.rois[i - moved] = std::move(work.rois[i]);
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
    if (fb)
        fb->resize(fb->getSamplesNum() - moved);
}

static void filter_work(IWorkManager::Work* work, const std::vector<bool>& keep)
{
    move_work_to_fallback(nullptr, work, keep);
}

void ImageGenericDecoder::Worker::processBatch(std::unique_ptr<Work> work) noexcept
{
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());
    //assert(->rois.empty() || work->rois.size() == work->code_streams_.size());
    //assert(work->temp_buffers.empty() || work->temp_buffers.size() == work->cpu_outputs.size());

    IImageDecoder* decoder = getDecoder(work->params_);
    std::vector<bool> mask;
    if (decoder) {
        NVIMGCDCS_LOG_DEBUG("code streams: " << work->code_streams_.size());
        decoder->canDecode(work->code_streams_, work->images_, work->params_, &mask);
    } else {
        //TODO throw
    }
    std::unique_ptr<IWorkManager::Work> fallback_work;
    auto fallback_worker = getFallback();
    if (fallback_worker) {
        fallback_work = work_manager_->createNewWork(std::move(work->results_), work->params_);
        move_work_to_fallback(fallback_work.get(), work.get(), mask);
        if (!fallback_work->empty())
            fallback_worker->addWork(std::move(fallback_work));
    } else {
        filter_work(work.get(), mask);
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i])
                work->results_->set(work->indices_[i], ProcessingResult::failure(nullptr));
        }
    }

    if (!work->code_streams_.empty()) {
        bool decode_to_gpu = produces_gpu_output_;

        // TODO
        // if (!decode_to_gpu && work->cpu_outputs.empty()) {
        //     work->alloc_temp_cpu_outputs();
        for (size_t i = 0; i < work->images_.size(); ++i) {
            if (decode_states_.size() == i) {
                decode_states_.push_back(decoder_->createDecodeState(nullptr));
            }
            work->images_[i]->attachDecodeState(decode_states_[i].get());
        }
        auto future = decoder_->decodeBatch(
            decode_state_batch_.get(), work->code_streams_, work->images_, work->params_);

        for (;;) {
            auto indices = future->waitForNew();
            if (indices.second == 0)
                break; // if wait_new returns with an empty result, it means that everything is ready

            for (size_t i = 0; i < indices.second; ++i) {
                int sub_idx = indices.first[i];
                work->images_[i]->detachDecodeState();
                ProcessingResult r = future->getOne(sub_idx);
                if (r.success) {
                    if (!decode_to_gpu /*&& !work->images_.empty()*/) {
                        try {
                            //TODO
                            // copy(work->gpu_outputs[sub_idx], work->cpu_outputs[sub_idx],
                            //     work->ctx.stream);
                        } catch (...) {
                            r = ProcessingResult::failure(std::current_exception());
                        }
                    }
                    work->results_->set(work->indices_[sub_idx], r);
                } else { // failed to decode
                    if (fallback_worker) {
                        // if there's fallback, we don't set the result, but try to use the fallback first
                        if (!fallback_work)
                            fallback_work = work_manager_->createNewWork(
                                std::move(work->results_), work->params_);
                        fallback_work->moveEntry(work.get(), sub_idx);
                    } else {
                        // no fallback - just propagate the result to the original promise
                        work->results_->set(work->indices_[sub_idx], r);
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
    : capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT, NVIMGCDCS_CAPABILITY_HOST_OUTPUT,
          NVIMGCDCS_CAPABILITY_BATCH}
    , codec_registry_(codec_registry)
{
}

ImageGenericDecoder::~ImageGenericDecoder()
{
}

std::unique_ptr<IDecodeState> ImageGenericDecoder::createDecodeState(
    [[maybe_unused]] cudaStream_t cuda_stream) const
{
    return createDecodeStateBatch(cuda_stream);
}

std::unique_ptr<IDecodeState> ImageGenericDecoder::createDecodeStateBatch(
    [[maybe_unused]] cudaStream_t cuda_stream) const
{
    return std::make_unique<DecodeStateBatch>(nullptr, nullptr, nullptr);
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
        throw Exception(
            INVALID_PARAMETER, "Could not get decoder capabilities since size pointer is null", "");
    }
}

bool ImageGenericDecoder::canDecode([[maybe_unused]] nvimgcdcsCodeStreamDesc_t code_stream,
    nvimgcdcsImageDesc_t image, [[maybe_unused]] const nvimgcdcsDecodeParams_t* params) const
{
    return true;
}

void ImageGenericDecoder::canDecode(const std::vector<ICodeStream*>& code_streams,
    [[maybe_unused]] const std::vector<IImage*>& images,
    [[maybe_unused]] const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result) const
{
    result->resize(code_streams.size(), true);
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericDecoder::decode(
    ICodeStream* code_stream, IImage* image, const nvimgcdcsDecodeParams_t* params)
{
    std::vector<ICodeStream*> code_streams{code_stream};
    std::vector<IImage*> images{image};
    return decodeBatch(image->getAttachedDecodeState(), code_streams, images, params);
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericDecoder::decodeBatch(
    IDecodeState* decode_state_batch, const std::vector<ICodeStream*>& code_streams,
    const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params)
{
    int N = images.size();
    assert(in.size() == N);

    std::unique_ptr<ProcessingResultsPromise> results =
        std::make_unique<ProcessingResultsPromise>(N);
    auto future = results->getFuture();
    for (size_t i = 0; i < images.size(); ++i) {
        images[i]->setProcessingStatus(NVIMGCDCS_PROCESSING_STATUS_DECODING);
    }

    auto work = createNewWork(std::move(results), params);
    work->init(decode_state_batch, code_streams, images);

    distributeWork(std::move(work));

    return future;
}

std::unique_ptr<ImageGenericDecoder::Work> ImageGenericDecoder::createNewWork(
    std::unique_ptr<ProcessingResultsPromise> results, const nvimgcdcsDecodeParams_t* params)
{
    if (free_work_items_) {
        std::lock_guard<std::mutex> g(work_mutex_);
        if (free_work_items_) {
            auto ptr = std::move(free_work_items_);
            free_work_items_ = std::move(ptr->next_);
            ptr->results_ = std::move(results);
            ptr->params_ = std::move(params);
            return ptr;
        }
    }

    return std::make_unique<Work>(std::move(results), params);
}

void ImageGenericDecoder::recycleWork(std::unique_ptr<IWorkManager::Work> work)
{
    std::lock_guard<std::mutex> g(work_mutex_);
    work->clear();
    work->next_ = std::move(free_work_items_);
    free_work_items_ = std::move(work);
}

void ImageGenericDecoder::combineWork(
    IWorkManager::Work* target, std::unique_ptr<IWorkManager::Work> source)
{
    assert(target.results == source->results);

    // if only one has temporary CPU storage, allocate it in the other
    //TODO
    // if (target.temp_buffers.empty() && !source->temp_buffers.empty())
    //     target.alloc_temp_cpu_outputs();
    // else if (!target.temp_buffers.empty() && source->temp_buffers.empty())
    //     source->alloc_temp_cpu_outputs();

    auto move_append = [](auto& dst, auto& src) {
        dst.reserve(dst.size() + src.size());
        for (auto& x : src)
            dst.emplace_back(std::move(x));
    };

    move_append(target->images_, source->images_);
    move_append(target->code_streams_, source->code_streams_);
    move_append(target->indices_, source->indices_);
    //TODO move_append(target.rois, source->rois);
    //TODO move_append(target.temp_buffers, source->temp_buffers);
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

            work->results_->set(i, ProcessingResult::failure(
                                       std::make_exception_ptr(std::runtime_error(msg_ss.str()))));
            continue;
        }
        auto& w = dist[codec];
        if (!w)
            w = createNewWork(std::move(work->results_), work->params_);
        w->moveEntry(work.get(), i);
    }

    int device_id = 0;
    for (int i = 0; i < work->params_->num_backends; ++i) {
        if (work->params_->backends->useGPU) {
            device_id = work->params_->backends->cudaDeviceId;
            break;
        }
    }

    for (auto& [codec, w] : dist) {
        auto worker = getWorker(codec, device_id);
        worker->addWork(std::move(w));
    }
}

} // namespace nvimgcdcs