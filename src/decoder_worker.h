/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <condition_variable>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include "work.h"
#include "iimage_decoder.h"
#include "idecode_state.h"
#include "iwork_manager.h"

namespace nvimgcdcs {

class ICodec;

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
class DecoderWorker
{
  public:
    /**
   * @brief Constructs a decoder worker for a given decoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the decoder for this worker
   */
    DecoderWorker(IWorkManager<nvimgcdcsDecodeParams_t>* work_manager, int device_id, const std::vector<nvimgcdcsBackend_t>& backends,
        const std::string& options, const ICodec* codec, int index);
    ~DecoderWorker();

    void addWork(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work);

    DecoderWorker* getFallback();
    IImageDecoder* getDecoder();

  private:
    void start();
    void stop();

    /**
   * @brief Processes a (sub)batch of work.
   *
   * The work is scheduled and the results are waited for. Any failed samples will be added
   * to a fallback work, if a fallback decoder is present.
   */
    void processBatch(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work) noexcept;

    /**
   * @brief The main loop of the worker thread.
   */
    void run();

    IWorkManager<nvimgcdcsDecodeParams_t>* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    int device_id_ = 0;
    const std::vector<nvimgcdcsBackend_t>& backends_;
    const std::string& options_;

    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    std::unique_ptr<IImageDecoder> decoder_;
    bool is_device_output_ = false;
    std::unique_ptr<IDecodeState> decode_state_batch_;
    std::unique_ptr<DecoderWorker> fallback_ = nullptr;
};


}
