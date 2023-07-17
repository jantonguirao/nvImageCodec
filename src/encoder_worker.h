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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iencode_state.h"
#include "iimage_encoder.h"
#include "iwork_manager.h"
#include "work.h"

namespace nvimgcdcs {

class ICodec;
class ILogger;

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
class EncoderWorker
{
  public:
    /**
   * @brief Constructs a encoder worker for a given encoder.
   *
   * @param work_manager   - creates and recycles work
   * @param codec   - the factory that constructs the encoder for this worker
   */
    EncoderWorker(ILogger* logger, IWorkManager<nvimgcdcsEncodeParams_t>* work_manager, int device_id, const std::vector<nvimgcdcsBackend_t>& backends,
        const std::string& options, const ICodec* codec, int index);
    ~EncoderWorker();

    void start();
    void stop();
    void addWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work);

    EncoderWorker* getFallback();
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

    ILogger* logger_;
    IWorkManager<nvimgcdcsEncodeParams_t>* work_manager_ = nullptr;
    const ICodec* codec_ = nullptr;
    int index_ = 0;
    int device_id_ = 0;
    const std::vector<nvimgcdcsBackend_t>& backends_;
    const std::string& options_;

    std::mutex mtx_;
    std::condition_variable cv_;

    std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work_;
    std::thread worker_;
    bool stop_requested_ = false;
    std::once_flag started_;

    std::unique_ptr<IImageEncoder> encoder_;
    bool is_input_expected_in_device_ = false;
    std::unique_ptr<IEncodeState> encode_state_batch_;
    std::unique_ptr<EncoderWorker> fallback_ = nullptr;
};


} // namespace nvimgcdcs
