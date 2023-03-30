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
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nvimgcdcs {

/**
 * @brief Results of a processing operation.
 */
struct ProcessingResult
{
    nvimgcdcsProcessingStatus_t status_ = NVIMGCDCS_PROCESSING_STATUS_UNKNOWN;
    std::exception_ptr exception_ = nullptr;

    static ProcessingResult success() { return {NVIMGCDCS_PROCESSING_STATUS_SUCCESS, {}}; }
    static ProcessingResult failure(nvimgcdcsProcessingStatus_t status) { return {status, {}}; }
    static ProcessingResult failure(std::exception_ptr exception) { return {NVIMGCDCS_PROCESSING_STATUS_ERROR, std::move(exception)}; }
};

class ProcessingResultsSharedState;
class ProcessingResultsFuture;

/**
 * @brief A promise object for processing results.
 *
 * When asynchronous decoding is performed, a promise object and copied among the workers.
 * At exit, a future object is obtained from it by a call to get_future.
 * The promise object is what the workers use to notify the caller about the results.
 * The future object is what the caller uses to wait for and access the results.
 */
class ProcessingResultsPromise
{
  public:
    explicit ProcessingResultsPromise(int num_samples);
    ~ProcessingResultsPromise();

    ProcessingResultsPromise(const ProcessingResultsPromise& other) { *this = other; }
    ProcessingResultsPromise(ProcessingResultsPromise&&) = default;
    ProcessingResultsPromise& operator=(const ProcessingResultsPromise&);
    ProcessingResultsPromise& operator=(ProcessingResultsPromise&&) = default;

    /**
   * @brief Obtains a future object for the caller/consume
   */
    std::unique_ptr<ProcessingResultsFuture> getFuture() const;

    /**
   * @brief The number of samples in this promise
   */
    int getNumSamples() const;

    /**
   * @brief Sets the result for a specific sample
   */
    void set(int index, ProcessingResult res);

    /**
   * @brief Sets all results at once
   */
    void setAll(ProcessingResult* res, size_t size);

    /**
   * @brief Checks if two promises point to the same shared state.
   */
    bool operator==(const ProcessingResultsPromise& other) const { return impl_ == other.impl_; }

    /**
   * @brief Checks if two promises point to different shared states.
   */
    bool operator!=(const ProcessingResultsPromise& other) const { return !(*this == other); }

  private:
    std::shared_ptr<ProcessingResultsSharedState> impl_ = nullptr;
};

/**
 * @brief The object returned by asynchronous decoding requests
 *
 * The future object allows the caller of asynchronous decoding APIs to wait for and obtain
 * partial results, so it can react incrementally to the decoding of mulitple samples,
 * perfomed in the background.
 */
class ProcessingResultsFuture
{
  public:
    ProcessingResultsFuture(ProcessingResultsFuture&& other) = default;
    ProcessingResultsFuture(const ProcessingResultsFuture& other) = delete;

    /**
   * @brief Destroys the future object and terminates the program if the results have
   *        not been consumed
   */
    ~ProcessingResultsFuture();

    ProcessingResultsFuture& operator=(const ProcessingResultsFuture&) = delete;
    ProcessingResultsFuture& operator=(ProcessingResultsFuture&& other)
    {
        std::swap(impl_, other.impl_);
        return *this;
    }

    /**
   * @brief Waits for all results to be ready
   */
    void waitForAll() const;

    /**
   * @brief Waits for any results that have appeared since the previous call to wait_new
   *        (or any results, if this is the first call).
   *
   * @return The indices of results that are ready. They can be read with `get_one` without waiting.
   */
    std::pair<int*, size_t> waitForNew() const;

    /**
   * @brief Waits for the result of a  particualr sample
   */
    void waitForOne(int index) const;

    /**
   * @brief The total number of exepcted results.
   */
    int getNumSamples() const;

    /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
    std::pair<ProcessingResult*, size_t> getAllRef() const;

    /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
    std::vector<ProcessingResult> getAllCopy() const;

    /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
    std::pair<ProcessingResult*, size_t> getAll() const& { return getAllRef(); }

    /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
    std::vector<ProcessingResult> getAll() && { return getAllCopy(); }

    /**
   * @brief Waits for a result and returns it.
   */
    ProcessingResult getOne(int index) const;

  private:
    explicit ProcessingResultsFuture(std::shared_ptr<ProcessingResultsSharedState> impl);
    friend class ProcessingResultsPromise;
    // friend std::unique_ptr<ProcessingResultsFuture> std::make_unique<ProcessingResultsFuture>(
    //     std::shared_ptr<nvimgcdcs::ProcessingResultsSharedState>&);
    // friend std::unique_ptr<ProcessingResultsFuture> std::make_unique<ProcessingResultsFuture>(
    //     std::shared_ptr<nvimgcdcs::ProcessingResultsSharedState>&&);
    std::shared_ptr<ProcessingResultsSharedState> impl_ = nullptr;
};

} // namespace nvimgcdcs
