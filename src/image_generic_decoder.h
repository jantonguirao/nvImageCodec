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

#include <nvimgcodec.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <mutex>

#include "iexecutor.h"
#include "iimage_decoder.h"
#include "iwork_manager.h"
#include "processing_results.h"

namespace nvimgcodec {

class IDecodeState;
class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;
class DecoderWorker;
class ILogger;

class ImageGenericDecoder : public IWorkManager <nvimgcodecDecodeParams_t>
{
  public:
    explicit ImageGenericDecoder(
        ILogger* logger, ICodecRegistry* codec_registry, const nvimgcodecExecutionParams_t* exec_params, const char* options = nullptr);
    ~ImageGenericDecoder();
    void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params,
        nvimgcodecProcessingStatus_t* processing_status, int force_format);
    std::unique_ptr<ProcessingResultsFuture> decode(
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params);

  private:
    DecoderWorker* getWorker(const ICodec* codec);

    std::unique_ptr<Work<nvimgcodecDecodeParams_t>> createNewWork(const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work) override;
    void combineWork(Work<nvimgcodecDecodeParams_t>* target, std::unique_ptr<Work<nvimgcodecDecodeParams_t>> source);
    void distributeWork(std::unique_ptr<Work<nvimgcodecDecodeParams_t>> work);

    ILogger* logger_;
    ICodecRegistry* codec_registry_;
    std::mutex work_mutex_;
    std::unique_ptr<Work<nvimgcodecDecodeParams_t>> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<DecoderWorker>> workers_;
    nvimgcodecExecutionParams_t exec_params_;
    std::vector<nvimgcodecBackend_t> backends_;
    std::string options_;
    std::unique_ptr<IExecutor> executor_;
};

} // namespace nvimgcodec
