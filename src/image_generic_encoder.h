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
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "iexecutor.h"
#include "iimage_encoder.h"
#include "iwork_manager.h"
#include "processing_results.h"
#include "work.h"

namespace nvimgcodec {

class IEncodeState;
class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;
class EncoderWorker;
class ILogger;

class ImageGenericEncoder: public IWorkManager<nvimgcodecEncodeParams_t>
{
  public:
    explicit ImageGenericEncoder(
        ILogger* logger, ICodecRegistry* codec_registry, const nvimgcodecExecutionParams_t* exec_params, const char* options = nullptr);
    ~ImageGenericEncoder() override;
    void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params,
        nvimgcodecProcessingStatus_t* processing_status, int force_format);
    std::unique_ptr<ProcessingResultsFuture> encode(const std::vector<IImage*>& images,
        const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params);

  private:
    EncoderWorker* getWorker(const ICodec* codec);

    std::unique_ptr<Work<nvimgcodecEncodeParams_t>> createNewWork(
        const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work) override;
    void combineWork(Work<nvimgcodecEncodeParams_t>* target, std::unique_ptr<Work<nvimgcodecEncodeParams_t>> source);
    void distributeWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work);

    ILogger* logger_;
    std::mutex work_mutex_;
    std::unique_ptr<Work<nvimgcodecEncodeParams_t>> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<EncoderWorker>> workers_;
    ICodecRegistry* codec_registry_;
    nvimgcodecExecutionParams_t exec_params_;
    std::vector<nvimgcodecBackend_t> backends_;
    std::string options_;
    std::unique_ptr<IExecutor> executor_;
};

} // namespace nvimgcodec
