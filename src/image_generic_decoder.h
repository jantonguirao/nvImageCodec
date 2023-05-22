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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <mutex>
#include "iimage_decoder.h"
#include "processing_results.h"
#include "iwork_manager.h"

namespace nvimgcdcs {

class IDecodeState;
class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;

class ImageGenericDecoder : public IWorkManager <nvimgcdcsDecodeParams_t>
{
  public:
    explicit ImageGenericDecoder(ICodecRegistry* codec_registry, int device_id, const char *options = nullptr);
    ~ImageGenericDecoder();
    void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params,
        nvimgcdcsProcessingStatus_t* processing_status, bool force_format);
    std::unique_ptr<ProcessingResultsFuture> decode(
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params);

  private:
    class Worker;
    ImageGenericDecoder::Worker *getWorker(const ICodec* codec, int device_id, const std::string& options);

    std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> createNewWork(const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work) override;
    void combineWork(Work<nvimgcdcsDecodeParams_t>* target, std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> source);
    void distributeWork(std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> work);

    std::mutex work_mutex_;
    std::unique_ptr<Work<nvimgcdcsDecodeParams_t>> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<Worker>> workers_;
    ICodecRegistry* codec_registry_;
    int device_id_;
    std::string options_;
};

} // namespace nvimgcdcs
