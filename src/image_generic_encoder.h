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
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "iimage_encoder.h"
#include "iwork_manager.h"
#include "processing_results.h"
#include "work.h"

namespace nvimgcdcs {

class IEncodeState;
class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;
class EncoderWorker;

class ImageGenericEncoder: public IWorkManager<nvimgcdcsEncodeParams_t>
{
  public:
    explicit ImageGenericEncoder(int device_id, int num_backends, const nvimgcdcsBackend_t* backends, const char *options = nullptr);
    ~ImageGenericEncoder() override;
    void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params,
        nvimgcdcsProcessingStatus_t* processing_status, bool force_format);
    std::unique_ptr<ProcessingResultsFuture> encode(const std::vector<IImage*>& images,
        const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params);

  private:
    EncoderWorker* getWorker(const ICodec* codec);

    std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> createNewWork(
        const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work) override;
    void combineWork(Work<nvimgcdcsEncodeParams_t>* target, std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> source);
    void distributeWork(std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> work);

    std::mutex work_mutex_;
    std::unique_ptr<Work<nvimgcdcsEncodeParams_t>> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<EncoderWorker>> workers_;
    ICodecRegistry* codec_registry_;
    int device_id_;
    std::vector<nvimgcdcsBackend_t> backends_;
    std::string options_;
};

} // namespace nvimgcdcs
