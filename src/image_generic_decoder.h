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

class ImageGenericDecoder : public IImageDecoder, public IWorkManager
{
  public:
    explicit ImageGenericDecoder(ICodecRegistry* codec_registry);
    ~ImageGenericDecoder() override;
    std::unique_ptr<IDecodeState> createDecodeStateBatch() const override;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) override;
    void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params,
        std::vector<bool>* result, std::vector<nvimgcdcsProcessingStatus_t>* status) const override;
    std::unique_ptr<ProcessingResultsFuture> decode(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcdcsDecodeParams_t* params) override;

  private:
    class Worker;
    ImageGenericDecoder::Worker *getWorker(const ICodec* codec, int device_id);

    std::unique_ptr<IWorkManager::Work> createNewWork(
        const ProcessingResultsPromise& results, const void* params);
    void recycleWork(std::unique_ptr<IWorkManager::Work> work) override;
    void combineWork(IWorkManager::Work* target, std::unique_ptr<IWorkManager::Work> source);
    void distributeWork(std::unique_ptr<Work> work);

    std::mutex work_mutex_;
    std::unique_ptr<Work> free_work_items_;
    std::map<const ICodec*, std::unique_ptr<Worker>> workers_;
    std::set<const ICodec*> filtered_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    ICodecRegistry* codec_registry_;
};

} // namespace nvimgcdcs
