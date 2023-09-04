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

#include <nppdefs.h>
#include <nvimgcodecs.h>
#include <nvjpeg2k.h>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "error_handling.h"

namespace nvjpeg2k {

class NvJpeg2kDecoderPlugin
{
  public:
    explicit NvJpeg2kDecoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsDecoderDesc_t* getDecoderDesc();

  private:
    struct Decoder;

    struct ParseState
    {
        explicit ParseState(const char* id, const nvimgcdcsFrameworkDesc_t* framework);
        ~ParseState();

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::vector<unsigned char> buffer_;
    };

    struct DecodeState
    {
        explicit DecodeState(const char* id, const nvimgcdcsFrameworkDesc_t* framework, nvjpeg2kHandle_t handle,
            nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator, int device_id, int num_threads,
            int num_parallel_tiles);
        ~DecodeState();

        struct PerThreadResources
        {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kDecodeState_t state_;
            std::unique_ptr<ParseState> parse_state_;
            NppStreamContext npp_ctx_;
        };

        struct PerTileResources {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kDecodeState_t state_;
        };

        struct PerTileResourcesPool {
            const char* plugin_id_;
            const nvimgcdcsFrameworkDesc_t* framework_;
            nvjpeg2kHandle_t handle_ = nullptr;

            std::vector<PerTileResources> res_;
            std::queue<PerTileResources*> free_;
            std::mutex mtx_;
            std::condition_variable cv_;

            PerTileResourcesPool(const char* id, const nvimgcdcsFrameworkDesc_t* framework, nvjpeg2kHandle_t handle, int num_parallel_tiles)
                : plugin_id_(id)
                , framework_(framework)
                , handle_(handle)
                , res_(num_parallel_tiles) {
                for (auto& tile_res : res_) {
                    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&tile_res.stream_, cudaStreamNonBlocking));
                    XM_CHECK_CUDA(cudaEventCreate(&tile_res.event_));
                    XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle, &tile_res.state_));
                    free_.push(&tile_res);
                }
            }

            ~PerTileResourcesPool() {
                for (auto& tile_res : res_) {
                    if (tile_res.event_) {
                        XM_CUDA_LOG_DESTROY(cudaEventDestroy(tile_res.event_));
                    }
                    if (tile_res.stream_) {
                        XM_CUDA_LOG_DESTROY(cudaStreamDestroy(tile_res.stream_));
                    }
                    if (tile_res.state_) {
                        XM_NVJPEG2K_D_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(tile_res.state_));
                    }
                }
            }

            size_t size() const {
                return res_.size();
            }

            PerTileResources* Acquire() {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { return !free_.empty(); });
                auto res_ptr = free_.front();
                free_.pop();
                return res_ptr;
            }

            void Release(PerTileResources* res_ptr) {
                std::lock_guard<std::mutex> lock(mtx_);
                free_.push(res_ptr);
                cv_.notify_one();
            }
        };

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream;
            nvimgcdcsImageDesc_t* image;
            const nvimgcdcsDecodeParams_t* params;
        };

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        nvjpeg2kHandle_t handle_ = nullptr;
        nvimgcdcsDeviceAllocator_t* device_allocator_;
        nvimgcdcsPinnedAllocator_t* pinned_allocator_;
        int device_id_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
        PerTileResourcesPool per_tile_res_;
    };

    struct Decoder
    {
        Decoder(
            const char* id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params, const char* options);
        ~Decoder();

        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_stream,
            nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decode(int sample_idx, bool immediate);
        nvimgcdcsStatus_t decodeBatch(
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvjpeg2kHandle_t getNvjpeg2kHandle();

        void parseOptions(const char* options);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        const char* plugin_id_;
        nvjpeg2kHandle_t handle_;
        nvjpeg2kDeviceAllocatorV2_t device_allocator_;
        nvjpeg2kPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        const nvimgcdcsExecutionParams_t* exec_params_;
        int num_parallel_tiles_;

        struct CanDecodeCtx {
            Decoder *this_ptr;
            nvimgcdcsProcessingStatus_t* status;
            nvimgcdcsCodeStreamDesc_t** code_streams;
            nvimgcdcsImageDesc_t** images;
            const nvimgcdcsDecodeParams_t* params;
            int num_samples;
            int num_blocks;
            std::vector<std::promise<void>> promise;
        };
    };

    nvimgcdcsStatus_t create(
        nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg2k_decoder";
    nvimgcdcsDecoderDesc_t decoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvjpeg2k
