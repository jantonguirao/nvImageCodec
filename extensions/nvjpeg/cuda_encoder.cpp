#include "cuda_encoder.h"
#include <nvimgcodecs.h>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nvjpeg.h>

#include "errors_handling.h"
#include "type_convert.h"
#include "log.h"


namespace nvjpeg {

NvJpegCudaEncoderPlugin::NvJpegCudaEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : encoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC, NULL,
          this,             // instance
          "nvjpeg_encoder", // id
          0x00000100,       // version
          "jpeg",           // codec_type
          static_create, Encoder::static_destroy, Encoder::static_get_capabilities, Encoder::static_can_encode,
          Encoder::static_encode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_INPUT}
    , framework_(framework)
{
}

nvimgcdcsEncoderDesc_t NvJpegCudaEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "jpeg") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].use_gpu) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_can_encode");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not check if nvjpge can encode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaEncoderPlugin::Encoder::Encoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, const nvimgcdcsEncodeParams_t* params)
    : capabilities_(capabilities)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
{
    if (framework->device_allocator && framework->device_allocator->device_malloc && framework->device_allocator->device_free) {
        device_allocator_.dev_ctx = framework->device_allocator->device_ctx;
        device_allocator_.dev_malloc = framework->device_allocator->device_malloc;
        device_allocator_.dev_free = framework->device_allocator->device_free;
    }

    if (framework->pinned_allocator && framework->pinned_allocator->pinned_malloc && framework->pinned_allocator->pinned_free) {
        pinned_allocator_.pinned_ctx = framework->pinned_allocator->pinned_ctx;
        pinned_allocator_.pinned_malloc = framework->pinned_allocator->pinned_malloc;
        pinned_allocator_.pinned_free = framework->pinned_allocator->pinned_free;
    }

    if (device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, 0, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &handle_));
    }

    if (framework->device_allocator && (framework->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(framework->device_allocator->device_mem_padding, handle_));
    }
    if (framework->pinned_allocator && (framework->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(framework->pinned_allocator->pinned_mem_padding, handle_));
    }

    //TODO create resources per thread
    //nvimgcdcsExecutorDesc_t executor;
    //framework_->getExecutor(framework_->instance, &executor);
    //int num_threads = executor->get_num_threads(executor->instance);

    encode_state_batch_ = std::make_unique<NvJpegCudaEncoderPlugin::EncodeState>(this /*,num_threads*/);
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::create(nvimgcdcsEncoder_t* encoder, const nvimgcdcsEncodeParams_t* params)
{
    *encoder = reinterpret_cast<nvimgcdcsEncoder_t>(new NvJpegCudaEncoderPlugin::Encoder(capabilities_, framework_, params));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::static_create(void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_create_encoder");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(params);
        NvJpegCudaEncoderPlugin* handle = reinterpret_cast<NvJpegCudaEncoderPlugin*>(instance);
        handle->create(encoder, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not create nvjpeg encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegCudaEncoderPlugin::Encoder::~Encoder()
{
    try {
        encode_state_batch_.reset();
        XM_CHECK_NVJPEG(nvjpegDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg encoder - " << e.what());
    }
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_destroy(nvimgcdcsEncoder_t encoder)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_destroy_encoder");
        XM_CHECK_NULL(encoder);
        NvJpegCudaEncoderPlugin::Encoder* handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    if (capabilities) {
        *capabilities = capabilities_.data();
    }

    if (size) {
        *size = capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_get_capabilities(
    nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg_get_capabilities");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        NvJpegCudaEncoderPlugin::Encoder* handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not retrive nvjpeg encoder capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaEncoderPlugin::EncodeState::EncodeState(NvJpegCudaEncoderPlugin::Encoder* encoder)
    : encoder_(encoder)
{
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));
    XM_CHECK_NVJPEG(nvjpegEncoderStateCreate(encoder_->handle_, &handle_, stream_));
}

NvJpegCudaEncoderPlugin::EncodeState::~EncodeState()
{
    try {
        if (event_) {
            XM_CHECK_CUDA(cudaEventDestroy(event_));
        }

        if (stream_) {
            XM_CHECK_CUDA(cudaStreamDestroy(stream_));
        }

        if (handle_) {
            XM_CHECK_NVJPEG(nvjpegEncoderStateDestroy(handle_));
        }

    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not destroy encode state - " << e.what());
    }
}

NvJpegCudaEncoderPlugin::EncodeState* NvJpegCudaEncoderPlugin::EncodeState::getSampleEncodeState(int sample_idx)
{
    if (static_cast<size_t>(sample_idx) == per_sample_encode_state_.size()) {
        per_sample_encode_state_.emplace_back(std::make_unique<NvJpegCudaEncoderPlugin::EncodeState>(encoder_));
    }

    return per_sample_encode_state_[sample_idx].get();
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::encode(NvJpegCudaEncoderPlugin::EncodeState* encode_state, nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params)
{

    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    image->getImageInfo(image->instance, &image_info);

    nvimgcdcsImageInfo_t out_image_info;
    code_stream->getImageInfo(code_stream->instance, &out_image_info);

    if (image_info.plane_info[0].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED);
        NVIMGCDCS_E_LOG_ERROR("Unsupported sample data type. Only UINT8 is supported.");
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

    nvjpegEncoderParams_t encode_params_;
    XM_CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &encode_params_, encode_state->stream_));
    std::unique_ptr<std::remove_pointer<nvjpegEncoderParams_t>::type, decltype(&nvjpegEncoderParamsDestroy)> encode_params(
        encode_params_, &nvjpegEncoderParamsDestroy);
    int nvjpeg_format = nvimgcdcs_to_nvjpeg_format(image_info.sample_format);
    if (nvjpeg_format < 0) {
        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    nvjpegInputFormat_t input_format = static_cast<nvjpegInputFormat_t>(nvjpeg_format);

    nvjpegImage_t input_image;
    unsigned char* ptr = device_buffer;
    for (uint32_t p = 0; p < image_info.num_planes; ++p) {
        input_image.channel[p] = ptr;
        input_image.pitch[p] = image_info.plane_info[p].row_stride;
        ptr += input_image.pitch[p] * image_info.plane_info[p].height;
    }

    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params.get(), static_cast<int>(params->quality), encode_state->stream_));
    NVIMGCDCS_E_LOG_DEBUG(" - quality: " << static_cast<int>(params->quality));

    nvimgcdcsJpegEncodeParams_t* jpeg_encode_params = static_cast<nvimgcdcsJpegEncodeParams_t*>(params->next);
    while (jpeg_encode_params && jpeg_encode_params->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS)
        jpeg_encode_params = static_cast<nvimgcdcsJpegEncodeParams_t*>(jpeg_encode_params->next);
    if (jpeg_encode_params) {
        nvjpegJpegEncoding_t encoding = nvimgcdcs_to_nvjpeg_encoding(jpeg_encode_params->encoding);
        NVIMGCDCS_E_LOG_DEBUG(" - encoding: " << encoding);
        XM_CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params.get(), encoding, encode_state->stream_));
        NVIMGCDCS_E_LOG_DEBUG(" - optimized huffman: " << jpeg_encode_params->optimized_huffman);
        if (jpeg_encode_params->optimized_huffman) {
            XM_CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params.get(), 1, NULL));
        }
    }
    nvjpegChromaSubsampling_t chroma_subsampling = nvimgcdcs_to_nvjpeg_css(out_image_info.chroma_subsampling);
    if (chroma_subsampling != NVJPEG_CSS_UNKNOWN) {
        XM_CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params.get(), chroma_subsampling, NULL));
    }
    if (image_info.color_spec == NVIMGCDCS_COLORSPEC_SYCC || image_info.color_spec == NVIMGCDCS_COLORSPEC_YCCK) {
        XM_CHECK_NVJPEG(nvjpegEncodeYUV(handle_, encode_state->handle_, encode_params.get(), &input_image, chroma_subsampling,
            image_info.plane_info[0].width, image_info.plane_info[0].height, encode_state->stream_));
    } else {
        XM_CHECK_NVJPEG(nvjpegEncodeImage(handle_, encode_state->handle_, encode_params.get(), &input_image, input_format,
            image_info.plane_info[0].width, image_info.plane_info[0].height, encode_state->stream_));
    }

    XM_CHECK_CUDA(cudaEventRecord(encode_state->event_, encode_state->stream_));

    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    encode_state->image_ = image;
    encode_state->code_stream_ = code_stream;
    int cuda_device_id = params->backends ? params->backends->cuda_device_id : 0;
    int sample_idx = 0; // doesn't matter now
    executor->launch(
        executor->instance, cuda_device_id, sample_idx, encode_state, [](int thread_id, int sample_idx, void* task_context) -> void {
            auto encode_state = reinterpret_cast<NvJpegCudaEncoderPlugin::EncodeState*>(task_context);
            XM_CHECK_CUDA(cudaEventSynchronize(encode_state->event_));

            size_t length;
            XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
                encode_state->encoder_->handle_, encode_state->handle_, NULL, &length, encode_state->stream_));

            encode_state->compressed_data_.resize(length);

            XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(encode_state->encoder_->handle_, encode_state->handle_,
                encode_state->compressed_data_.data(), &length, encode_state->stream_));

            nvimgcdcsIoStreamDesc_t io_stream = encode_state->code_stream_->io_stream;
            size_t output_size;
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            io_stream->write(io_stream->instance, &output_size, static_cast<void*>(&encode_state->compressed_data_[0]),
                encode_state->compressed_data_.size());

            encode_state->image_->imageReady(encode_state->image_->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
        });
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg_encode_batch");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_E_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);

        nvimgcdcsStatus_t result = NVIMGCDCS_STATUS_SUCCESS;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            auto enc_state = handle->encode_state_batch_->getSampleEncodeState(sample_idx);
            result = handle->encode(enc_state, images[sample_idx], code_streams[sample_idx], params);
            if (result != NVIMGCDCS_STATUS_SUCCESS) {
                return result;
            }
        }
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not encode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg
