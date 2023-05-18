#include "opencv_decoder.h"
#include <cstring>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "convert.h"
#include "log.h"
#include "nvimgcodecs.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

namespace opencv {



static void color_convert(cv::Mat& img, cv::ColorConversionCodes conversion)
{
    NVIMGCDCS_D_LOG_TRACE("Before cvtColor - " << img.rows << " x " << img.cols);
    if (img.data == nullptr || img.rows == 0 || img.cols == 0)
        throw std::runtime_error("Invalid input image");
    cv::cvtColor(img, img, conversion);
    NVIMGCDCS_D_LOG_TRACE("After cvtColor");
}

template <typename DestType, typename SrcType>
nvimgcdcsStatus_t ConvertPlanar(DestType* destinationBuffer, uint32_t plane_stride, uint32_t row_stride_bytes, const cv::Mat& image)
{
    using nvimgcdcs::ConvertSatNorm;
    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    size_t height = image.size[0];
    size_t width = image.size[1];
    for (size_t ch = 0; ch < planes.size(); ++ch) {
        const cv::Mat& srcPlane = planes[ch];
        const SrcType* srcPlanePtr = srcPlane.ptr<SrcType>();
        DestType* destPlanePtr = destinationBuffer + ch * plane_stride;
        for (size_t i = 0; i < height; ++i) {
            const SrcType* srcRow = srcPlanePtr + i * width;
            DestType* destRow = reinterpret_cast<DestType*>(
                reinterpret_cast<uint8_t*>(destPlanePtr) + i * row_stride_bytes);
            for (size_t j = 0; j < width; ++j) {
                destRow[j] = ConvertSatNorm<DestType>(srcRow[j]);
            }
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

template <typename DestType, typename SrcType>
nvimgcdcsStatus_t ConvertInterleaved(DestType* destinationBuffer, uint32_t row_stride_bytes, const cv::Mat& image)
{
    using nvimgcdcs::ConvertSatNorm;
    size_t height = image.size[0];
    size_t width = image.size[1];
    size_t channels = image.channels();
    for (size_t i = 0; i < height; ++i) {
        const SrcType* srcRow = image.ptr<SrcType>() + i * width * channels;
        DestType* destRow = reinterpret_cast<DestType*>(
            reinterpret_cast<uint8_t*>(destinationBuffer) + i * row_stride_bytes);
        for (size_t j = 0; j < width; ++j) {
            for (size_t c = 0; c < channels; c++) {
                destRow[j * channels + c] = ConvertSatNorm<DestType>(srcRow[j * channels + c]);
            }
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t ConvertPlanar(nvimgcdcsImageInfo_t& info, const cv::Mat& decoded)
{

#define CaseConvertPlanar(OUT_SAMPLE_TYPE, OutType, img_info, image)                                                          \
    case OUT_SAMPLE_TYPE:                                                                                                     \
        switch (image.depth()) {                                                                                              \
        case CV_8U:                                                                                                           \
            return ConvertPlanar<OutType, uint8_t>(reinterpret_cast<OutType*>(img_info.buffer),                               \
                img_info.plane_info[0].row_stride * img_info.plane_info[0].height, img_info.plane_info[0].row_stride, image); \
        case CV_16U:                                                                                                          \
            return ConvertPlanar<OutType, uint16_t>(reinterpret_cast<OutType*>(img_info.buffer),                              \
                img_info.plane_info[0].row_stride * img_info.plane_info[0].height, img_info.plane_info[0].row_stride, image); \
        default:                                                                                                              \
            return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;                                                               \
        }                                                                                                                     \
        break;

    switch (info.plane_info[0].sample_type) {
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, uint8_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8, int8_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16, int16_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32, float, info, decoded);
    default:
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
#undef CaseConvertPlanar
}

nvimgcdcsStatus_t ConvertInterleaved(nvimgcdcsImageInfo_t& info, const cv::Mat& decoded)
{

#define CaseConvertInterleaved(OUT_SAMPLE_TYPE, OutType, img_info, image)                               \
    case OUT_SAMPLE_TYPE:                                                                               \
        switch (image.depth()) {                                                                        \
        case CV_8U:                                                                                     \
            return ConvertInterleaved<OutType, uint8_t>(                                                \
                reinterpret_cast<OutType*>(img_info.buffer), img_info.plane_info[0].row_stride, image); \
        case CV_16U:                                                                                    \
            return ConvertInterleaved<OutType, uint16_t>(                                               \
                reinterpret_cast<OutType*>(img_info.buffer), img_info.plane_info[0].row_stride, image); \
        default:                                                                                        \
            return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;                                         \
        }                                                                                               \
        break;

    switch (info.plane_info[0].sample_type) {
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, uint8_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8, int8_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16, int16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32, float, info, decoded);
    default:
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

nvimgcdcsStatus_t Convert(nvimgcdcsImageInfo_t& info, const cv::Mat& decoded)
{
    if (info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB ||
        info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
        info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED) {
        return ConvertPlanar(info, decoded);
    } else {
        return ConvertInterleaved(info, decoded);
    }
}

struct DecodeState
{
    DecodeState(int num_threads)
        : per_thread_(num_threads)
    {}
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcdcsCodeStreamDesc_t code_stream;
        nvimgcdcsImageDesc_t image;
        const nvimgcdcsDecodeParams_t* params;
    };
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;
};

struct DecoderImpl
{
    DecoderImpl(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id);
    ~DecoderImpl();

    nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
    nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images,
        int batch_size, const nvimgcdcsDecodeParams_t* params);
    nvimgcdcsStatus_t decodeBatch(
        nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
    static nvimgcdcsStatus_t static_get_capabilities(nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
    static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
        nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
        nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    const std::vector<nvimgcdcsCapability_t>& capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
    int device_id_;
    std::unique_ptr<DecodeState> decode_state_batch_;
};

OpenCVDecoderPlugin::OpenCVDecoderPlugin(const char* codec_name, const nvimgcdcsFrameworkDesc_t framework)
    : codec_name_(codec_name)
    , decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,             // instance
          "opencv_decoder", // id
          codec_name_,      // codec_type
          static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_capabilities, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_HOST_OUTPUT}
    , framework_(framework)
{}

nvimgcdcsDecoderDesc_t OpenCVDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t DecoderImpl::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "jpeg") != 0 && strcmp(codec_name, "jpeg2k") != 0 && strcmp(codec_name, "png") != 0 &&
            strcmp(codec_name, "tiff") != 0 && strcmp(codec_name, "bmp") != 0 && strcmp(codec_name, "pnm") != 0 &&
            strcmp(codec_name, "webp") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }

        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].kind == NVIMGCDCS_BACKEND_KIND_CPU_ONLY) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }

        nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &info);

        switch (info.sample_format) {
        case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        default:
            break; // supported
        }

        if (info.num_planes > 1) {
            for (size_t p = 0; p < info.num_planes; p++) {
                if (info.plane_info[p].num_channels != 1)
                    *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        } else if (info.num_planes == 1) {
            if (info.plane_info[0].num_channels != 3
                && info.plane_info[0].num_channels != 4
                && info.plane_info[0].num_channels != 1)
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        } else {
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        auto sample_type = info.plane_info[0].sample_type;
        for (size_t p = 1; p < info.num_planes; p++) {
            if (info.plane_info[p].sample_type != sample_type) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
        switch (sample_type) {
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
            break;
        default:
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            break;
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("opencv_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if opencv can decode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

DecoderImpl::DecoderImpl(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id)
    : capabilities_(capabilities)
    , framework_(framework)
    , device_id_(device_id)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(num_threads);
}

nvimgcdcsStatus_t OpenCVDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new DecoderImpl(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t OpenCVDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("opencv_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<OpenCVDecoderPlugin*>(instance);
        handle->create(decoder, device_id, options);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create opencv decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    try {
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy opencv decoder");
    }
}

nvimgcdcsStatus_t DecoderImpl::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("opencv_destroy");
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy opencv decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t DecoderImpl::static_get_capabilities(nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("opencv_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve opencv decoder capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t decodeImpl(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params, std::vector<uint8_t>& buffer)
{
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    auto ret = image->getImageInfo(image->instance, &info);
    if (ret != NVIMGCDCS_STATUS_SUCCESS)
        return ret;

    if (info.region.ndim != 0 && info.region.ndim != 2) {
        NVIMGCDCS_D_LOG_ERROR("Invalid region of interest");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    auto io_stream = code_stream->io_stream;
    size_t encoded_length;
    ret = io_stream->size(io_stream->instance, &encoded_length);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }

    const void* ptr;
    ret = io_stream->raw_data(io_stream->instance, &ptr);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }
    const uint8_t* encoded_data = static_cast<const uint8_t*>(ptr);
    if (!ptr && encoded_length > 0) {
        buffer.resize(encoded_length);
        size_t read_nbytes = 0;
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        ret = io_stream->read(io_stream->instance, &read_nbytes, buffer.data(), buffer.size());
        if (ret != NVIMGCDCS_STATUS_SUCCESS)
            return ret;
        if (read_nbytes != buffer.size()) {
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        encoded_data = buffer.data();
    }

    int num_channels = std::max(info.num_planes, info.plane_info[0].num_channels);
    int flags = num_channels > 1 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
    if (num_channels > 3)
        flags |= cv::IMREAD_UNCHANGED;
    if (!params->enable_orientation)
        flags |= cv::IMREAD_IGNORE_ORIENTATION;
    if (info.plane_info[0].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        flags |= cv::IMREAD_ANYDEPTH;
    auto decoded = cv::imdecode(cv::_InputArray(encoded_data, encoded_length), flags);

    if (decoded.data == nullptr) {
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    } else if (info.buffer_kind != NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (info.region.ndim == 2) {
        int start_y = info.region.start[0];
        int start_x = info.region.start[1];
        int crop_h = info.region.end[0] - info.region.start[0];
        int crop_w = info.region.end[1] - info.region.start[1];
        if (crop_h < 0 || crop_w < 0 || start_x < 0 || start_y < 0 || (start_y + crop_h) > decoded.rows ||
            (start_x + crop_w) > decoded.cols) {
            NVIMGCDCS_D_LOG_ERROR("Region of interest is out of bounds");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        cv::Rect roi(start_x, start_y, crop_w, crop_h);
        cv::Mat tmp;
        decoded(roi).copyTo(tmp);
        std::swap(tmp, decoded);
    }

    switch (info.sample_format) {
    case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
    case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        color_convert(decoded, cv::COLOR_BGR2RGB); // opencv decodes as BGR layout
        break;
    case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
    case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        if (num_channels == 4)
            color_convert(decoded, cv::COLOR_BGRA2RGBA);
        break;
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
    case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        break;
    default:
        NVIMGCDCS_D_LOG_ERROR("Unsupported sample_format: " << info.sample_format);
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return Convert(info, decoded);
}

nvimgcdcsStatus_t DecoderImpl::decodeBatch(
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    decode_state_batch_->samples_.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        decode_state_batch_->samples_[i].code_stream = code_streams[i];
        decode_state_batch_->samples_[i].image = images[i];
        decode_state_batch_->samples_[i].params = params;
    }

    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        executor->launch(executor->instance, NVIMGCDCS_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(),
            [](int tid, int sample_idx, void* context) -> void {
                nvtx3::scoped_range marker{"opencv decode " + std::to_string(sample_idx)};
                auto* decode_state = reinterpret_cast<DecodeState*>(context);
                auto& sample = decode_state->samples_[sample_idx];
                auto& thread_resources = decode_state->per_thread_[tid];
                auto result = decodeImpl(sample.code_stream, sample.image, sample.params, thread_resources.buffer);
                if (result == NVIMGCDCS_STATUS_SUCCESS) {
                    sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
                } else {
                    sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                }
            });
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("opencv_decode_batch");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        nvimgcdcsStatus_t result = handle->decodeBatch(code_streams, images, batch_size, params);
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace opencv
