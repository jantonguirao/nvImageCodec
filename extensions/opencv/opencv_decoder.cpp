#include "opencv_decoder.h"
#include <cstring>
#include <future>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "convert.h"
#include "error_handling.h"
#include "log.h"
#include "nvimgcodecs.h"

namespace opencv {

static void color_convert(cv::Mat& img, cv::ColorConversionCodes conversion)
{
    if (img.data == nullptr || img.rows == 0 || img.cols == 0)
        throw std::runtime_error("Invalid input image");
    cv::cvtColor(img, img, conversion);
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
            DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destPlanePtr) + i * row_stride_bytes);
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
        DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destinationBuffer) + i * row_stride_bytes);
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
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_INT8, int8_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertPlanar(NVIMGCDCS_SAMPLE_DATA_TYPE_INT16, int16_t, info, decoded);
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
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_INT8, int8_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_INT16, int16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32, float, info, decoded);
    default:
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

nvimgcdcsStatus_t Convert(nvimgcdcsImageInfo_t& info, const cv::Mat& decoded)
{
    if (info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB || info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
        info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED) {
        return ConvertPlanar(info, decoded);
    } else {
        return ConvertInterleaved(info, decoded);
    }
}

struct DecodeState
{
    DecodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, int num_threads)
        : plugin_id_(plugin_id)
        , framework_(framework)
        , per_thread_(num_threads)
    {}
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcdcsCodeStreamDesc_t* code_stream;
        nvimgcdcsImageDesc_t* image;
        const nvimgcdcsDecodeParams_t* params;
    };

    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;
};

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params);
    ~DecoderImpl();

    nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image,
        const nvimgcdcsDecodeParams_t* params);
    nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
        nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    static nvimgcdcsStatus_t decode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
        nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params,
        std::vector<uint8_t>& buffer);
    nvimgcdcsStatus_t decodeBatch(
        nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
    static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
        nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
        nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    const nvimgcdcsExecutionParams_t* exec_params_;
    std::unique_ptr<DecodeState> decode_state_batch_;

    struct CanDecodeCtx
    {
        DecoderImpl* this_ptr;
        nvimgcdcsProcessingStatus_t* status;
        nvimgcdcsCodeStreamDesc_t** code_streams;
        nvimgcdcsImageDesc_t** images;
        const nvimgcdcsDecodeParams_t* params;
        int num_samples;
        int num_blocks;
        std::vector<std::promise<void>> promise;
    };
};

OpenCVDecoderPlugin::OpenCVDecoderPlugin(const std::string& codec_name, const nvimgcdcsFrameworkDesc_t* framework)
    : codec_name_(codec_name)
    , plugin_id_("opencv_" + codec_name_ + "_decoder")
    , decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL, this, plugin_id_.c_str(), codec_name_.c_str(),
          NVIMGCDCS_BACKEND_KIND_CPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_batch}
    , framework_(framework)
{}

nvimgcdcsDecoderDesc_t* OpenCVDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t DecoderImpl::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_stream,
    nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "opencv_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);

        *status = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0 && strcmp(cs_image_info.codec_name, "jpeg2k") != 0 &&
            strcmp(cs_image_info.codec_name, "png") != 0 && strcmp(cs_image_info.codec_name, "tiff") != 0 &&
            strcmp(cs_image_info.codec_name, "bmp") != 0 && strcmp(cs_image_info.codec_name, "pnm") != 0 &&
            strcmp(cs_image_info.codec_name, "webp") != 0) {
            *status = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCDCS_STATUS_SUCCESS;
        }

        nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        image->getImageInfo(image->instance, &info);

        switch (info.sample_format) {
        case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
            *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
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
                    *status |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        } else if (info.num_planes == 1) {
            if (info.plane_info[0].num_channels != 3 && info.plane_info[0].num_channels != 4 && info.plane_info[0].num_channels != 1)
                *status |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        } else {
            *status |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        auto sample_type = info.plane_info[0].sample_type;
        for (size_t p = 1; p < info.num_planes; p++) {
            if (info.plane_info[p].sample_type != sample_type) {
                *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
        switch (sample_type) {
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
            break;
        default:
            *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            break;
        }

        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcdcsStatus_t DecoderImpl::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
    nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "opencv_can_decode");
        nvtx3::scoped_range marker{"opencv_can_decode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, batch_size, num_blocks};
            canDecodeCtx.promise.resize(num_threads);
            std::vector<std::future<void>> fut;
            fut.reserve(num_threads);
            for (auto& pr : canDecodeCtx.promise)
                fut.push_back(pr.get_future());
            auto task = [](int tid, int block_idx, void* context) -> void {
                auto* ctx = reinterpret_cast<CanDecodeCtx*>(context);
                int64_t i_start = ctx->num_samples * block_idx / ctx->num_blocks;
                int64_t i_end = ctx->num_samples * (block_idx + 1) / ctx->num_blocks;
                for (int i = i_start; i < i_end; i++) {
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, task);
            }
            task(-1, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

DecoderImpl::DecoderImpl(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    auto executor = exec_params->executor;
    int num_threads = executor->getNumThreads(executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(plugin_id_, framework_, num_threads);
}

nvimgcdcsStatus_t OpenCVDecoderPlugin::create(
    nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_.c_str(), "opencv_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params)
        *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new DecoderImpl(plugin_id_.c_str(), framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_.c_str(), "Could not create opencv decoder - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t OpenCVDecoderPlugin::static_create(
    void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<OpenCVDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "opencv_destroy");
}

nvimgcdcsStatus_t DecoderImpl::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::decode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
    nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params,
    std::vector<uint8_t>& buffer)
{
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    auto ret = image->getImageInfo(image->instance, &info);
    if (ret != NVIMGCDCS_STATUS_SUCCESS)
        return ret;

    if (info.region.ndim != 0 && info.region.ndim != 2) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Invalid region of interest");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    auto io_stream = code_stream->io_stream;
    size_t encoded_length;
    ret = io_stream->size(io_stream->instance, &encoded_length);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }

    void* ptr = nullptr;
    ret = io_stream->map(io_stream->instance, &ptr, 0, encoded_length);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }
    auto auto_unmap = std::shared_ptr<void>(
        ptr, [io_stream, encoded_length](void* addr) { io_stream->unmap(io_stream->instance, addr, encoded_length); });
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
    if (!params->apply_exif_orientation)
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
            NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Region of interest is out of bounds");
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
        else if (num_channels == 3)
            color_convert(decoded, cv::COLOR_BGR2RGB);
        break;
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
    case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        break;
    default:
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Unsupported sample_format: " << info.sample_format);
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return Convert(info, decoded);
}

nvimgcdcsStatus_t DecoderImpl::decodeBatch(
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "opencv_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        decode_state_batch_->samples_.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            decode_state_batch_->samples_[i].code_stream = code_streams[i];
            decode_state_batch_->samples_[i].image = images[i];
            decode_state_batch_->samples_[i].params = params;
        }

        auto executor = exec_params_->executor;
        auto task = [](int tid, int sample_idx, void* context) -> void {
            nvtx3::scoped_range marker{"opencv decode " + std::to_string(sample_idx)};
            auto* decode_state = reinterpret_cast<DecodeState*>(context);
            auto& sample = decode_state->samples_[sample_idx];
            auto& thread_resources = decode_state->per_thread_[tid];
            auto& plugin_id = decode_state->plugin_id_;
            auto& framework = decode_state->framework_;
            auto result = decode(plugin_id, framework, sample.code_stream, sample.image, sample.params, thread_resources.buffer);
            if (result == NVIMGCDCS_STATUS_SUCCESS) {
                sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } else {
                sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
        };
        if (batch_size == 1) {
            task(0, 0, decode_state_batch_.get());
        } else {
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                executor->launch(executor->instance, NVIMGCDCS_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(), task);
            }
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not decode batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcdcsStatus_t DecoderImpl::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
    nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace opencv
