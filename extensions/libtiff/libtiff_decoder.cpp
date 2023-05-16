#include "libtiff_decoder.h"
#include <tiffio.h>
#include <cstring>
#include <nvtx3/nvtx3.hpp>
#include "convert.h"
#include "log.h"
#include "nvimgcodecs.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                                                                     \
    do {                                                                                                       \
        int retcode = (call);                                                                                  \
        if (LIBTIFF_CALL_SUCCESS != retcode)                                                                   \
            throw std::runtime_error("libtiff call failed with code " + std::to_string(retcode) + ": " #call); \
    } while (0)

namespace libtiff {

class DecoderHelper
{
  public:
    explicit DecoderHelper(nvimgcdcsIoStreamDesc_t io_stream)
        : io_stream_(io_stream)
    {}

    static tmsize_t read(thandle_t handle, void* buffer, tmsize_t n)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t read_nbytes = 0;
        if (helper->io_stream_->read(helper->io_stream_->instance, &read_nbytes, buffer, n) != NVIMGCDCS_STATUS_SUCCESS)
            return 0;
        else
            return read_nbytes;
    }

    static tmsize_t write(thandle_t, void*, tmsize_t)
    {
        // Not used for decoding.
        return 0;
    }

    static toff_t seek(thandle_t handle, toff_t offset, int whence)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        if (helper->io_stream_->seek(helper->io_stream_->instance, offset, whence) != NVIMGCDCS_STATUS_SUCCESS)
            return -1;
        size_t curr_offset = 0;
        if (helper->io_stream_->tell(helper->io_stream_->instance, &curr_offset) != NVIMGCDCS_STATUS_SUCCESS)
            return -1;
        return curr_offset;
    }

    static int map(thandle_t handle, void** base, toff_t* size)
    {
        // This function will be used by LibTIFF only if input is InputKind::HostMemory.
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        const void* raw_data = nullptr;
        size_t data_size = 0;
        if (helper->io_stream_->raw_data(helper->io_stream_->instance, &raw_data) != NVIMGCDCS_STATUS_SUCCESS)
            return -1;
        if (raw_data == nullptr)
            return -1;
        if (helper->io_stream_->size(helper->io_stream_->instance, &data_size) != NVIMGCDCS_STATUS_SUCCESS)
            return -1;
        *base = const_cast<void*>(raw_data);
        *size = data_size;
        return 0;
    }

    static toff_t size(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        size_t data_size = 0;
        if (helper->io_stream_->size(helper->io_stream_->instance, &data_size) != NVIMGCDCS_STATUS_SUCCESS)
            return 0;
        return data_size;
    }

    static int close(thandle_t handle)
    {
        DecoderHelper* helper = reinterpret_cast<DecoderHelper*>(handle);
        delete helper;
        return 0;
    }

  private:
    nvimgcdcsIoStreamDesc_t io_stream_;
};

std::unique_ptr<TIFF, void (*)(TIFF*)> OpenTiff(nvimgcdcsIoStreamDesc_t io_stream)
{
    TIFF* tiffptr;
    TIFFMapFileProc mapproc = nullptr;
    const void* raw_data = nullptr;
    if (io_stream->raw_data(io_stream->instance, &raw_data) == NVIMGCDCS_STATUS_SUCCESS && raw_data != nullptr)
        mapproc = &DecoderHelper::map;

    DecoderHelper* helper = new DecoderHelper(io_stream);
    tiffptr = TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(helper), &DecoderHelper::read, &DecoderHelper::write,
        &DecoderHelper::seek, &DecoderHelper::close, &DecoderHelper::size, mapproc,
        /* unmap */ 0);
    if (!tiffptr)
        delete helper;
    if (tiffptr == nullptr)
        std::runtime_error("Unable to open TIFF image");
    return {tiffptr, &TIFFClose};
}

struct TiffInfo
{
    uint32_t image_width, image_height;
    uint16_t channels;

    uint32_t rows_per_strip;
    uint16_t bit_depth;
    uint16_t orientation;
    uint16_t compression;
    uint16_t photometric_interpretation;
    uint16_t fill_order;

    bool is_tiled;
    bool is_palette;
    bool is_planar;
    struct
    {
        uint16_t *red, *green, *blue;
    } palette;

    uint32_t tile_width, tile_height;
};

TiffInfo GetTiffInfo(TIFF* tiffptr)
{
    TiffInfo info = {};

    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGEWIDTH, &info.image_width));
    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGELENGTH, &info.image_height));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_SAMPLESPERPIXEL, &info.channels));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_BITSPERSAMPLE, &info.bit_depth));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ORIENTATION, &info.orientation));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_COMPRESSION, &info.compression));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ROWSPERSTRIP, &info.rows_per_strip));
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_FILLORDER, &info.fill_order));

    info.is_tiled = TIFFIsTiled(tiffptr);
    if (info.is_tiled) {
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILEWIDTH, &info.tile_width));
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILELENGTH, &info.tile_height));
    } else {
        // We will be reading data line-by-line and pretend that lines are tiles
        info.tile_width = info.image_width;
        info.tile_height = 1;
    }

    if (TIFFGetField(tiffptr, TIFFTAG_PHOTOMETRIC, &info.photometric_interpretation)) {
        info.is_palette = (info.photometric_interpretation == PHOTOMETRIC_PALETTE);
    } else {
        info.photometric_interpretation = PHOTOMETRIC_MINISBLACK;
    }

    uint16_t planar_config;
    LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_PLANARCONFIG, &planar_config));
    info.is_planar = (planar_config == PLANARCONFIG_SEPARATE);

    if (info.is_palette) {
        LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_COLORMAP, &info.palette.red, &info.palette.green, &info.palette.blue));
        info.channels = 3; // Palette is always RGB
    }

    return info;
}

template <int depth>
struct depth2type;

template <>
struct depth2type<8>
{
    using type = uint8_t;
};

template <>
struct depth2type<16>
{
    using type = uint16_t;
};
template <>
struct depth2type<32>
{
    using type = uint32_t;
};

/**
 * @brief Unpacks packed bits and/or converts palette data to RGB.
 *
 * @tparam OutputType Required output type
 * @tparam normalize If true, values will be upscaled to OutputType's dynamic range
 * @param nbits Number of bits per value
 * @param out Output array
 * @param in Pointer to the bits to unpack
 * @param n Number of input values to unpack
 */
template <typename OutputType, bool normalize = true>
void TiffConvert(const TiffInfo& info, OutputType* out, const void* in, size_t n)
{
    // We don't care about endianness here, because we read byte-by-byte and:
    // 1) "The library attempts to hide bit- and byte-ordering differences between the image and the
    //    native machine by converting data to the native machine order."
    //    http://www.libtiff.org/man/TIFFReadScanline.3t.html
    // 2) We only support FILL_ORDER=1 (i.e. big endian), which is TIFF's default and the only fill
    //    order required in Baseline TIFF readers.
    //    https://www.awaresystems.be/imaging/tiff/tifftags/fillorder.html

    size_t nbits = info.bit_depth;
    size_t out_type_bits = 8 * sizeof(OutputType);
    if (out_type_bits < nbits)
        throw std::logic_error("Unpacking bits failed: OutputType too small");
    if (n == 0)
        return;

    auto in_ptr = static_cast<const uint8_t*>(in);
    uint8_t buffer = *(in_ptr++);
    constexpr size_t buffer_capacity = 8 * sizeof(buffer);
    size_t bits_in_buffer = buffer_capacity;

    for (size_t i = 0; i < n; i++) {
        OutputType result = 0;
        size_t bits_to_read = nbits;
        while (bits_to_read > 0) {
            if (bits_in_buffer >= bits_to_read) {
                // If we have enough bits in the buffer, we store them and finish
                result <<= bits_to_read;
                result |= buffer >> (buffer_capacity - bits_to_read);
                bits_in_buffer -= bits_to_read;
                buffer <<= bits_to_read;
                bits_to_read = 0;
            } else {
                // If we don't have enough bits, we store what we have and refill the buffer
                result <<= bits_in_buffer;
                result |= buffer >> (buffer_capacity - bits_in_buffer);
                bits_to_read -= bits_in_buffer;
                buffer = *(in_ptr++);
                bits_in_buffer = buffer_capacity;
            }
        }
        if (info.is_palette) {
            using nvimgcdcs::ConvertNorm;
            out[3 * i + 0] = ConvertNorm<OutputType>(info.palette.red[result]);
            out[3 * i + 1] = ConvertNorm<OutputType>(info.palette.green[result]);
            out[3 * i + 2] = ConvertNorm<OutputType>(info.palette.blue[result]);
        } else {
            if (normalize) {
                double coeff = static_cast<double>((1ull << out_type_bits) - 1) / ((1ull << nbits) - 1);
                result *= coeff;
            }
            out[i] = result;
        }
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

LibtiffDecoderPlugin::LibtiffDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,              // instance
          "libtiff_decoder", // id
          0x00000100,        // version
          "tiff",            // codec_type
          static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_capabilities, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_HOST_OUTPUT}
    , framework_(framework)
{}

nvimgcdcsDecoderDesc_t LibtiffDecoderPlugin::getDecoderDesc()
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

        if (strcmp(codec_name, "tiff") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }

        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].use_cpu) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);

        switch (image_info.sample_format) {
        case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        default:
            break; // supported
        }

        if (image_info.num_planes == 1) {
            if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB)
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        } else if (image_info.num_planes > 1) {
            if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED)
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }
        if (image_info.num_planes != 1 && image_info.num_planes != 3 &&
            image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED)
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;

        if (image_info.plane_info[0].num_channels == 1) {
            if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB)
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        } else if (image_info.plane_info[0].num_channels > 1) {
            if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB ||
                image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED)
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }
        if (image_info.plane_info[0].num_channels != 1 && image_info.plane_info[0].num_channels != 3 &&
            image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED)
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;

        // This codec doesn't apply EXIF orientation
        if (params->enable_orientation &&
            (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libtiff_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if libtiff can decode - " << e.what());
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

nvimgcdcsStatus_t LibtiffDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new DecoderImpl(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t LibtiffDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libtiff_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<LibtiffDecoderPlugin*>(instance);
        handle->create(decoder, device_id);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create libtiff decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    try {
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy libtiff decoder");
    }
}

nvimgcdcsStatus_t DecoderImpl::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libtiff_destroy");
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy libtiff decoder - " << e.what());
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
        NVIMGCDCS_D_LOG_TRACE("libtiff_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve libtiff decoder capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

template <typename Output, typename Input>
nvimgcdcsStatus_t decodeImplTyped2(nvimgcdcsImageInfo_t& image_info, TIFF* tiff, const TiffInfo& info)
{
    if (info.photometric_interpretation != PHOTOMETRIC_RGB && info.photometric_interpretation != PHOTOMETRIC_MINISBLACK &&
        info.photometric_interpretation != PHOTOMETRIC_PALETTE) {
        NVIMGCDCS_D_LOG_ERROR("Unsupported photometric interpretation: " << info.photometric_interpretation);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_planar) {
        NVIMGCDCS_D_LOG_ERROR("Planar TIFFs are not supported");
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.bit_depth > 32) {
        NVIMGCDCS_D_LOG_ERROR("Unsupported bit depth: " << info.bit_depth);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_tiled && (info.tile_width % 16 != 0 || info.tile_height % 16 != 0)) {
        // http://www.libtiff.org/libtiff.html
        // (...) tile width and length must each be a multiple of 16 pixels
        NVIMGCDCS_D_LOG_ERROR("TIFF tile dimensions must be a multiple of 16");
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    if (info.is_tiled && (info.bit_depth != 8 && info.bit_depth != 16 && info.bit_depth != 32)) {
        NVIMGCDCS_D_LOG_ERROR("Unsupported bit depth in tiled TIFF: " << info.bit_depth);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    // Other fill orders are rare and discouraged by TIFF specification, but can happen
    if (info.fill_order != FILLORDER_MSB2LSB) {
        NVIMGCDCS_D_LOG_ERROR("Only FILL_ORDER=1 is supported");
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    size_t buf_nbytes;
    if (!info.is_tiled) {
        buf_nbytes = TIFFScanlineSize(tiff);
    } else {
        buf_nbytes = TIFFTileSize(tiff);
    }

    std::unique_ptr<void, void (*)(void*)> buf{_TIFFmalloc(buf_nbytes), _TIFFfree};
    if (buf.get() == nullptr)
        throw std::runtime_error("Could not allocate memory");

    int num_channels;
    bool planar;
    switch (image_info.sample_format) {
    case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        num_channels = 3;
        planar = false;
        break;
    case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        num_channels = 3;
        planar = true;
        break;
    case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        num_channels = info.channels;
        planar = false;
        break;
    case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        num_channels = info.channels;
        planar = true;
        break;
    case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        num_channels = 1;
        planar = true;
        break;
    default:
        NVIMGCDCS_D_LOG_ERROR("Unsupported sample_format: " << image_info.sample_format);
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (image_info.num_planes > info.channels) {
        NVIMGCDCS_D_LOG_ERROR("Invalid number of planes: " << image_info.num_planes);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    int64_t region_start_y = image_info.region.ndim == 2 ? image_info.region.start[0] : 0;
    int64_t region_start_x = image_info.region.ndim == 2 ? image_info.region.start[1] : 0;
    int64_t region_end_y = image_info.region.ndim == 2 ? image_info.region.end[0] : image_info.plane_info[0].height;
    int64_t region_end_x = image_info.region.ndim == 2 ? image_info.region.end[1] : image_info.plane_info[0].width;
    int64_t region_size_x = region_end_x - region_start_x;
    int64_t stride_y = planar ? region_size_x : region_size_x * num_channels;
    int64_t stride_x = planar ? 1 : num_channels;
    int64_t tile_stride_y = info.tile_width * info.channels;
    int64_t tile_stride_x = info.channels;
    
    const bool allow_random_row_access = (info.compression == COMPRESSION_NONE || info.rows_per_strip == 1);
    // If random access is not allowed, need to read sequentially all previous rows
    // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
    // Compression algorithm does not support random access. Data was requested in a non-sequential
    // order from a file that uses a compression algorithm and that has RowsPerStrip greater than
    // one. That is, data in the image is stored in a compressed form, and with multiple rows packed
    // into a strip. In this case, the library does not support random access to the data. The data
    // should either be accessed sequentially, or the file should be converted so that each strip is
    // made up of one row of data.
    if (!info.is_tiled && !allow_random_row_access) {
        // Need to read sequentially since not all the images support random access
        // If random access is not allowed, need to read sequentially all previous rows
        for (int64_t y = 0; y < region_start_y; y++) {
            LIBTIFF_CALL(TIFFReadScanline(tiff, buf.get(), y, 0));
        }
    }

    bool convert_needed = info.bit_depth != (sizeof(Input) * 8) || info.is_palette;
    Input* in;
    std::vector<uint8_t> scratch;
    if (!convert_needed) {
        in = static_cast<Input*>(buf.get());
    } else {
        scratch.resize(info.tile_height * info.tile_width * info.channels * sizeof(Input));
        in = reinterpret_cast<Input*>(scratch.data());
    }

    Output* img_out = reinterpret_cast<Output*>(image_info.buffer);

    // For non-tiled TIFFs first_tile_x is always 0, because the scanline spans the whole image.
    int64_t first_tile_y = region_start_y - region_start_y % info.tile_height;
    int64_t first_tile_x = region_start_x - region_start_x % info.tile_width;

    for (int64_t tile_y = first_tile_y; tile_y < region_end_y; tile_y += info.tile_height) {
        for (int64_t tile_x = first_tile_x; tile_x < region_end_x; tile_x += info.tile_width) {
            int64_t tile_begin_y = std::max(tile_y, region_start_y);
            int64_t tile_begin_x = std::max(tile_x, region_start_x);
            int64_t tile_end_y = std::min(tile_y + info.tile_height, region_end_y);
            int64_t tile_end_x = std::min(tile_x + info.tile_width, region_end_x);
            int64_t tile_size_y = tile_end_y - tile_begin_y;
            int64_t tile_size_x = tile_end_x - tile_begin_x;

            if (info.is_tiled) {
                auto ret = TIFFReadTile(tiff, buf.get(), tile_x, tile_y, 0, 0);
                if (ret <= 0) {
                    throw std::runtime_error("TIFFReadTile failed");
                }
            } else {
                LIBTIFF_CALL(TIFFReadScanline(tiff, buf.get(), tile_y, 0));
            }

            if (convert_needed) {
                size_t input_values = info.tile_height * info.tile_width * info.channels;
                if (info.is_palette)
                    input_values /= info.channels;
                TiffConvert(info, in, buf.get(), input_values);
            }

            Output* dst = img_out + (tile_begin_y - region_start_y) * stride_y + (tile_begin_x - region_start_x) * stride_x;
            const Input* src = in + (tile_begin_y - tile_y) * tile_stride_y + (tile_begin_x - tile_x) * tile_stride_x;

            switch (image_info.sample_format) {
            case NVIMGCDCS_SAMPLEFORMAT_P_Y:
                switch (info.channels) {
                case 1:
                {
                    auto* plane = dst;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = *(tile_row + j * tile_stride_x);
                        }
                    }
                }
                break;

                case 3:
                {
                    uint32_t plane_stride = image_info.plane_info[0].height * image_info.plane_info[0].row_stride;
                    for (uint32_t c = 0; c < image_info.num_planes; c++) {
                        auto* plane = dst + c * plane_stride;
                        for (uint32_t i = 0; i < tile_size_y; i++) {
                            auto* row = plane + i * stride_y;
                            auto* tile_row = src + i * tile_stride_y;
                            for (uint32_t j = 0; j < tile_size_x; j++) {
                                auto* pixel = tile_row + j * tile_stride_x;
                                auto* out_pixel = row + j * stride_x;
                                auto r = *(pixel + 0);
                                auto g = *(pixel + 1);
                                auto b = *(pixel + 2);
                                *(out_pixel) = 0.299f * r + 0.587f * g + 0.114f * b;
                            }
                        }
                    }
                }
                break;

                default:
                    NVIMGCDCS_D_LOG_ERROR("Unexpected number of channels for conversion to grayscale: " << info.channels);
                    return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
                }
                break;
            case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
            case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
            case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
            {   
                uint32_t plane_stride = image_info.plane_info[0].height * image_info.plane_info[0].row_stride;
                for (uint32_t c = 0; c < image_info.num_planes; c++) {
                    uint32_t dst_p = c;
                    if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR)
                        dst_p = c == 2 ? 0 : c == 0 ? 2 : c;
                    auto* plane = dst + dst_p * plane_stride;
                    for (uint32_t i = 0; i < tile_size_y; i++) {
                        auto* row = plane + i * stride_y;
                        auto* tile_row = src + i * tile_stride_y;
                        for (uint32_t j = 0; j < tile_size_x; j++) {
                            *(row + j * stride_x) = *(tile_row + j * tile_stride_x + c);
                        }
                    }
                }
            }
            break;

            case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
            case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
            case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED: 
            {
                for (uint32_t i = 0; i < tile_size_y; i++) {
                    auto* row = dst + i * stride_y;
                    auto* tile_row = src + i * tile_stride_y;
                    for (uint32_t j = 0; j < tile_size_x; j++) {
                        auto *pixel = row + j * stride_x;
                        auto *tile_pixel = tile_row + j * tile_stride_x;
                        for (uint32_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                            uint32_t out_c = c;
                            if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR)
                                out_c = c == 2 ? 0 : c == 0 ? 2 : c;
                            *(pixel + out_c) = *(tile_pixel + c);
                        }
                    }
                }
            }
            break;
            
            case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
            default:
                NVIMGCDCS_D_LOG_ERROR("Unsupported sample_format: " << image_info.sample_format);
                return NVIMGCDCS_STATUS_INVALID_PARAMETER;
            }
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

template <typename Output>
nvimgcdcsStatus_t decodeImplTyped(nvimgcdcsImageInfo_t& image_info, TIFF* tiff, const TiffInfo& info)
{
    if (info.bit_depth <= 8) {
        return decodeImplTyped2<Output, uint8_t>(image_info, tiff, info);
    } else if (info.bit_depth <= 16) {
        return decodeImplTyped2<Output, uint16_t>(image_info, tiff, info);
    } else if (info.bit_depth <= 32) {
        return decodeImplTyped2<Output, uint32_t>(image_info, tiff, info);
    } else {
        NVIMGCDCS_D_LOG_ERROR("Unsupported bit depth: " << info.bit_depth);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }
}

nvimgcdcsStatus_t decodeImpl(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params, std::vector<uint8_t>& buffer)
{
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    auto ret = image->getImageInfo(image->instance, &image_info);
    if (ret != NVIMGCDCS_STATUS_SUCCESS)
        return ret;

    if (image_info.region.ndim != 0 && image_info.region.ndim != 2) {
        NVIMGCDCS_D_LOG_ERROR("Invalid region of interest");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (image_info.buffer_kind != NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        NVIMGCDCS_D_LOG_ERROR("Unexpected buffer kind");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    auto io_stream = code_stream->io_stream;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    auto tiff = OpenTiff(io_stream);
    auto info = GetTiffInfo(tiff.get());
    switch(image_info.plane_info[0].sample_type) {
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
            return decodeImplTyped<uint8_t>(image_info, tiff.get(), info);
        case NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8:
            return decodeImplTyped<uint8_t>(image_info, tiff.get(), info);
        case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
            return decodeImplTyped<uint8_t>(image_info, tiff.get(), info);
        case NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16:
            return decodeImplTyped<uint8_t>(image_info, tiff.get(), info);
        case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
            return decodeImplTyped<uint8_t>(image_info, tiff.get(), info);
        default:
            NVIMGCDCS_D_LOG_ERROR("Invalid data type: " << image_info.plane_info[0].sample_type);
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
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
                nvtx3::scoped_range marker{"libtiff decode " + std::to_string(sample_idx)};
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
        NVIMGCDCS_D_LOG_TRACE("libtiff_decode_batch");
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
        NVIMGCDCS_D_LOG_ERROR("Could not decode tiff batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace libtiff