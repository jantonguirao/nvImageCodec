/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVIMGCDCS_HEADER
#define NVIMGCDCS_HEADER

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include "nvimgcdcs_version.h"

#ifndef NVIMGCDCSAPI
    #ifdef _WIN32
        #define NVIMGCDCSAPI __declspec(dllexport)
    #elif __GNUC__ >= 4
        #define NVIMGCDCSAPI __attribute__((visibility("default")))
    #else
        #define NVIMGCDCSAPI
    #endif
#endif

#if defined(__cplusplus)
extern "C"
{
#endif

#define NVIMGCDCS_MAX_CODEC_NAME_SIZE 256

    typedef enum
    {
        NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_DEVICE_ALLOCATOR,
        NVIMGCDCS_STRUCTURE_TYPE_PINNED_ALLOCATOR,
        NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION,
        NVIMGCDCS_STRUCTURE_TYPE_REGION,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_PLANE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODING,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_TILE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_TILE_COMPOENT_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_RESOLUTION_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_DECODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_IO_STREAM_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_FRAMEWORK_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_CODE_STREAM_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA,
        NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_EXECUTOR_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_PROCESSOR_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsStructureType_t;

    typedef int (*nvimgcdcsDeviceMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsDeviceFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsPinnedMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsPinnedFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        nvimgcdcsDeviceMalloc_t device_malloc;
        nvimgcdcsDeviceFree_t device_free;
        void* device_ctx;
        size_t device_mem_padding; // any device memory allocation
                                   // would be padded to the multiple of specified number of bytes
    } nvimgcdcsDeviceAllocator_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        nvimgcdcsPinnedMalloc_t pinned_malloc;
        nvimgcdcsPinnedFree_t pinned_free;
        void* pinned_ctx;
        size_t pinned_mem_padding; // any pinned host memory allocation
                                   // would be padded to the multiple of specified number of bytes
    } nvimgcdcsPinnedAllocator_t;

    typedef enum
    {
        NVIMGCDCS_STATUS_SUCCESS = 0,
        NVIMGCDCS_STATUS_NOT_INITIALIZED = 1,
        NVIMGCDCS_STATUS_INVALID_PARAMETER = 2,
        NVIMGCDCS_STATUS_BAD_CODESTREAM = 3,
        NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED = 4,
        NVIMGCDCS_STATUS_ALLOCATOR_FAILURE = 5,
        NVIMGCDCS_STATUS_EXECUTION_FAILED = 6,
        NVIMGCDCS_STATUS_ARCH_MISMATCH = 7,
        NVIMGCDCS_STATUS_INTERNAL_ERROR = 8,
        NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED = 9,
        NVIMGCDCS_STATUS_MISSED_DEPENDENCIES = 10,
        NVIMGCDCS_STATUS_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsStatus_t;

    typedef enum
    {
        NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN = 0,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 = 1,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 = 2,
        NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 = 3,
        NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 = 4,
        NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 = 5,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED = -1,
        NVIMGCDCS_SAMPLE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsSampleDataType_t;

    typedef enum
    {
        NVIMGCDCS_SAMPLING_NONE = 0,
        NVIMGCDCS_SAMPLING_444 = NVIMGCDCS_SAMPLING_NONE,
        NVIMGCDCS_SAMPLING_422 = 2,
        NVIMGCDCS_SAMPLING_420 = 3,
        NVIMGCDCS_SAMPLING_440 = 4,
        NVIMGCDCS_SAMPLING_411 = 5,
        NVIMGCDCS_SAMPLING_410 = 6,
        NVIMGCDCS_SAMPLING_GRAY = 7,
        NVIMGCDCS_SAMPLING_410V = 8,
        NVIMGCDCS_SAMPLING_UNSUPPORTED = -1,
        NVIMGCDCS_SAMPLING_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsChromaSubsampling_t;

     typedef enum
    {
        NVIMGCDCS_SAMPLEFORMAT_UNKNOWN = 0,
        NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED = 1, //unchanged planar
        NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED = 2, //unchanged interleaved
        NVIMGCDCS_SAMPLEFORMAT_P_RGB = 3,       //planar RGB
        NVIMGCDCS_SAMPLEFORMAT_I_RGB = 4,       //interleaved RGB
        NVIMGCDCS_SAMPLEFORMAT_P_BGR = 5,       //planar BGR
        NVIMGCDCS_SAMPLEFORMAT_I_BGR = 6,       //interleaved BGR
        NVIMGCDCS_SAMPLEFORMAT_P_Y = 7,         //Y component only
        NVIMGCDCS_SAMPLEFORMAT_P_YUV = 9,       //YUV planar format
        NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED = -1,
        NVIMGCDCS_SAMPLEFORMAT_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsSampleFormat_t;

    typedef enum
    {
        NVIMGCDCS_COLORSPEC_UNKNOWN = 0,
        NVIMGCDCS_COLORSPEC_SRGB = 1,
        NVIMGCDCS_COLORSPEC_GRAY = 2,
        NVIMGCDCS_COLORSPEC_SYCC = 3,
        NVIMGCDCS_COLORSPEC_CMYK = 4,
        NVIMGCDCS_COLORSPEC_YCCK = 5,
        NVIMGCDCS_COLORSPEC_UNSUPPORTED = -1,
        NVIMGCDCS_COLORSPEC_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsColorSpec_t;

    typedef enum
    {
        NVIMGCDCS_SCALE_NONE = 0,   // decoded output is not scaled
        NVIMGCDCS_SCALE_1_BY_2 = 1, // decoded output width and height is scaled by a factor of 1/2
        NVIMGCDCS_SCALE_1_BY_4 = 2, // decoded output width and height is scaled by a factor of 1/4
        NVIMGCDCS_SCALE_1_BY_8 = 3, // decoded output width and height is scaled by a factor of 1/8
        NVIMGCDCS_SCALE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsScaleFactor_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        int rotated; //Clockwise
        bool flip_x;
        bool flip_y;
    } nvimgcdcsOrientation_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        uint32_t width;
        uint32_t height;
        size_t row_stride;
        uint32_t num_channels;
        nvimgcdcsSampleDataType_t sample_type;
    } nvimgcdcsImagePlaneInfo_t;

#define NVIMGCDCS_MAX_NUM_DIM 5
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        int ndim; // number of dimensions, 0 means no region
        int start[NVIMGCDCS_MAX_NUM_DIM];
        int end[NVIMGCDCS_MAX_NUM_DIM];
    } nvimgcdcsRegion_t;

    typedef enum
    {
        NVIMGCDCS_IMAGE_BUFFER_KIND_UNKNOWN = 0,
        NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE = 1,
        NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST = 2,
        NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED = -1,
        NVIMGCDCS_IMAGE_BUFFER_KIND_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsImageBufferKind_t;

#define NVIMGCDCS_MAX_NUM_PLANES 32
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t width;
        uint32_t height;

        nvimgcdcsColorSpec_t color_spec;
        nvimgcdcsChromaSubsampling_t sampling;
        nvimgcdcsSampleDataType_t sample_type;
        nvimgcdcsOrientation_t orientation;
        nvimgcdcsSampleFormat_t sample_format;
        nvimgcdcsRegion_t region;

        void* buffer;
        size_t buffer_size;
        nvimgcdcsImageBufferKind_t buffer_kind;
        cudaStream_t cuda_stream; // stream to synchronize with
        uint32_t num_planes;
        nvimgcdcsImagePlaneInfo_t plane_info[NVIMGCDCS_MAX_NUM_PLANES];
    } nvimgcdcsImageInfo_t;

    // Currently parseable JPEG encodings (SOF markers)
    typedef enum
    {
        NVIMGCDCS_JPEG_ENCODING_UNKNOWN = 0x0,
        NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT = 0xc0,
        NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN = 0xc1,
        NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN = 0xc2,
        NVIMGCDCS_JPEG_ENCODING_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsJpegEncoding_t;

    //TODO fill with data in nvJpeg2k
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t width;
        uint32_t height;

    } nvimgcdcsJpeg2kResolutionInfo_t;

#define NVIMGCDCS_MAX_NUM_RESOLUTIONS 32
    //TODO fill with data in nvJpeg2k
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t tile_width;
        uint32_t tile_height;

        nvimgcdcsJpeg2kResolutionInfo_t resolution_info;
    } nvimgcdcsJpeg2kTileComponentInfo_t;

    //TODO fill with data in nvJpeg2k
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t num_resolutions;
        nvimgcdcsJpeg2kTileComponentInfo_t component_info[NVIMGCDCS_MAX_NUM_PLANES];
    } nvimgcdcsJpeg2kTileInfo_t;

    //TODO fill with data in nvJpeg2k
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t tile_width;
        uint32_t tile_height;
        uint32_t num_tiles_x;                // no of tiles in horizontal direction
        uint32_t num_tiles_y;                // no of tiles in vertical direction
        nvimgcdcsJpeg2kTileInfo_t tile_info; //for each tile in raster scan order
    } nvimgcdcsJpeg2kImageInfo_t;

    //TODO fill with data in nvJpeg
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        nvimgcdcsJpegEncoding_t encoding;
    } nvimgcdcsJpegImageInfo_t;

    struct nvimgcdcsImage;
    typedef struct nvimgcdcsImage* nvimgcdcsImage_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        bool use_cpu;
        bool use_gpu;
        bool use_hw_eng;
        int variant;
        int cuda_device_id;
    } nvimgcdcsBackend_t;

    typedef enum
    {
        NVIMGCDCS_PROCESSING_STATUS_SUCCESS = 0,
        NVIMGCDCS_PROCESSING_STATUS_INCOMPLETE = 1,
        NVIMGCDCS_PROCESSING_STATUS_IMAGE_CORRUPTED = 2,
        NVIMGCDCS_PROCESSING_STATUS_IMAGE_UNSUPPORTED = 3,
        NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED = 4,
        NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED = 5,
        NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED = 6,
        NVIMGCDCS_PROCESSING_STATUS_SCALING_UNSUPPORTED = 7,
        NVIMGCDCS_PROCESSING_STATUS_UNKNOWN_ORIENTATION = 8,
        NVIMGCDCS_PROCESSING_STATUS_DECODING = 9,
        NVIMGCDCS_PROCESSING_STATUS_ENCODING = 10,
        NVIMGCDCS_PROCESSING_STATUS_ERROR = 11,
        //...
        NVIMGCDCS_PROCESSING_STATUS_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsProcessingStatus_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        bool enable_orientation;
        bool enable_scaling;
        nvimgcdcsScaleFactor_t scale;
        bool enable_roi;
        //For Jpeg with 4 color components assumes CMYK colorspace and converts to RGB/YUV.
        //For Jpeg2k and 422/420 chroma subsampling enable conversion to RGB.
        bool enable_color_conversion;

        int num_backends; // Zero means that all backends are allowed.
        nvimgcdcsBackend_t* backends;
    } nvimgcdcsDecodeParams_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        uint32_t tile_id;
        uint32_t num_res_levels;
    } nvimgcdcsJpeg2kDecodeParams_t;

    typedef enum
    {
        NVIMGCDCS_MCT_MODE_YCC = 0, //transform RGB color images to YUV (default false)
        NVIMGCDCS_MCT_MODE_RGB = 1,
        NVIMGCDCS_MCT_MODE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsMctMode_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        float quality;
        float target_psnr;
        nvimgcdcsMctMode_t mct_mode;

        int num_backends; //Zero means that all backends are allowed.
        nvimgcdcsBackend_t* backends;
    } nvimgcdcsEncodeParams_t;

#define NVIMGCDCS_JPEG2K_MAXRES 33

    typedef enum
    {
        NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP = 0,
        NVIMGCDCS_JPEG2K_PROG_ORDER_RLCP = 1,
        NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL = 2,
        NVIMGCDCS_JPEG2K_PROG_ORDER_PCRL = 3,
        NVIMGCDCS_JPEG2K_PROG_ORDER_CPRL = 4,
        NVIMGCDCS_JPEG2K_PROG_ORDER_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsJpeg2kProgOrder_t;

    typedef enum
    {
        NVIMGCDCS_JPEG2K_STREAM_J2K = 0,
        NVIMGCDCS_JPEG2K_STREAM_JP2 = 1,
        NVIMGCDCS_JPEG2K_STREAM_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsJpeg2kBitstreamType_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        nvimgcdcsJpeg2kBitstreamType_t stream_type;
        uint16_t rsiz;
        uint32_t enable_SOP_marker;
        uint32_t enable_EPH_marker;
        nvimgcdcsJpeg2kProgOrder_t prog_order;
        uint32_t num_layers;
        uint32_t num_resolutions;
        uint32_t code_block_w;
        uint32_t code_block_h;
        uint32_t encode_modes;
        uint32_t irreversible;
        uint32_t enable_custom_precincts;
        uint32_t precint_width[NVIMGCDCS_JPEG2K_MAXRES];
        uint32_t precint_height[NVIMGCDCS_JPEG2K_MAXRES];
    } nvimgcdcsJpeg2kEncodeParams_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;
        nvimgcdcsJpegEncoding_t encoding;
        bool optimized_huffman;
    } nvimgcdcsJpegEncodeParams_t;

    typedef enum
    {
        NVIMGCDCS_CAPABILITY_UNKNOWN = 0,
        NVIMGCDCS_CAPABILITY_SCALING = 1,
        NVIMGCDCS_CAPABILITY_ORIENTATION = 2,
        NVIMGCDCS_CAPABILITY_ROI = 3,
        NVIMGCDCS_CAPABILITY_BATCH = 4,
        NVIMGCDCS_CAPABILITY_HOST_INPUT = 5,
        NVIMGCDCS_CAPABILITY_HOST_OUTPUT = 6,
        NVIMGCDCS_CAPABILITY_DEVICE_INPUT = 7,
        NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT = 8,
        NVIMGCDCS_CAPABILITY_LAYOUT_PLANAR = 9,
        NVIMGCDCS_CAPABILITY_LAYOUT_INTERLEAVED = 10,
        NVIMGCDCS_CAPABILITY_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsCapability_t;

    struct nvimgcdcsIOStreamDesc
    {
        nvimgcdcsStructureType_t type;
        void* next;

        void* instance;

        nvimgcdcsStatus_t (*read)(void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*write)(void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);
        nvimgcdcsStatus_t (*skip)(void* instance, size_t count);
        nvimgcdcsStatus_t (*seek)(void* instance, size_t offset, int whence);
        nvimgcdcsStatus_t (*tell)(void* instance, size_t* offset);
        nvimgcdcsStatus_t (*size)(void* instance, size_t* size);
        nvimgcdcsStatus_t (*raw_data)(void* instance, const void**);
    };
    typedef struct nvimgcdcsIOStreamDesc* nvimgcdcsIoStreamDesc_t;

    struct nvimgcdcsInstance;
    typedef struct nvimgcdcsInstance* nvimgcdcsInstance_t;

    struct nvimgcdcsCodeStream;
    typedef struct nvimgcdcsCodeStream* nvimgcdcsCodeStream_t;

    struct nvimgcdcsEncoder;
    typedef struct nvimgcdcsEncoder* nvimgcdcsEncoder_t;

    struct nvimgcdcsDecoder;
    typedef struct nvimgcdcsDecoder* nvimgcdcsDecoder_t;

    struct nvimgcdcsEncodeState;
    typedef nvimgcdcsEncodeState* nvimgcdcsEncodeState_t;

    struct nvimgcdcsDecodeState;
    typedef nvimgcdcsDecodeState* nvimgcdcsDecodeState_t;

    struct nvimgcdcsDebugMessenger;
    typedef nvimgcdcsDebugMessenger* nvimgcdcsDebugMessenger_t;

    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_NONE = 0,
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE = 1, // Diagnostic message useful for developers
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG = 2, // Diagnostic message useful for developers
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO = 4,  // Informational message like the creation of a resource
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING =
            8, // Message about behavior that is not necessarily an error, but very likely a bug in your application
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR =
            16, // Message about behavior that is invalid and may cause impropare execution or result of operation (e.g. can't open file) but not application
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL =
            24, // Message about behavior that is invalid and may cause crashes and forcing to shutdown application
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ALL = 0x0FFFFFFF, //Used in case filtering out by message severity
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsDebugMessageSeverity_t;

    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_NONE = 0,
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL = 1,      // Some event has happened that is unrelated to the specification or performance
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_VALIDATION = 2,   // Something has happened that indicates a possible mistake
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_PERFORMANCE = 4,  // Potential non-optimal use
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL = 0x0FFFFFFF, //Used in case filtering out by message type
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsDebugMessageType_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        const char* message;         //null-terminated string detailing the trigger conditions
        uint32_t internal_status_id; //it is internal codec status id
        const char* codec;           //codec name if codec is rising message or NULL otherwise (e.g framework)
        const char* codec_id;
        uint32_t codec_version;
    } nvimgcdcsDebugMessageData_t;

    typedef bool (*nvimgcdcsDebugCallback_t)(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* callback_data,
        void* user_data // pointer that was specified during the setup of the callback
    );

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        uint32_t message_severity;
        uint32_t message_type;
        nvimgcdcsDebugCallback_t user_callback;
        void* userData;
    } nvimgcdcsDebugMessengerDesc_t;

    struct nvimgcdcsFuture;
    typedef struct nvimgcdcsFuture* nvimgcdcsFuture_t;

    struct nvimgcdcsParser;
    typedef struct nvimgcdcsParser* nvimgcdcsParser_t;

    struct nvimgcdcsParseState;
    typedef struct nvimgcdcsParseState* nvimgcdcsParseState_t;

    struct nvimgcdcsCodeStreamDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance;

        nvimgcdcsIoStreamDesc_t io_stream;
        nvimgcdcsParseState_t parse_state;

        nvimgcdcsStatus_t (*getCodecName)(void* instance, char* codec_name);
        nvimgcdcsStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* result);
    };
    typedef struct nvimgcdcsCodeStreamDesc* nvimgcdcsCodeStreamDesc_t;

    struct nvimgcdcsImageDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance;
        nvimgcdcsStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* result);
        nvimgcdcsStatus_t (*imageReady)(void* instance, nvimgcdcsProcessingStatus_t processing_status);
    };
    typedef struct nvimgcdcsImageDesc* nvimgcdcsImageDesc_t;

    struct nvimgcdcsFrameworkDesc;
    typedef struct nvimgcdcsFrameworkDesc* nvimgcdcsFrameworkDesc_t;

    struct nvimgcdcsParserDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canParse)(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream);
        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsParser_t* parser);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsParser_t parser);

        nvimgcdcsStatus_t (*createParseState)(nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state);
        nvimgcdcsStatus_t (*destroyParseState)(nvimgcdcsParseState_t parse_state);

        nvimgcdcsStatus_t (*getImageInfo)(nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* result, nvimgcdcsCodeStreamDesc_t code_stream);

        nvimgcdcsStatus_t (*getCapabilities)(nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size);
    };

    struct nvimgcdcsEncoderDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canEncode)(void* instance, bool* result, nvimgcdcsImageDesc_t image, nvimgcdcsCodeStreamDesc_t code_stream,
            const nvimgcdcsEncodeParams_t* params);

        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsEncoder_t encoder);

        nvimgcdcsStatus_t (*createEncodeState)(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state);
        nvimgcdcsStatus_t (*createEncodeStateBatch)(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state);
        nvimgcdcsStatus_t (*destroyEncodeState)(nvimgcdcsEncodeState_t encode_state);

        nvimgcdcsStatus_t (*getCapabilities)(nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size);

        nvimgcdcsStatus_t (*encode)(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encode_state, nvimgcdcsImageDesc_t image,
            nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params);

        nvimgcdcsStatus_t (*encodeBatch)(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encode_state_batch,
            nvimgcdcsEncodeState_t* encode_states, nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size,
            const nvimgcdcsEncodeParams_t* params);
    };

    typedef struct nvimgcdcsEncoderDesc* nvimgcdcsEncoderDesc_t;

    struct nvimgcdcsDecoderDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canDecode)(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image,
            const nvimgcdcsDecodeParams_t* params);

        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsDecoder_t decoder);

        nvimgcdcsStatus_t (*createDecodeState)(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
        nvimgcdcsStatus_t (*createDecodeStateBatch)(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
        nvimgcdcsStatus_t (*destroyDecodeState)(nvimgcdcsDecodeState_t decode_state);

        nvimgcdcsStatus_t (*getCapabilities)(nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);

        nvimgcdcsStatus_t (*decode)(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decode_state, nvimgcdcsCodeStreamDesc_t code_stream,
            nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params);

        nvimgcdcsStatus_t (*decodeBatch)(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decode_state_batch,
            nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    };

    typedef struct nvimgcdcsDecoderDesc* nvimgcdcsDecoderDesc_t;

    struct nvimgcdcsExecutorDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance;

        nvimgcdcsStatus_t (*launch)(void* instance, int device_id, int sample_idx, void* task_context,
            void (*task)(int thread_id, int sample_idx, void* task_context));
        int (*get_num_threads)(void* instance);
    };

    typedef struct nvimgcdcsExecutorDesc* nvimgcdcsExecutorDesc_t;

    typedef struct
    {
        // Output
        void* out_buffer;
        size_t out_buffer_sz;
        nvimgcdcsSampleFormat_t out_sample_format;
        nvimgcdcsSampleDataType_t out_sample_type;
        // Input
        void* in_buffer;
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsRegion_t region;

        bool flip_y;
        bool flip_x;
        bool swap_xy;

        float multiplier; // used to adjust dynamic range
    } nvimgcdcsImageProcessorConvertParams_t;

    struct nvimgcdcsImageProcessorDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance;

        nvimgcdcsStatus_t (*convert_cpu)(
            void* instance, const nvimgcdcsImageProcessorConvertParams_t* params, nvimgcdcsPinnedAllocator_t* pinned_allocator);
        nvimgcdcsStatus_t (*convert_gpu)(void* instance, const nvimgcdcsImageProcessorConvertParams_t* params,
            nvimgcdcsDeviceAllocator_t dev_allocator, cudaStream_t cuda_stream);
    };

    typedef struct nvimgcdcsImageProcessorDesc* nvimgcdcsImageProcessorDesc_t;

    typedef nvimgcdcsStatus_t (*nvimgcdcsLogFunc_t)(void* instance, const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data);

    struct nvimgcdcsFrameworkDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        const char* id; // framework named identifier e.g. nvImageCodecs
        uint32_t version;
        void* instance;

        nvimgcdcsDeviceAllocator_t* device_allocator;
        nvimgcdcsPinnedAllocator_t* pinned_allocator;

        nvimgcdcsStatus_t (*registerEncoder)(void* instance, const nvimgcdcsEncoderDesc_t desc);
        nvimgcdcsStatus_t (*registerDecoder)(void* instance, const nvimgcdcsDecoderDesc_t desc);
        nvimgcdcsStatus_t (*registerParser)(void* instance, const struct nvimgcdcsParserDesc* desc);
        nvimgcdcsStatus_t (*getExecutor)(void* instance, nvimgcdcsExecutorDesc_t* result);
        nvimgcdcsStatus_t (*getImageProcessor)(void* instance, nvimgcdcsImageProcessorDesc_t* result);
        nvimgcdcsLogFunc_t log;
    };

    struct nvimgcdcsExtension;
    typedef struct nvimgcdcsExtension* nvimgcdcsExtension_t;

    typedef struct nvimgcdcsExtensionDesc
    {
        nvimgcdcsStructureType_t type;
        void* next;

        const char* id;
        uint32_t version;

        nvimgcdcsStatus_t (*create)(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension);
        nvimgcdcsStatus_t (*destroy)(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension);
    } nvimgcdcsExtensionDesc_t;

    typedef nvimgcdcsStatus_t (*nvimgcdcsExtensionModuleEntryFunc_t)(nvimgcdcsExtensionDesc_t* ext_desc);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc);

    // Instance
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        void* next;

        nvimgcdcsDeviceAllocator_t* device_allocator;
        nvimgcdcsPinnedAllocator_t* pinned_allocator;
        bool load_extension_modules;  //Discover and load extension modules on start
        bool default_debug_messenger; //Create default debug messenger
        uint32_t message_severity;    //severity for default debug messenger
        uint32_t message_type;        //message type for default debug messenger
        int num_cpu_threads;
        nvimgcdcsExecutorDesc_t executor = nullptr;
    } nvimgcdcsInstanceCreateInfo_t;

    // Instance
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceCreate(nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t create_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceDestroy(nvimgcdcsInstance_t instance);

    // Extension
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsExtension_t* extension, nvimgcdcsExtensionDesc_t* extension_desc);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionDestroy(nvimgcdcsExtension_t extension);

    // Debug Messenger
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessengerCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsDebugMessenger_t* dbg_messenger, const nvimgcdcsDebugMessengerDesc_t* messenger_desc);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessengerDestroy(nvimgcdcsDebugMessenger_t dbg_messenger);

    // Future
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsFutureWaitForAll(nvimgcdcsFuture_t future);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsFutureDestroy(nvimgcdcsFuture_t future);

    // Image
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDestroy(nvimgcdcsImage_t image);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetImageInfo(nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetProcessingStatus(
        nvimgcdcsImage_t image, nvimgcdcsProcessingStatus_t* processing_status);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageAttachEncodeState(nvimgcdcsImage_t image, nvimgcdcsEncodeState_t encode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDetachEncodeState(nvimgcdcsImage_t image);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageAttachDecodeState(nvimgcdcsImage_t image, nvimgcdcsDecodeState_t decode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDetachDecodeState(nvimgcdcsImage_t image);

    // CodeStream
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromFile(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const char* file_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const unsigned char* data, size_t length);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle,
        const char* file_name, const char* codec_name, const nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle,
        unsigned char* output_buffer, size_t length, const char* codec_name, const nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamDestroy(nvimgcdcsCodeStream_t stream_handle);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamGetCodecName(nvimgcdcsCodeStream_t stream_handle, char* codec_name);

    //Decoder
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDestroy(nvimgcdcsDecoder_t decoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStream_t stream, nvimgcdcsImage_t image,
        const nvimgcdcsDecodeParams_t* params, nvimgcdcsFuture_t* future, bool blocking);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDecodeBatch(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decode_state_batch,
        nvimgcdcsCodeStream_t* streams, nvimgcdcsImage_t* images, int batch_size, nvimgcdcsDecodeParams_t* params,
        nvimgcdcsFuture_t* future, bool blocking);

    //DecodeState
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateCreate(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateBatchCreate(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateDestroy(nvimgcdcsDecodeState_t decode_state);

    //Encoder
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderDestroy(nvimgcdcsEncoder_t encoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, nvimgcdcsCodeStream_t stream,
        nvimgcdcsImage_t input_image, const nvimgcdcsEncodeParams_t* params, nvimgcdcsFuture_t* future, bool blocking);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderEncodeBatch(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encode_state_batch,
        nvimgcdcsImage_t* images, nvimgcdcsCodeStream_t* streams, int batch_size, nvimgcdcsEncodeParams_t* params,
        nvimgcdcsFuture_t* future, bool blocking);

    //EncodeState
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateCreate(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateBatchCreate(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state_batch);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateDestroy(nvimgcdcsEncodeState_t encode_state);

    //High-level API

    typedef enum
    {
        NVIMGCDCS_IMREAD_UNCHANGED = -1,
        NVIMGCDCS_IMREAD_GRAYSCALE = 0, // do not convert to RGB
        //for jpeg with 4 color components assumes CMYK colorspace and converts to RGB
        //for Jpeg2k and 422/420 chroma subsampling enable conversion to RGB
        NVIMGCDCS_IMREAD_COLOR = 1,
        NVIMGCDCS_IMREAD_IGNORE_ORIENTATION = 128, //Ignore orientation from Exif
        NVIMGCDCS_IMREAD_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsImreadFlags_t;

    typedef enum
    {
        NVIMGCDCS_IMWRITE_JPEG_QUALITY = 1, // 0-100 default 95
        NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE = 2,
        NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE = 3, //optimized_huffman
        NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR = 7,

        NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR = 100,     // default 50
        NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS = 101,     // num_decomps default 5
        NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE = 103, // code_block_w code_block_h (default 64 64)
        NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE = 104,
        NVIMGCDCS_IMWRITE_MCT_MODE = 500, // nvimgcdcsMctMode_t value (default NVIMGCDCS_MCT_MODE_RGB )
        NVIMGCDCS_IMWRITE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsImwriteParams_t;

    typedef enum
    {
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444 = 0x111111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422 = 0x211111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420 = 0x221111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440 = 0x121111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411 = 0x411111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410 = 0x441111,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410V = 0x440000,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY = 0x110000,
        NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsImwriteSamplingFactor_t;

    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImRead(nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const char* file_name, int flags);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImWrite(
        nvimgcdcsInstance_t instance, nvimgcdcsImage_t image, const char* file_name, const int* params);

#if defined(__cplusplus)
}
#endif

#endif
