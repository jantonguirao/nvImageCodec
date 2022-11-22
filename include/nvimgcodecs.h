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
//#include "library_types.h"
#include "nvimgcdcs_data.h"
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
        NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION,
        NVIMGCDCS_STRUCTURE_TYPE_IO_STREAM_DESC,
        NVIMGCDCS_STRUCTURE_TYPE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsStructureType_t;

    // Prototype for device memory allocation, modelled after cudaMalloc()
    typedef int (*nvimgcdcsDeviceMalloc_t)(void**, size_t);
    // Prototype for device memory release
    typedef int (*nvimgcdcsDeviceFree_t)(void*);

    // Prototype for pinned memory allocation, modelled after cudaHostAlloc()
    typedef int (*nvimgcdcsPinnedMalloc_t)(void**, size_t, unsigned int flags);
    // Prototype for device memory release
    typedef int (*nvimgcdcsPinnedFree_t)(void*);

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsDeviceMalloc_t device_malloc;
        nvimgcdcsDeviceFree_t device_free;
    } nvimgcdcsDeviceAllocator_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsPinnedMalloc_t pinned_malloc;
        nvimgcdcsPinnedFree_t pinned_free;
    } nvimgcdcsPinnedAllocator_t;

    typedef int (*nvimgcdcsDeviceMallocV2_t)(
        void* ctx, void** ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsDeviceFreeV2_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsPinnedMallocV2_t)(
        void* ctx, void** ptr, size_t size, cudaStream_t stream);

    typedef int (*nvimgcdcsPinnedFreeV2_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsDeviceMallocV2_t dev_malloc;
        nvimgcdcsDeviceFreeV2_t dev_free;
        void* dev_ctx;
    } nvimgcdcsDevAllocatorV2_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsPinnedMallocV2_t pinned_malloc;
        nvimgcdcsPinnedFreeV2_t pinned_free;
        void* pinned_ctx;
    } nvimgcdcsPinnedAllocatorV2_t;

    typedef enum
    {
        NVIMGCDCS_STATUS_UNKNOWN                      = -1,
        NVIMGCDCS_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_STATUS_BAD_CODESTREAM               = 3,
        NVIMGCDCS_STATUS_CODESTREAM_NOT_SUPPORTED     = 4,
        NVIMGCDCS_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
        NVIMGCDCS_STATUS_MISSED_DEPENDENCIES          = 10,
        NVIMGCDCS_STATUS_ENUM_FORCE_INT               = 0xFFFFFFFF
    } nvimgcdcsStatus_t;

    typedef enum
    {
        NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN       = -1,
        NVIMGCDCS_SAMPLE_DATA_TYPE_NOT_SUPPORTED = 0,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8         = 1,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16        = 2,
        NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8         = 3,
        NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16        = 4,
        NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32       = 5,
        NVIMGCDCS_SAMPLE_ENUM_FORCE_INT          = 0xFFFFFFFF
    } nvimgcdcsSampleDataType_t;

    typedef enum
    {
        NVIMGCDCS_SAMPLING_UNKNOWN        = -1,
        NVIMGCDCS_SAMPLING_NOT_SUPPORTED  = 0,
        NVIMGCDCS_SAMPLING_444            = 1,
        NVIMGCDCS_SAMPLING_422            = 2,
        NVIMGCDCS_SAMPLING_420            = 3,
        NVIMGCDCS_SAMPLING_440            = 4,
        NVIMGCDCS_SAMPLING_411            = 5,
        NVIMGCDCS_SAMPLING_410            = 6,
        NVIMGCDCS_SAMPLING_GRAY           = 7,
        NVIMGCDCS_SAMPLING_410V           = 8,
        NVIMGCDCS_SAMPLING_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsSampling_t;

    //Interleave - even Planar - odd
    typedef enum
    {
        NVIMGCDCS_SAMPLEFORMAT_UNKNOWN        = -1,
        NVIMGCDCS_SAMPLEFORMAT_NOT_SUPPORTED  = 0,
        NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED    = 1, //unchanged planar
        NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED    = 2, //unchanged interleave
        NVIMGCDCS_SAMPLEFORMAT_P_RGB          = 3, //planar RGB
        NVIMGCDCS_SAMPLEFORMAT_I_RGB          = 4, //interleaved RGB
        NVIMGCDCS_SAMPLEFORMAT_P_BGR          = 5, //planar BGR
        NVIMGCDCS_SAMPLEFORMAT_I_BGR          = 6, //interleaved BGR
        NVIMGCDCS_SAMPLEFORMAT_P_Y            = 7, //Y component only
        NVIMGCDCS_SAMPLEFORMAT_P_YUV          = 9, //YUV planar format
        NVIMGCDCS_SAMPLEFORMAT_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsSampleFormat_t;

    typedef enum
    {
        NVIMGCDCS_COLORSPACE_UNKNOWN        = -1,
        NVIMGCDCS_COLORSPACE_NOT_SUPPORTED  = 0,
        NVIMGCDCS_COLORSPACE_SRGB           = 1,
        NVIMGCDCS_COLORSPACE_GRAY           = 2,
        NVIMGCDCS_COLORSPACE_SYCC           = 3,
        NVIMGCDCS_COLORSPACE_CMYK           = 4,
        NVIMGCDCS_COLORSPACE_YCCK           = 5,
        NVIMGCDCS_COLORSPACE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsColorSpace_t;

    typedef enum
    {
        NVIMGCDCS_SCALE_NONE   = 0, // decoded output is not scaled
        NVIMGCDCS_SCALE_1_BY_2 = 1, // decoded output width and height is scaled by a factor of 1/2
        NVIMGCDCS_SCALE_1_BY_4 = 2, // decoded output width and height is scaled by a factor of 1/4
        NVIMGCDCS_SCALE_1_BY_8 = 3, // decoded output width and height is scaled by a factor of 1/8
        NVIMGCDCS_SCALE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsScaleFactor_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        int rotated;
        bool flip_x;
        bool flip_y;
    } nvimgcdcsOrientation_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        uint32_t component_width;
        uint32_t component_height;
        size_t host_pitch_in_bytes;
        size_t device_pitch_in_bytes;
        nvimgcdcsSampleDataType_t sample_type;
    } nvimgcdcsImageComponentInfo_t;

#define NVIMGCDCS_MAX_NUM_COMPONENTS 32
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        uint32_t image_width;
        uint32_t image_height;
        uint32_t tile_width;
        uint32_t tile_height;
        uint32_t num_tiles_x; // no of tiles in horizontal direction
        uint32_t num_tiles_y; // no of tiles in vertical direction
        uint32_t num_components;
        nvimgcdcsImageComponentInfo_t component_info[NVIMGCDCS_MAX_NUM_COMPONENTS];
        nvimgcdcsColorSpace_t color_space;
        nvimgcdcsSampleFormat_t sample_format;
        nvimgcdcsSampling_t sampling;
        nvimgcdcsSampleDataType_t sample_type;
    } nvimgcdcsImageInfo_t;

    struct nvimgcdcsImage;
    typedef struct nvimgcdcsImage* nvimgcdcsImage_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        bool useCPU;
        bool useGPU;
        bool useHwEng;
        int variant;
    } nvimgcdcsBackend_t;

    typedef enum
    {
        NVIMGCDCS_PROCESSING_STATUS_UNKNOWN                   = -1,
        NVIMGCDCS_PROCESSING_STATUS_SUCCESS                   = 0,
        NVIMGCDCS_PROCESSING_STATUS_INCOMPLETE                = 1,
        NVIMGCDCS_PROCESSING_STATUS_IMAGE_CORRUPTED           = 2,
        NVIMGCDCS_PROCESSING_STATUS_IMAGE_NOT_SUPPORTED       = 3,
        NVIMGCDCS_PROCESSING_STATUS_SAMPLING_NOT_SUPPORTED    = 4,
        NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_NOT_SUPPORTED = 5,
        NVIMGCDCS_PROCESSING_STATUS_SCALING_NOT_SUPPORTED     = 6,
        NVIMGCDCS_PROCESSING_STATUS_DECODING                  = 7,
        NVIMGCDCS_PROCESSING_STATUS_ENCODING                  = 8,
        //...
        NVIMGCDCS_PROCESSING_STATUS_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsProcessingStatus_t;

    typedef enum
    {
        NVIMGCDCS_DECODE_PHASE_ALL            = 0,
        NVIMGCDCS_DECODE_PHASE_HOST           = 1,
        NVIMGCDCS_DECODE_PHASE_MIXED          = 2,
        NVIMGCDCS_DECODE_PHASE_DEVICE         = 3,
        NVIMGCDCS_DECODE_PHASE_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsDecodePhase_t;

    typedef enum
    {
        NVIMGCDCS_DECODE_STEP_ALL              = 0,
        NVIMGCDCS_DECODE_STEP_COLOR_TRANSFORM  = 1,
        NVIMGCDCS_DECODE_STEP_DOMAIN_TRANSFORM = 2,
        NVIMGCDCS_DECODE_STEP_QUANTIZATION     = 3,
        NVIMGCDCS_DECODE_STEP_PRE_ENTROPY      = 4,
        NVIMGCDCS_DECODE_STEP_ENTROPY          = 5,
        NVIMGCDCS_DECODE_STEP_PACKAGING        = 6,
        NVIMGCDCS_DECODE_STEP_ENUM_FORCE_INT   = 0xFFFFFFFF
    } nvimgcdcsDecodeStep_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        nvimgcdcsDecodeStep_t decode_step;
        nvimgcdcsDecodePhase_t decode_phase;
        bool apply_orientation;
        nvimgcdcsOrientation_t orientation;
        bool apply_scaling;
        nvimgcdcsScaleFactor_t scale;
        int num_backends; //Zero means that all backendsa re allowed.
        nvimgcdcsBackend_t* backends;  
        int maxCpuThreads;
        int cudaDeviceId;
        cudaStream_t cuda_stream;
        nvimgcdcsDataDict_t config;
    } nvimgcdcsDecodeParams_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        double qstep;
        double target_psnr;
        const char* codec;
        nvimgcdcsBackend_t backend;
        nvimgcdcsDataDict_t config;
        cudaStream_t cuda_stream;
    } nvimgcdcsEncodeParams_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        int stream_count;

    } nvimgcdcsContainer;
    typedef nvimgcdcsContainer* nvimgcdcsContainer_t;

    typedef enum
    {
        NVIMGCDCS_CAPABILITY_UNKNOWN        = -1,
        NVIMGCDCS_CAPABILITY_SCALING        = 1,
        NVIMGCDCS_CAPABILITY_ORIENTATION    = 2,
        NVIMGCDCS_CAPABILITY_PARTIAL        = 3,
        NVIMGCDCS_CAPABILITY_ROI            = 4,
        NVIMGCDCS_CAPABILITY_BATCH          = 5,
        NVIMGCDCS_CAPABILITY_HOST_INPUT     = 6,
        NVIMGCDCS_CAPABILITY_HOST_OUTPUT    = 7,
        NVIMGCDCS_CAPABILITY_DEVICE_INPUT   = 8,
        NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT  = 9,
        NVIMGCDCS_CAPABILITY_PHASE_DECODING = 10,
        NVIMGCDCS_CAPABILITY_STEP_DECODING  = 11,
        NVIMGCDCS_CAPABILITY_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsCapability_t;

    struct nvimgcdcsIOStreamDesc
    {
        nvimgcdcsStructureType_t type;
        const void* next;

        void* instance;

        nvimgcdcsStatus_t (*read)(void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*write)(void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);
        nvimgcdcsStatus_t (*skip)(void* instance, size_t count);
        nvimgcdcsStatus_t (*seek)(void* instance, size_t offset, int whence);
        nvimgcdcsStatus_t (*tell)(void* instance, size_t* offset);
        nvimgcdcsStatus_t (*size)(void* instance, size_t* size);
    };
    typedef struct nvimgcdcsIOStreamDesc* nvimgcdcsIoStreamDesc_t;

    struct nvimgcdcsHandle;
    typedef struct nvimgcdcsHandle* nvimgcdcsInstance_t;

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

    // Instance
    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsDeviceAllocator_t* device_allocator; //TODO
        nvimgcdcsPinnedAllocator_t* pinned_allocator; //TODO
        size_t deviceMemPadding;                      //TODO any device memory allocation
        // would be padded to the multiple of specified number of bytes.
        size_t pinnedMemPadding; //TODO any pinned host memory allocation
        // would be padded to the multiple of specified number of bytes
    } nvimgcdcsInstanceCreateInfo_t;

    struct nvimgcdcsDebugMessage;
    typedef nvimgcdcsDebugMessage* nvimgcdcsDebugMessage_t;

    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_VERBOSE_BIT = 1, // Diagnostic message
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO_BIT =
            2, // Informational message like the creation of a resource
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING_BIT =
            4, // Message about behavior that is not necessarily an error, but very likely a bug in your application
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR_BIT =
            8, // Message about behavior that is invalid and may cause crashes
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ENUM_FORCE_INT = 0xFFFFFFFF
    } nvimgcdcsDebugMessageSeverity_t;

    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL_BIT =
            1, // Some event has happened that is unrelated to the specification or performance
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_VALIDATION_BIT =
            2, // Something has happened that indicates a possible mistake
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_PERFORMANCE_BIT = 4, // Potential non-optimal use
        NVIMGCDCS_DEBUG_MESSAGE_TYPE_ENUM_FORCE_INT  = 0xFFFFFFFF
    } nvimgcdcsDebugMessageType_t;

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        const char* message;       //null-terminated string detailing the trigger conditions
        uint32_t internalStatusId; //it is internal codec status id
        const char* codec; //codec name if codec is rising message or NULL otherwise (e.g framework)
        const char* codecId;
        uint32_t codecVersion;
    } nvimgcdcsDebugMessageData_t;

    typedef bool (*nvimgcdcsDebugCallback_t)(nvimgcdcsDebugMessageSeverity_t messageSeverity,
        const nvimgcdcsDebugMessageType_t messageType,
        const nvimgcdcsDebugMessageData_t pCallbackData,
        void* user_data // pointer that was specified during the setup of the callback
    );

    typedef struct
    {
        nvimgcdcsStructureType_t type;
        const void* next;
        nvimgcdcsDebugMessageSeverity_t messageSeverity;
        nvimgcdcsDebugMessageType_t messageType;
        nvimgcdcsDebugCallback_t userCallback;
        void* userData;
    } nvimgcdcsDebugMessengerCreateInfo_t;

    // Instance
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceCreate(
        nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t createInfo);

    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceDestroy(nvimgcdcsInstance_t instance);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceGetReadyImage(nvimgcdcsInstance_t instance,
        nvimgcdcsImage_t* image, nvimgcdcsProcessingStatus_t* processing_status, bool blocking);

    // Debug Messanger
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessangerCreate(nvimgcdcsInstance_t instance,
        nvimgcdcsDebugMessage_t* dbgMessenger,
        nvimgcdcsDebugMessengerCreateInfo_t createInfo); //TODO
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessangerDestroy(
        nvimgcdcsDebugMessage_t* dbgMessenger); //TODO

    // Image
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDestroy(nvimgcdcsImage_t image);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageSetHostBuffer(
        nvimgcdcsImage_t image, void* buffer, size_t size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetHostBuffer(
        nvimgcdcsImage_t image, void** buffer, size_t* size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageSetDeviceBuffer(
        nvimgcdcsImage_t image, void* buffer, size_t size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetDeviceBuffer(
        nvimgcdcsImage_t image, void** buffer, size_t* size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageSetImageInfo(
        nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetImageInfo(
        nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetProcessingStatus(
        nvimgcdcsImage_t image, nvimgcdcsProcessingStatus_t* processing_status);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageAttachEncodeState(
        nvimgcdcsImage_t image, nvimgcdcsEncodeState_t encode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDetachEncodeState(nvimgcdcsImage_t image);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageAttachDecodeState(
        nvimgcdcsImage_t image, nvimgcdcsDecodeState_t decode_state);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDetachDecodeState(nvimgcdcsImage_t image);

    // CodeStream
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromFile(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const char* file_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, unsigned char* data,
        size_t length);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromIOStream(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle,
        nvimgcdcsIoStreamDesc_t io_stream); //TODO
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(nvimgcdcsInstance_t instance,
        nvimgcdcsCodeStream_t* stream_handle, const char* file_name, const char* codec_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance,
        nvimgcdcsCodeStream_t* stream_handle, unsigned char* output_buffer, size_t length,
        const char* codec_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamDestroy(nvimgcdcsCodeStream_t stream_handle);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(
        nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamSetImageInfo(
        nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamGetCodecName(
        nvimgcdcsCodeStream_t stream_handle, char* codec_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateCopyExtMetaData(
        nvimgcdcsEncodeState_t encodeState, nvimgcdcsCodeStream_t dst_stream_handle,
        nvimgcdcsCodeStream_t src_stream_handle); //TODO

    //Decoder
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCreate(nvimgcdcsInstance_t instance,
        nvimgcdcsDecoder_t* decoder, nvimgcdcsCodeStream_t stream, nvimgcdcsDecodeParams_t* params);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDestroy(nvimgcdcsDecoder_t decoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder,
        nvimgcdcsCodeStream_t stream, nvimgcdcsImage_t image, nvimgcdcsDecodeParams_t* params);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDecodeBatch(nvimgcdcsDecoder_t decoder,
        nvimgcdcsDecodeParams_t* params, nvimgcdcsContainer_t container, int batchSize,
        nvimgcdcsImage_t* image); //TODO
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderGetCapabilities(
        nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCanUseDecodeState(
        nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decodeState, bool* canUse); //TODO

    //DecodeState
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateCreate(
        nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state, cudaStream_t cuda_stream);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateDestroy(nvimgcdcsDecodeState_t decode_state);

    //Encoder
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCreate(nvimgcdcsInstance_t instance,
        nvimgcdcsEncoder_t* encoder, nvimgcdcsCodeStream_t stream, nvimgcdcsEncodeParams_t* params);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderDestroy(nvimgcdcsEncoder_t encoder);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder,
        nvimgcdcsCodeStream_t stream, nvimgcdcsImage_t input_image,
        nvimgcdcsEncodeParams_t* encode_params);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderEncodeBatch(nvimgcdcsEncoder_t encoder,
        nvimgcdcsDecodeParams_t* params, nvimgcdcsContainer_t container, int batchSize,
        nvimgcdcsImage_t* image); //TODO
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderGetCapabilities(
        nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCanUseEncodeState(
        nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encodeState, bool* canUse); //TODO

    //EncodeState
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateCreate(
        nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state, cudaStream_t cuda_stream);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncodeStateDestroy(nvimgcdcsEncodeState_t encode_state);

    //High-level API
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImRead(
        nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const char* file_name);
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImWrite(nvimgcdcsInstance_t instance,
        nvimgcdcsImage_t image, const char* file_name, const int* params);
#if defined(__cplusplus)
}
#endif

#endif
