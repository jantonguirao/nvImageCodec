/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 * 
 */

/** 
 * @brief The nvImageCodecs library and extension API
 * 
 * @file nvimgcodecs.h
 *   
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

/**
 * @brief Maximal size of codec name
 */
#define NVIMGCDCS_MAX_CODEC_NAME_SIZE 256

/**
 * @brief Defines device id as current device
 */
#define NVIMGCDCS_DEVICE_CURRENT -1

/**
 * @brief Defines device id as a CPU only 
 */
#define NVIMGCDCS_DEVICE_CPU_ONLY -99999

/**
 * @brief Maximal number of dimensions.
 */
#define NVIMGCDCS_MAX_NUM_DIM 5

/**
 * @brief Maximum number of image planes.
 */
#define NVIMGCDCS_MAX_NUM_PLANES 32

/**
 * @brief Maximum number of JPEG2000 resolutions.
 */
#define NVIMGCDCS_JPEG2K_MAXRES 33

    /**
     * @brief Opaque nvImageCodecs library instance type.
     */
    struct nvimgcdcsInstance;

    /**
     * @brief Handle to opaque nvImageCodecs library instance type.
     */
    typedef struct nvimgcdcsInstance* nvimgcdcsInstance_t;

    /**
     * @brief Opaque Image type.
     */
    struct nvimgcdcsImage;

    /**
     * @brief Handle to opaque Image type.
     */
    typedef struct nvimgcdcsImage* nvimgcdcsImage_t;

    /**
     * @brief Opaque Code Stream type.
     */
    struct nvimgcdcsCodeStream;

    /**
     * @brief Handle to opaque Code Stream type.
     */
    typedef struct nvimgcdcsCodeStream* nvimgcdcsCodeStream_t;

    /**
     * @brief Opaque Parser type.
     */
    struct nvimgcdcsParser;

    /**
     * @brief Handle to opaque Parser type.
     */
    typedef struct nvimgcdcsParser* nvimgcdcsParser_t;

    /**
     * @brief Opaque Encoder type.
     */
    struct nvimgcdcsEncoder;

    /**
     * @brief Handle to opaque Encoder type.
     */
    typedef struct nvimgcdcsEncoder* nvimgcdcsEncoder_t;

    /**
     * @brief Opaque Decoder type.
     */
    struct nvimgcdcsDecoder;

    /**
     * @brief Handle to opaque Decoder type.
     */
    typedef struct nvimgcdcsDecoder* nvimgcdcsDecoder_t;

    /**
     * @brief Opaque Debug Messenger type.
     */
    struct nvimgcdcsDebugMessenger;

    /**
     * @brief Handle to opaque Debug Messenger type.
     */
    typedef struct nvimgcdcsDebugMessenger* nvimgcdcsDebugMessenger_t;

    /**
     * @brief Opaque Extension type.
     */
    struct nvimgcdcsExtension;

    /**
     * @brief Handle to opaque Extension type.
     */
    typedef struct nvimgcdcsExtension* nvimgcdcsExtension_t;

    /**
     * @brief Opaque Future type.
     */
    struct nvimgcdcsFuture;

    /**
     * @brief Handle to opaque Future type.
     */
    typedef struct nvimgcdcsFuture* nvimgcdcsFuture_t;

    /**
     * @brief Structure types supported by the nvImageCodecs API.
     * 
     * Each value corresponds to a particular structure with a type member and matching  structure name.
     */
    typedef enum
    {
        NVIMGCDCS_STRUCTURE_TYPE_PROPERTIES,
        NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_DEVICE_ALLOCATOR,
        NVIMGCDCS_STRUCTURE_TYPE_PINNED_ALLOCATOR,
        NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION,
        NVIMGCDCS_STRUCTURE_TYPE_REGION,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_IMAGE_PLANE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_IMAGE_INFO,
        NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_BACKEND,
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
        NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_EXECUTION_PARAMS,
        NVIMGCDCS_STRUCTURE_TYPE_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsStructureType_t;

    /**
     * @brief The nvImageCodecs properties.
     * 
     * @see nvimgcdcsGetProperties
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        uint32_t version;              /**< The nvImageCodecs library version. */
        uint32_t ext_api_version;      /**< The nvImageCodecs extension API version. */
        uint32_t cudart_version;       /**< The version of CUDA Runtime with which nvImageCodecs library was built. */

    } nvimgcdcsProperties_t;

    /** 
     * @brief Function type for device memory resource allocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer where to write pointer to allocated memory.
     * @param [in] size How many bytes to allocate.
     * @param [in] stream CUDA stream    
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcdcsDeviceMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for device memory deallocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer to memory buffer to be deallocated.
     *                 If NULL, the operation must do nothing, successfully.
     * @param [in] size How many bytes was allocated (size passed during allocation).
     * @param [in] stream CUDA stream   
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcdcsDeviceFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for host pinned memory resource allocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer where to write pointer to allocated memory.
     * @param [in] size How many bytes to allocate.
     * @param [in] stream CUDA stream    
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcdcsPinnedMalloc_t)(void* ctx, void** ptr, size_t size, cudaStream_t stream);

    /** 
     * @brief Function type for host pinned memory deallocation.
     *
     * @param [in] ctx Pointer to user context.
     * @param [in] ptr Pointer to memory buffer to be deallocated.
     *                 If NULL, the operation must do nothing, successfully.
     * @param [in] size How many bytes was allocated (size passed during allocation). 
     * @param [in] stream CUDA stream   
     * @returns They will return 0 in case of success, and non-zero otherwise
     */
    typedef int (*nvimgcdcsPinnedFree_t)(void* ctx, void* ptr, size_t size, cudaStream_t stream);

    /**
     * @brief Device memory allocator.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;         /**< Is the type of this structure. */
        void* next;                            /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsDeviceMalloc_t device_malloc; /**< Allocate memory on the device. */
        nvimgcdcsDeviceFree_t device_free;     /**< Frees memory on the device.*/
        void* device_ctx;                      /**< When invoking the allocators, this context will 
                                                    be pass as input to allocator functions.*/
        size_t device_mem_padding;             /**< Any device memory allocation 
                                                    would be padded to the multiple of specified number of bytes */
    } nvimgcdcsDeviceAllocator_t;

    /** 
     * @brief Host pinned memory allocator. 
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;         /**< Is the type of this structure. */
        void* next;                            /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsPinnedMalloc_t pinned_malloc; /**< Allocate host pinned memory: memory directly 
                                                    accessible by both CPU and cuda-enabled GPU. */
        nvimgcdcsPinnedFree_t pinned_free;     /**< Frees host pinned memory.*/
        void* pinned_ctx;                      /**< When invoking the allocators, this context will
                                                    be pass as input to allocator functions.*/
        size_t pinned_mem_padding;             /**< Any pinned host memory allocation
                                                    would be padded to the multiple of specified number of bytes */
    } nvimgcdcsPinnedAllocator_t;

    /** 
     * @brief The return status codes of the nvImageCodecs API
     */
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
        NVIMGCDCS_STATUS_EXTENSION_NOT_INITIALIZED = 11,
        NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER = 12,
        NVIMGCDCS_STATUS_EXTENSION_BAD_CODE_STREAM = 13,
        NVIMGCDCS_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED = 14,
        NVIMGCDCS_STATUS_EXTENSION_ALLOCATOR_FAILURE = 15,
        NVIMGCDCS_STATUS_EXTENSION_ARCH_MISMATCH = 16,
        NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR = 17,
        NVIMGCDCS_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED = 18,
        NVIMGCDCS_STATUS_EXTENSION_INCOMPLETE_BITSTREAM = 19,
        NVIMGCDCS_STATUS_EXTENSION_EXECUTION_FAILED = 20,
        NVIMGCDCS_STATUS_EXTENSION_CUDA_CALL_ERROR = 21,
        NVIMGCDCS_STATUS_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsStatus_t;

    /**
     * @brief Describes type sample of data. 
     * 
     * Meaning of bits:
     * 0 bit      -> 0 - unsigned, 1- signed
     * 1..7 bits  -> define type
     * 8..15 bits -> type bitdepth
     * 
     */
    typedef enum
    {
        NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN = 0,
        NVIMGCDCS_SAMPLE_DATA_TYPE_INT8 = 0x0801,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 = 0x0802,
        NVIMGCDCS_SAMPLE_DATA_TYPE_INT16 = 0x1003,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 = 0x1004,
        NVIMGCDCS_SAMPLE_DATA_TYPE_INT32 = 0x2005,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32 = 0x2006,
        NVIMGCDCS_SAMPLE_DATA_TYPE_INT64 = 0x4007,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64 = 0x4008,
        NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16 = 0x1009,
        NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 = 0x200B,
        NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64 = 0x400D,
        NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED = -1,
        NVIMGCDCS_SAMPLE_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsSampleDataType_t;

    /** 
     * @brief Chroma subsampling.
    */
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
        NVIMGCDCS_SAMPLING_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsChromaSubsampling_t;

    /**
     * @brief Provides information how color components are matched to channels in given order and channels are matched to planes.
     */
    typedef enum
    {
        NVIMGCDCS_SAMPLEFORMAT_UNKNOWN = 0,
        NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED = 1, //**< unchanged planar */
        NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED = 2, //**< unchanged interleaved */
        NVIMGCDCS_SAMPLEFORMAT_P_RGB = 3,       //**< planar RGB */
        NVIMGCDCS_SAMPLEFORMAT_I_RGB = 4,       //**< interleaved RGB */
        NVIMGCDCS_SAMPLEFORMAT_P_BGR = 5,       //**< planar BGR */
        NVIMGCDCS_SAMPLEFORMAT_I_BGR = 6,       //**< interleaved BGR */
        NVIMGCDCS_SAMPLEFORMAT_P_Y = 7,         //**< Y component only */
        NVIMGCDCS_SAMPLEFORMAT_P_YUV = 9,       //**< YUV planar format */
        NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED = -1,
        NVIMGCDCS_SAMPLEFORMAT_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsSampleFormat_t;

    /** 
     * @brief Defines color specification.
     */
    typedef enum
    {
        NVIMGCDCS_COLORSPEC_UNKNOWN = 0,
        NVIMGCDCS_COLORSPEC_UNCHANGED = NVIMGCDCS_COLORSPEC_UNKNOWN,
        NVIMGCDCS_COLORSPEC_SRGB = 1,
        NVIMGCDCS_COLORSPEC_GRAY = 2,
        NVIMGCDCS_COLORSPEC_SYCC = 3,
        NVIMGCDCS_COLORSPEC_CMYK = 4,
        NVIMGCDCS_COLORSPEC_YCCK = 5,
        NVIMGCDCS_COLORSPEC_UNSUPPORTED = -1,
        NVIMGCDCS_COLORSPEC_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsColorSpec_t;

    /** 
     *  @brief Defines orientation of an image.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        int rotated;                   /**< Rotation angle in degrees (clockwise). Only multiples of 90 are allowed. */
        int flip_x;                    /**< Flip horizontal 0 or 1*/
        int flip_y;                    /**< Flip vertical 0 or 1*/
    } nvimgcdcsOrientation_t;

    /**
     * @brief Defines plane of an image.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;         /**< Is the type of this structure. */
        void* next;                            /**< Is NULL or a pointer to an extension structure type. */

        uint32_t width;                        /**< Plane width. First plane defines width of image. */
        uint32_t height;                       /**< Plane height. First plane defines height of image.*/
        size_t row_stride;                     /**< Number of bytes need to offset to next row of plane. */
        uint32_t num_channels;                 /**< Number of channels. Color components, are always first
                                                    but there can be more channels than color components.*/
        nvimgcdcsSampleDataType_t sample_type; /**< Sample data type. @see  nvimgcdcsSampleDataType_t */
        uint8_t precision;                     /**< Value 0 means that precision is equal to sample type bitdepth */
    } nvimgcdcsImagePlaneInfo_t;

    /**
     * @brief Defines region of an image.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;    /**< Is the type of this structure. */
        void* next;                       /**< Is NULL or a pointer to an extension structure type. */

        int ndim;                         /**< Number of dimensions, 0 value means no region. */
        int start[NVIMGCDCS_MAX_NUM_DIM]; /**< Region start position at the particular dimension. */
        int end[NVIMGCDCS_MAX_NUM_DIM];   /**< Region end position at the particular dimension. */
    } nvimgcdcsRegion_t;

    /**
     * @brief Defines buffer kind in which image data is stored.
     */
    typedef enum
    {
        NVIMGCDCS_IMAGE_BUFFER_KIND_UNKNOWN = 0,
        NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE = 1, /**< GPU-accessible with planes in pitch-linear layout. */
        NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST = 2,   /**< Host-accessible with planes in pitch-linear layout. */
        NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED = -1,
        NVIMGCDCS_IMAGE_BUFFER_KIND_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsImageBufferKind_t;

    /**
     * @brief Defines information about an image.
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type;                   /**< Is the type of this structure. */
        void* next;                                      /**< Is NULL or a pointer to an extension structure type. */

        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];  /**< Information about codec used. Only valid when used with code stream. */

        nvimgcdcsColorSpec_t color_spec;                 /**< Image color specification. */
        nvimgcdcsChromaSubsampling_t chroma_subsampling; /**< Image chroma subsampling. Only valid with chroma components. */
        nvimgcdcsSampleFormat_t sample_format; /**< Defines how color components are matched to channels in given order and channels
                                                    are matched to planes. */
        nvimgcdcsOrientation_t orientation;    /**< Image orientation. */
        nvimgcdcsRegion_t region;              /**< Region of interest. */

        uint32_t num_planes;                   /**< Number of image planes. */
        nvimgcdcsImagePlaneInfo_t plane_info[NVIMGCDCS_MAX_NUM_PLANES]; /**< Array with information about image planes. */

        void* buffer;                                                   /**< Pointer to buffer in which image data is stored. */
        size_t buffer_size;                                             /**< Size of buffer in which image data is stored. */
        nvimgcdcsImageBufferKind_t buffer_kind;                         /**< buffer kind in which image data is stored.*/

        cudaStream_t cuda_stream;                                       /**< CUDA stream to synchronize with */
    } nvimgcdcsImageInfo_t;

    /** 
     * @brief JPEG Encoding
     *  
     * Currently parseable JPEG encodings (SOF markers)
     * https://www.w3.org/Graphics/JPEG/itu-t81.pdf
     * Table B.1 Start of Frame markers
     */
    typedef enum
    {
        NVIMGCDCS_JPEG_ENCODING_UNKNOWN = 0x0,
        NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT = 0xc0,
        NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN = 0xc1,
        NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN = 0xc2,
        NVIMGCDCS_JPEG_ENCODING_LOSSLESS_HUFFMAN = 0xc3,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN = 0xc5,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN = 0xc6,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN = 0xc7,
        NVIMGCDCS_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS = 0xc8,
        NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC = 0xc9,
        NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC = 0xca,
        NVIMGCDCS_JPEG_ENCODING_LOSSLESS_ARITHMETIC = 0xcb,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC = 0xcd,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC = 0xce,
        NVIMGCDCS_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC = 0xcf,
        NVIMGCDCS_JPEG_ENCODING_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsJpegEncoding_t;

    /** 
     * @brief Defines image information related to JPEG format.
     * 
     * This structure extends information provided in nvimgcdcsImageInfo_t
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type;    /**< Is the type of this structure. */
        void* next;                       /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsJpegEncoding_t encoding; /**< JPEG encoding type. */
    } nvimgcdcsJpegImageInfo_t;

    /**
     * @brief Defines decoding/encoding backend kind.
     */
    typedef enum
    {
        NVIMGCDCS_BACKEND_KIND_CPU_ONLY = 1,       /**< Decoding/encoding is executed only on CPU. */
        NVIMGCDCS_BACKEND_KIND_GPU_ONLY = 2,       /**< Decoding/encoding is executed only on GPU. */
        NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU = 3, /**< Decoding/encoding is executed on both CPU and GPU.*/
        NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY = 4,    /**< Decoding/encoding is executed on GPU dedicated hardware engine. */
    } nvimgcdcsBackendKind_t;

    /** 
     * @brief Defines decoding/encoding backend parameters.
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        /** 
         * Fraction of the batch items that will be picked by this backend.
         * The remaining items will be marked as "saturated" status and will be picked by the next backend.
         * This is just a hint and a particular implementation can choose to ignore it. */
        float load_hint;
    } nvimgcdcsBackendParams_t;

    /** 
     * @brief Defines decoding/encoding backend.
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type;   /**< Is the type of this structure. */
        void* next;                      /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsBackendKind_t kind;     /**< Decoding/encoding backend kind. */
        nvimgcdcsBackendParams_t params; /**< Decoding/encoding backend parameters. */
    } nvimgcdcsBackend_t;

    /**
     * @brief Processing status bitmask for decoding or encoding . 
     */
    typedef enum
    {
        NVIMGCDCS_PROCESSING_STATUS_UNKNOWN = 0x0,
        NVIMGCDCS_PROCESSING_STATUS_SUCCESS = 0x1,                 /**< Processing finished with success. */
        NVIMGCDCS_PROCESSING_STATUS_SATURATED = 0x2,               /**< Decoder/encoder could potentially process 
                                                                          image but is saturated. 
                                                                          @see nvimgcdcsBackendParams_t load_hint. */

        NVIMGCDCS_PROCESSING_STATUS_FAIL = 0x3,                    /**< Processing failed because unknown reason. */
        NVIMGCDCS_PROCESSING_STATUS_IMAGE_CORRUPTED = 0x7,         /**< Processing failed because compressed image stream is corrupted. */
        NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED = 0xb,       /**< Processing failed because codec is unsupported */
        NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED = 0x13,    /**< Processing failed because no one from allowed
                                                                          backends is supported. */
        NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED = 0x23,   /**< Processing failed because codec encoding is unsupported. */
        NVIMGCDCS_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED = 0x43, /**< Processing failed because image resolution is unsupported. */
        NVIMGCDCS_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED =
            0x83, /**< Processing failed because some feature of compressed stream is unsupported */

        //These values below describe cases when processing could be possible but with different image format or parameters
        NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED = 0x5,     /**< Color specification unsupported. */
        NVIMGCDCS_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED = 0x9,    /**< Apply orientation was enabled but it is unsupported. */
        NVIMGCDCS_PROCESSING_STATUS_ROI_UNSUPPORTED = 0x11,           /**< Decoding region of interest is unsupported. */
        NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED = 0x21,      /**< Selected unsupported chroma subsampling . */
        NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED = 0x41,   /**< Selected unsupported sample type. */
        NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED = 0x81, /**< Selected unsupported sample format. */
        NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED = 0x101,   /**< Unsupported number of planes to decode/encode. */
        NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED = 0x201, /**< Unsupported number of channels to decode/encode. */

        NVIMGCDCS_PROCESSING_STATUS_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsProcessingStatus;

    /**
     * @brief Processing status type which combine processing status bitmasks
    */
    typedef uint32_t nvimgcdcsProcessingStatus_t;

    /**
     * @brief Decode parameters
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        int apply_exif_orientation;    /**<  Apply exif orientation if available. Valid values 0 or 1. */
        int enable_roi;                /**<  Enables region of interest. Valid values 0 or 1. */

    } nvimgcdcsDecodeParams_t;

    /**
     * @brief Encode parameters
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        /** 
         * Float value of quality which interpretation depends of particular codec.
         * 
         * For JPEG codec it is expected to be integer values between 1 and 100, where 100 is the highest quality. Default value is 70.
         * @warning For JPEG2000 it is unsupported and target_psnr should be used instead.
         */
        float quality;

        /** 
         * Float value of target PSNR (Peak Signal to Noise Ratio)
         * 
         * @warning It is valid only for lossy encoding.
         * @warning It not supported by all codecs.
        */
        float target_psnr;
    } nvimgcdcsEncodeParams_t;

    /**
     * @brief Progression orders defined in the JPEG2000 standard.
     */
    typedef enum
    {
        NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP = 0, //**< Layer-Resolution-Component-Position progression order. */
        NVIMGCDCS_JPEG2K_PROG_ORDER_RLCP = 1, //**< Resolution-Layer-Component-Position progression order. */
        NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL = 2, //**< Resolution-Position-Component-Layer progression order. */
        NVIMGCDCS_JPEG2K_PROG_ORDER_PCRL = 3, //**< Position-Component-Resolution-Layer progression order. */
        NVIMGCDCS_JPEG2K_PROG_ORDER_CPRL = 4, //**< Component-Position-Resolution-Layer progression order. */
        NVIMGCDCS_JPEG2K_PROG_ORDER_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsJpeg2kProgOrder_t;

    /**
     * @brief JPEG2000 code stream type
     */
    typedef enum
    {
        NVIMGCDCS_JPEG2K_STREAM_J2K = 0, /**< Corresponds to the JPEG2000 code stream.*/
        NVIMGCDCS_JPEG2K_STREAM_JP2 = 1, /**< Corresponds to the .jp2 container.*/
        NVIMGCDCS_JPEG2K_STREAM_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsJpeg2kBitstreamType_t;

    /** 
     * @brief JPEG2000 Encode parameters
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;              /**< Is the type of this structure. */
        void* next;                                 /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsJpeg2kBitstreamType_t stream_type; /**< JPEG2000 code stream type. */
        nvimgcdcsJpeg2kProgOrder_t prog_order;      /**< JPEG2000 progression order. */
        uint32_t num_resolutions;                   /**< Number of resolutions. */
        uint32_t code_block_w;                      /**< Code block width. Allowed values 32, 64 */
        uint32_t code_block_h;                      /**< Code block height. Allowed values 32, 64 */
        int irreversible;                           /**< Sets whether or not to use irreversible encoding. Valid values 0 or 1. */
    } nvimgcdcsJpeg2kEncodeParams_t;

    /**
     * @brief JPEG Encode parameters
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        /**
         * Sets whether or not to use optimized Huffman. Valid values 0 or 1.
         * 
         * @note  Using optimized Huffman produces smaller JPEG bitstream sizes with the same quality, but with slower performance.
         */
        int optimized_huffman;
    } nvimgcdcsJpegEncodeParams_t;

    /**
     * @brief Bitmask specifying which severities of events cause a debug messenger callback
     */
    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_NONE = 0x00000000,
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE = 0x00000001,   /**< Diagnostic message useful for developers */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG = 0x00000010,   /**< Diagnostic message useful for developers */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO = 0x00000100,    /**< Informational message like the creation of a resource */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING = 0x00001000, /**< Message about behavior that is not necessarily an error,
                                                                but very likely a bug in your application */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR = 0x00010000,   /**< Message about behavior that is invalid and may cause
                                                                improper execution or result of operation (e.g. can't open file)
                                                                but not application */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL = 0x00100000,   /**< Message about behavior that is invalid and may cause crashes
                                                                and forcing to shutdown application */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ALL = 0x0FFFFFFF,     /**< Used in case filtering out by message severity */
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT =
            NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL,
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsDebugMessageSeverity_t;

    /**
     * @brief Bitmask specifying which category of events cause a debug messenger callback
     */
    typedef enum
    {
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_NONE = 0x00000000,
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL =
            0x00000001, /**< Some event has happened that is unrelated to the specification or performance */
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_VALIDATION = 0x00000010,  /**< Something has happened that indicates a possible mistake */
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_PERFORMANCE = 0x00000100, /**< Potential non-optimal use */
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL = 0x0FFFFFFF,         /**< Used in case filtering out by message category */
        NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsDebugMessageCategory_t;

    /**
     * @brief Describing debug message passed to debug callback function
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        const char* message;           /**< Null-terminated string detailing the trigger conditions */
        uint32_t internal_status_id;   /**< It is internal codec status id */
        const char* codec;             /**< Codec name if codec is rising message or NULL otherwise (e.g framework) */
        const char* codec_id;          /**< Codec id if codec is rising message or NULL otherwise */
        uint32_t codec_version;        /**< Codec version if codec is rising message or 0 otherwise */
    } nvimgcdcsDebugMessageData_t;

    /**
     * @brief Debug callback function type.
     * 
     * @param message_severity [in] Message severity
     * @param message_category [in] Message category
     * @param callback_data [in] Debug message data 
     * @param user_data [in] Pointer that was specified during the setup of the callback 
     * @returns 1 if message should not be passed further to other callbacks and 0 otherwise 
     */
    typedef int (*nvimgcdcsDebugCallback_t)(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageCategory_t message_category, const nvimgcdcsDebugMessageData_t* callback_data, void* user_data);

    /**
     * @brief Debug messenger description.
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type;          /**< Is the type of this structure. */
        void* next;                             /**< Is NULL or a pointer to an extension structure type. */

        uint32_t message_severity;              /**< Bitmask of message severity to listen for e.g. error or warning.  */
        uint32_t message_category;              /**< Bitmask of message category to listen for e.g. general or performance related. */
        nvimgcdcsDebugCallback_t user_callback; /**< Debug callback function */
        void* user_data;                        /**< Pointer to user data which will be passed back to debug callback function. */
    } nvimgcdcsDebugMessengerDesc_t;

    /** 
     * @brief Executor description.
     *
     * Codec plugins can use executor available via execution parameters to schedule execution of asynchronous task.  
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        const void* next;              /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< Executor instance pointer which will be passed back in functions */

        /**
         * @brief Schedule execution of asynchronous task.
         * 
         * @param instance [in] Pointer to nvimgcdcsExecutorDesc_t instance. 
         * @param device_id [in] Device id on which task will be executed.
         * @param sample_idx [in] Index of batch sample to process task on; It will be passed back as an argument in task function. 
         * @param task_context [in] Pointer to task context which will be passed back as an argument in task function.
         * @param task [in] Pointer to task function to schedule.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*launch)(void* instance, int device_id, int sample_idx, void* task_context,
            void (*task)(int thread_id, int sample_idx, void* task_context));

        /** 
         * @brief Gets number of threads.
         * 
         * @param instance [in] Pointer to nvimgcdcsExecutorDesc_t instance. 
         * @return Number of threads in executor.
        */
        int (*getNumThreads)(void* instance);
    } nvimgcdcsExecutorDesc_t;

    /** 
     * @brief Execution parameters
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;                /**< Is the type of this structure. */
        void* next;                                   /**< Is NULL or a pointer to an extension structure type. */

        nvimgcdcsDeviceAllocator_t* device_allocator; /**< Custom allocator for device memory */
        nvimgcdcsPinnedAllocator_t* pinned_allocator; /**< Custom allocator for pinned memory */
        int max_num_cpu_threads;                      /**< Max number of CPU threads in default executor 
                                                           (0 means default value equal to number of cpu cores) */
        nvimgcdcsExecutorDesc_t* executor;            /**< Points an executor. If NULL default executor will be used. 
                                                           @note At plugin level API it always points to executor, either custom or default. */
        int device_id;                                /**< Device id to process decoding on. It can be also specified 
                                                           using defines NVIMGCDCS_DEVICE_CURRENT or NVIMGCDCS_DEVICE_CPU_ONLY. */
        int pre_init;                                 /**< If true, all relevant resources are initialized at creation of the instance */
        int num_backends;                             /**< Number of allowed backends passed (if any) 
                                                           in backends parameter. For 0, all backends are allowed.*/
        const nvimgcdcsBackend_t* backends;           /**< Points a nvimgcdcsBackend_t array with defined allowed backends.
                                                           For nullptr, all backends are allowed. */
    } nvimgcdcsExecutionParams_t;

    /**
     * @brief Input/Output stream description.
     * 
     * This abstracts source or sink for code stream bytes.
     *  
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< I/O stream description instance pointer which will be passed back in functions */

        /**
         * @brief Reads all requested data from the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of read bytes.
         * @param buf [in]   Pointer to output buffer
         * @param bytes [in] Number of bytes to read
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*read)(void* instance, size_t* output_size, void* buf, size_t bytes);

        /**
         * @brief Writes all requested data to the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of written bytes.
         * @param buf [in]   Pointer to input buffer
         * @param bytes [in] Number of bytes to write
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*write)(void* instance, size_t* output_size, void* buf, size_t bytes);

        /**
         * @brief Writes one character to the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param output_size [in/out] Pointer to where to return number of written bytes.
         * @param ch [in] Character to write.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);

        /**
         * @brief Skips `count` objects in the stream
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param count [in] Number bytes to skip
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*skip)(void* instance, size_t count);

        /**
         * @brief Moves the read pointer in the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param offset  [in] Offset to move.
         * @param whence  [in] Beginning - SEEK_SET, SEEK_CUR or SEEK_END.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*seek)(void* instance, ptrdiff_t offset, int whence);

        /**
         * @brief Retrieves current position, in bytes from the beginning, in the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param offset  [in/out] Pointer where to return current position.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*tell)(void* instance, ptrdiff_t* offset);

        /**
         * @brief Retrieves the length, in bytes, of the stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param size  [in/out] Pointer where to return length of the stream.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*size)(void* instance, size_t* size);

        /**
         * @brief Provides expected bytes which are going to be written.  
         * 
         *  This function gives possibility to pre/re-allocate map function.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param bytes [in] Number of expected bytes which are going to be written.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*reserve)(void* instance, size_t bytes);

        /**
         * @brief Requests all data to be written to the output.
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*flush)(void* instance);

        /**
         * @brief Maps data into host memory  
         * 
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.
         * @param buffer [in/out] Points where to return pointer to mapped data. If data cannot be mapped, NULL will be returned.
         * @param offset [in] Offset in the stream to begin mapping.
         * @param size [in] Length of the mapping
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*map)(void* instance, void** buffer, size_t offset, size_t size);

        /**
         * @brief Unmaps previously mapped data
         *  
         * @param instance [in] Pointer to nvimgcdcsIoStreamDesc_t instance.         * 
         * @param buffer [in] Pointer to mapped data
         * @param size [in] Length of data to unmap 
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*unmap)(void* instance, void* buffer, size_t size);
    } nvimgcdcsIoStreamDesc_t;

    /**
     * @brief Code stream description.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;      /**< Is the type of this structure. */
        const void* next;                   /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                     /**< Code stream description instance pointer which will be passed back in functions */

        nvimgcdcsIoStreamDesc_t* io_stream; /**< I/O stream which works as a source or sink of code stream bytes */

        /**
         * @brief Retrieves image information.
         * 
         * @param instance [in] Pointer to nvimgcdcsCodeStreamDesc_t instance.
         * @param image_info [in/out] Points where to return image information.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* image_info);
    } nvimgcdcsCodeStreamDesc_t;

    /**
     * @brief Image description.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        const void* next;              /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< Image instance pointer which will be passed back in functions */

        /**
         * @brief Retrieves image info information.
         * 
         * @param instance [in] Pointer to nvimgcdcsImageDesc_t instance.
         * @param image_info [in/out] Points where to return image information.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* image_info);

        /**
         * @brief Informs that host side of processing of image is ready.
         * 
         * @param instance [in] Pointer to nvimgcdcsImageDesc_t instance.
         * @param processing_status [in] Processing status.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
         */
        nvimgcdcsStatus_t (*imageReady)(void* instance, nvimgcdcsProcessingStatus_t processing_status);
    } nvimgcdcsImageDesc_t;

    /**
     * @brief Parser description.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        const void* next;              /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< Parser description instance pointer which will be passed back in functions */
        const char* id;                /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec;             /**< Codec name e.g. jpeg2000 */

        /** 
         * @brief Checks whether parser can parse given code stream.
         * 
         * @param instance [in] Pointer to nvimgcdcsParserDesc_t instance.
         * @param result [in/out] Points where to return result of parsing check. Valid values 0 or 1.
         * @param code_stream [in] Code stream to parse check.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*canParse)(void* instance, int* result, nvimgcdcsCodeStreamDesc_t* code_stream);

        /**
         * Creates parser.
         * 
         * @param [in] Pointer to nvimgcdcsParserDesc_t instance.
         * @param [in/out] Points where to return handle to created parser.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsParser_t* parser);

        /** 
         * Destroys parser.
         * 
         * @param parser [in] Parser handle to destroy.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsParser_t parser);

        /**
         * @brief Parses given code stream and returns image information.
         * 
         * @param parser [in] Parser handle.
         * @param image_info [in/out] Points where to return image information.
         * @param code_stream [in] Code stream to parse.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*getImageInfo)(
            nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream);
    } nvimgcdcsParserDesc_t;

    /**
     * @brief Encoder description.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;       /**< Is the type of this structure. */
        const void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                      /**< Encoder description instance pointer which will be passed back in functions */
        const char* id;                      /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec;                   /**< Codec name e.g. jpeg2000 */
        nvimgcdcsBackendKind_t backend_kind; /**< What kind of backend this encoder is using */

        /**
         * @brief Creates encoder.
         * 
         * @param instance [in] Pointer to nvimgcdcsEncoderDesc_t instance.
         * @param encoder [in/out] Points where to return handle to created encoder.
         * @param exec_params [in] Points an execution parameters.
         * @param options [in] String with optional, space separated, list of parameters for encoders, in format 
         *                     "<encoder_id>:<parameter_name>=<parameter_value>".
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*create)(
            void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

        /** 
         * Destroys encoder.
         * 
         * @param encoder [in] Encoder handle to destroy.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsEncoder_t encoder);

        /**
         * @brief Checks whether encoder can encode given batch of images to code stream and with provided parameters.
         * 
         * @param encoder [in] Encoder handle to use for check.
         * @param status [in/out] Points to array of batch size and nvimgcdcsProcessingStatus_t type, where result will be returned.
         * @param images [in] Pointer to array of pointers of batch size with input images to check encoding.
         * @param code_streams [in/out] Pointer to array of pointers of batch size with output code streams to check encoding with.
         * @param batch_size [in] Number of items in batch  to check.
         * @param params [in] Encode parameters which will be used with check.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*canEncode)(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        /**
         * @brief Encode given batch of images to code streams with provided parameters.
         * 
         * @param encoder [in] Encoder handle to use for check.
         * @param images [in] Pointer to array of pointers of batch size with input images to encode.
         * @param code_streams [in/out] Pointer to array of pointers of batch size with ouput code streams.
         * @param batch_size [in] Number of items in batch  to encode.
         * @param params [in] Encode parameters.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*encode)(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams,
            int batch_size, const nvimgcdcsEncodeParams_t* params);
    } nvimgcdcsEncoderDesc_t;

    /**
     * Decoder description.
    */
    typedef struct
    {
        nvimgcdcsStructureType_t type;       /**< Is the type of this structure. */
        const void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                      /**< Decoder description instance pointer which will be passed back in functions */
        const char* id;                      /**< Codec named identifier e.g. nvJpeg2000 */
        const char* codec;                   /**< Codec name e.g. jpeg2000 */
        nvimgcdcsBackendKind_t backend_kind; /**< Backend kind */

        /**
         * @brief Creates decoder.
         * 
         * @param instance [in] Pointer to nvimgcdcsDecoderDesc_t instance.
         * @param encoder [in/out] Points where to return handle to created decoder.
         * @param exec_params [in] Points an execution parameters.
         * @param options [in] String with optional, space separated, list of parameters for decoders, in format 
         *                     "<encoder_id>:<parameter_name>=<parameter_value>".
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*create)(
            void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

        /** 
         * Destroys decoder.
         * 
         * @param decoder [in] Decoder handle to destroy.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsDecoder_t decoder);

        /**
         * @brief Checks whether decoder can decode given batch of code streams to images with provided parameters.
         * 
         * @param decoder [in] Decoder handle to use for check.
         * @param status [in/out] Points to array of batch size and nvimgcdcsProcessingStatus_t type, where result will be returned.
         * @param code_streams [in] Pointer to array of pointers of batch size with input code streams to check decoding.
         * @param images [in] Pointer to array of pointers of batch size with output images to check decoding.
         * @param batch_size [in] Number of items in batch to check.
         * @param params [in] Decode parameters which will be used with check.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*canDecode)(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        /**
         * @brief Decode given batch of code streams to images with provided parameters.
         * 
         * @param encoder [in] Decoder handle to use for decoding.
         * @param code_streams [in] Pointer to array of pointers of batch size with input code streams.
         * @param images [in/out] Pointer to array of pointers of batch size with output images.
         * @param batch_size [in] Number of items in batch  to encode.
         * @param params [in] Decode parameters.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*decode)(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images,
            int batch_size, const nvimgcdcsDecodeParams_t* params);
    } nvimgcdcsDecoderDesc_t;

    /**
     * @brief Defines decoder or encoder priority in codec.
     * 
     * For each codec there can be more decoders and encoders registered. Every decoder and encoder is registered with defined priority.
     * Decoding process starts with selecting highest priority decoder and checks whether it can decode particular code stream. In case
     * decoding could not be handled by selected decoder, there is fallback mechanism which selects next in priority decoder. There can be 
     * more decoders registered with the same priority. In such case decoders with the same priority are selected in order of registration.
     */
    typedef enum
    {
        NVIMGCDCS_PRIORITY_HIGHEST = 0,
        NVIMGCDCS_PRIORITY_VERY_HIGH = 100,
        NVIMGCDCS_PRIORITY_HIGH = 200,
        NVIMGCDCS_PRIORITY_NORMAL = 300,
        NVIMGCDCS_PRIORITY_LOW = 400,
        NVIMGCDCS_PRIORITY_VERY_LOW = 500,
        NVIMGCDCS_PRIORITY_LOWEST = 1000,
        NVIMGCDCS_PRIORITY_ENUM_FORCE_INT = INT32_MAX
    } nvimgcdcsPriority_t;

    /**
     * @brief Pointer to logging function.
     * 
     * @param instance [in] Plugin framework instance pointer
     * @param message_severity [in] Message severity e.g. error or warning.
     * @param message_category [in]  Message category e.g. general or performance related.
     * @param data [in] Debug message data i.e. message string, status, codec etc.
     */
    typedef nvimgcdcsStatus_t (*nvimgcdcsLogFunc_t)(void* instance, const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageCategory_t message_category, const nvimgcdcsDebugMessageData_t* data);

    /**
     * @brief Plugin Framework
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        const void* next;              /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< Plugin framework instance pointer which will be passed back in functions */
        const char* id;                /**< Plugin framework named identifier e.g. nvImageCodecs */
        uint32_t version;              /**< Plugin framework version. */
        uint32_t ext_api_version;      /**< The nvImageCodecs extension API version. */
        uint32_t cudart_version;       /**< The version of CUDA Runtime with which plugin framework was built. */
        nvimgcdcsLogFunc_t log;        /**< Pointer to logging function. @see nvimgcdcsLogFunc_t */

        /**
         * @brief Registers encoder plugin.
         * 
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to encoder description.
         * @param priority [in] Priority of encoder. @see nvimgcdcsPriority_t
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*registerEncoder)(void* instance, const nvimgcdcsEncoderDesc_t* desc, float priority);

        /**
         * @brief Unregisters encoder plugin.
         *
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to encoder description to unregister.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*unregisterEncoder)(void* instance, const nvimgcdcsEncoderDesc_t* desc);

        /**
         * @brief Registers decoder plugin.
         * 
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to decoder description.
         * @param priority [in] Priority of decoder. @see nvimgcdcsPriority_t
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*registerDecoder)(void* instance, const nvimgcdcsDecoderDesc_t* desc, float priority);

        /**
         * @brief Unregisters decoder plugin.
         *
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to decoder description to unregister.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*unregisterDecoder)(void* instance, const nvimgcdcsDecoderDesc_t* desc);

        /**
         * @brief Registers parser plugin.
         * 
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to parser description.
         * @param priority [in] Priority of decoder. @see nvimgcdcsPriority_t
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*registerParser)(void* instance, const nvimgcdcsParserDesc_t* desc, float priority);

        /**
         * @brief Unregisters parser plugin.
         *
         * @param instance [in] Pointer to nvimgcdcsFrameworkDesc_t instance.
         * @param desc [in] Pointer to parser description to unregister.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
         */
        nvimgcdcsStatus_t (*unregisterParser)(void* instance, const nvimgcdcsParserDesc_t* desc);

    } nvimgcdcsFrameworkDesc_t;

    /**
     * @brief Extension description
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type; /**< Is the type of this structure. */
        void* next;                    /**< Is NULL or a pointer to an extension structure type. */

        void* instance;                /**< Extension instance pointer which will be passed back in functions */
        const char* id;                /**< Extension named identifier e.g. nvjpeg_ext */
        uint32_t version;              /**< Extension version. Used when registering extension to check if there are newer.*/
        uint32_t ext_api_version;      /**< The version of nvImageCodecs extension API with which the extension was built. */

        /**
         * @brief Creates extension.
         * 
         * @param instance [in] Pointer to nvimgcdcsExtensionDesc_t instance.
         * @param extension [in/out] Points where to return handle to created extension. 
         * @param framework [in] Pointer to framework description which can be use to register plugins.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework);

        /** 
         * Destroys extension.
         * 
         * @param extension [in] Extension handle to destroy.
         * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
        */
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsExtension_t extension);
    } nvimgcdcsExtensionDesc_t;

    /**
     * @brief Extension module entry function type
     * 
     * @param ext_desc [in/out] Points a nvimgcdcsExtensionDesc_t handle in which the extension description is returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    typedef nvimgcdcsStatus_t (*nvimgcdcsExtensionModuleEntryFunc_t)(nvimgcdcsExtensionDesc_t* ext_desc);

    /**
     * @brief Extension shared module exported entry function.
     * 
     * @param ext_desc [in/out] Points a nvimgcdcsExtensionDesc_t handle in which the extension description is returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc);

    /**
     * @brief Provides nvImageCodecs library properties.
     * 
     * @param properties [in/out] Points a nvimgcdcsProperties_t handle in which the properties are returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsGetProperties(nvimgcdcsProperties_t* properties);

    /** 
     * @brief The nvImageCodecs library instance create information structure.
     */
    typedef struct
    {
        nvimgcdcsStructureType_t type;      /**< Is the type of this structure. */
        void* next;                         /**< Is NULL or a pointer to an extension structure type. */

        int load_builtin_modules;           /**< Load default modules. Valid values 0 or 1. */
        int load_extension_modules;         /**< Discover and load extension modules on start. Valid values 0 or 1. */
        const char* extension_modules_path; /**< There may be several paths separated by ':' on Linux or ';' on Windows */
        int default_debug_messenger;        /**< Create default debug messenger. Valid values 0 or 1. */
        uint32_t message_severity;          /**< Severity for default debug messenger */
        uint32_t message_category;          /**< Message category for default debug messenger */
    } nvimgcdcsInstanceCreateInfo_t;

    /**
     * @brief Creates an instance of the library using the input arguments.
     * 
     * @param instance [in/out] Points a nvimgcdcsInstance_t handle in which the resulting instance is returned.
     * @param create_info [in] Pointer to a nvimgcdcsInstanceCreateInfo_t structure controlling creation of the instance.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceCreate(nvimgcdcsInstance_t* instance, const nvimgcdcsInstanceCreateInfo_t* create_info);

    /**
     * @brief Destroys the nvImageCodecs library instance.
     * 
     * @param instance [in] The library instance handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsInstanceDestroy(nvimgcdcsInstance_t instance);

    /**
     * @brief Creates library extension.
     *  
     * @param instance [in] The library instance handle the extension will be used with.
     * @param extension [in/out] Points a nvimgcdcsExtension_t handle in which the resulting extension is returned.
     * @param extension_desc [in] Pointer to a nvimgcdcsExtensionDesc_t structure which defines extension to create.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsExtension_t* extension, nvimgcdcsExtensionDesc_t* extension_desc);

    /**
     * @brief Destroys library extension.
     * 
     * @param extension [in] The extension handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsExtensionDestroy(nvimgcdcsExtension_t extension);

    /**
     * @brief Creates a debug messenger.
     *  
     * @param instance [in] The library instance handle the messenger will be used with.
     * @param dbg_messenger [in/out] Points a nvimgcdcsDebugMessenger_t handle in which the resulting debug messenger is returned.
     * @param messenger_desc [in]  Pointer to nvimgcdcsDebugMessengerDesc_t structure which defines debug messenger to create.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessengerCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsDebugMessenger_t* dbg_messenger, const nvimgcdcsDebugMessengerDesc_t* messenger_desc);

    /**
     * @brief Destroys debug messenger.
     * 
     * @param dbg_messenger [in] The debug messenger handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDebugMessengerDestroy(nvimgcdcsDebugMessenger_t dbg_messenger);

    /**
     * @brief Waits for processing items to be finished.
     *  
     * @param future [in] Handle to future object created by decode or encode functions.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     * @warning Please note that when future is ready, it only means that all host work was done and it can be that
     *          some work was scheduled to be executed on device (depending on codec). To further synchronize work on 
     *          device, there is cuda_stream field available in nvimgcdcsImageInfo_t which can be used to specify 
     *          cuda_stream to synchronize with.
     * @see  nvimgcdcsImageInfo_t cuda_stream field.
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsFutureWaitForAll(nvimgcdcsFuture_t future);

    /**
     * @brief Destroys future.
     * 
     * @param future [in] The future handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsFutureDestroy(nvimgcdcsFuture_t future);

    /**
     * @brief Receives processing statuses of batch items scheduled for decoding or encoding 
     * 
     * @param future [in] The future handle returned by decode or encode function for given batch items.
     * @param processing_status [in/out] Points a nvimgcdcsProcessingStatus_t handle in which the processing statuses is returned.
     * @param size [in/out]  Points a size_t in which the size of processing statuses returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsFutureGetProcessingStatus(
        nvimgcdcsFuture_t future, nvimgcdcsProcessingStatus_t* processing_status, size_t* size);

    /**
     * @brief Creates Image which wraps sample buffer together with format information.
     * 
     * @param instance [in] The library instance handle the image will be used with.
     * @param image [in/out] Points a nvimgcdcsImage_t handle in which the resulting image is returned.
     * @param image_info [in] Points a nvimgcdcsImageInfo_t struct which describes sample buffer together with format.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const nvimgcdcsImageInfo_t* image_info);

    /**
     * @brief Destroys image.
     * 
     * @param image [in] The image handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageDestroy(nvimgcdcsImage_t image);

    /**
     * @brief Retrieves image information from provided opaque image object. 
     *  
     * @param image [in] The image handle to retrieve information from.
     * @param image_info [in/out] Points a nvimgcdcsImageInfo_t handle in which the image information is returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsImageGetImageInfo(nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info);

    /**
     * @brief Creates code stream which wraps file source of compressed data 
     *  
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcdcsCodeStream_t handle in which the resulting code stream is returned.
     * @param file_name [in] File name with compressed image data to wrap.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromFile(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const char* file_name);

    /**
     * @brief Creates code stream which wraps host memory source of compressed data.
     * 
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcdcsCodeStream_t handle in which the resulting code stream is returned.
     * @param data [in] Pointer to buffer with compressed data.
     * @param length [in] Length of compressed data in provided buffer.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const unsigned char* data, size_t length);

    /**
     * @brief Creates code stream which wraps file sink for compressed data with given format.
     * 
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcdcsCodeStream_t handle in which the resulting code stream is returned.
     * @param file_name [in] File name sink for compressed image data to wrap.
     * @param image_info [in] Points a nvimgcdcsImageInfo_t struct which describes output image format.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(
        nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const char* file_name, const nvimgcdcsImageInfo_t* image_info);

    /**
     * @brief Function type to resize and provide host buffer.
     * 
     * @param ctx [in] Pointer to context provided together with function.
     * @param req_size [in] Requested size of buffer.
     * @return Pointer to requested buffer.
     * 
     * @note This function can be called multiple times and requested size can be lower at the end so buffer can be shrinked.
     */
    typedef unsigned char* (*nvimgcdcsResizeBufferFunc_t)(void* ctx, size_t req_size);

    /**
     * @brief Creates code stream which wraps host memory sink for compressed data with given format.
     *  
     * @param instance  [in] The library instance handle the code stream will be used with.
     * @param code_stream [in/out] Points a nvimgcdcsCodeStream_t handle in which the resulting code stream is returned.
     * @param ctx [in] Pointer to user defined context with which get buffer function will be called back.
     * @param resize_buffer_func [in] Points a nvimgcdcsResizeBufferFunc_t function handle which will be used to resize and providing host output buffer.
     * @param image_info [in] Points a nvimgcdcsImageInfo_t struct which describes output image format.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream,
        void* ctx, nvimgcdcsResizeBufferFunc_t resize_buffer_func, const nvimgcdcsImageInfo_t* image_info);

    /**
     * @brief Destroys code stream.
     * 
     * @param code_stream [in] The code stream handle to destroy 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamDestroy(nvimgcdcsCodeStream_t code_stream);

    /**
     * @brief Retrieves compressed image information from code stream. 
     *  
     * @param code_stream [in] The code stream handle to retrieve information from.
     * @param image_info [in/out] Points a nvimgcdcsImageInfo_t handle in which the image information is returned.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes}
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(nvimgcdcsCodeStream_t code_stream, nvimgcdcsImageInfo_t* image_info);

    /**
     * @brief Creates generic image decoder.
     * 
     * @param instance  [in] The library instance handle the decoder will be used with.
     * @param decoder  [in/out] Points a nvimgcdcsDecoder_t handle in which the decoder is returned.
     * @param exec_params [in] Points an execution parameters.
     * @param options [in] String with optional space separated list of parameters for specific decoders in format 
     *                     "<decoder_id>:<parameter_name>=<parameter_value>". For example  "nvjpeg:fancy_upsampling=1"
     * @return nvimgcdcsStatus_t - An error code as specified in
     {
        @link nvimgcdcsStatus_t API Return Status Codes
     }
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    /**
     * @brief Destroys decoder.
     * 
     * @param decoder [in] The decoder handle to destroy
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDestroy(nvimgcdcsDecoder_t decoder);

    /**
     * @brief Checks if decoder can decode provided code stream to given output images with specified parameters.
     *  
     * @param decoder [in] The decoder handle to use for checks. 
     * @param streams [in] Pointer to input nvimgcdcsCodeStream_t array to check decoding with.
     * @param images [in] Pointer to output nvimgcdcsImage_t array to check decoding with.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcdcsDecodeParams_t struct to check decoding with.
     * @param processing_status [in/out] Points a nvimgcdcsProcessingStatus_t handle in which the processing statuses is returned.
     * @param force_format [in] Valid values 0 or 1. If 1 value, and high priority codec does not support provided format it will
     *                          fallback to lower priority codec for further checks. For 0 value, when high priority codec does not
     *                          support provided format or parameters but it can process input in general, it will stop check and
     *                          return processing status with flags which shows what format or parameters need to be changed to 
     *                          avoid fallback to lower priority codec. 
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCanDecode(nvimgcdcsDecoder_t decoder, const nvimgcdcsCodeStream_t* streams,
        const nvimgcdcsImage_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params,
        nvimgcdcsProcessingStatus_t* processing_status, int force_format);

    /**
     * @brief Decode batch of provided code streams to given output images with specified parameters.
     *  
     * @param decoder [in] The decoder handle to use for decoding. 
     * @param streams [in] Pointer to input nvimgcdcsCodeStream_t array to decode.
     * @param images [in] Pointer to output nvimgcdcsImage_t array to decode to.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcdcsDecodeParams_t struct to decode with.
     * @param future [in/out] Points a nvimgcdcsFuture_t handle in which the future is returned. 
     *               The future object can be used to waiting and getting processing statuses.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     * 
     * @see nvimgcdcsFutureGetProcessingStatus
     * @see nvimgcdcsFutureWaitForAll
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, const nvimgcdcsCodeStream_t* streams,
        const nvimgcdcsImage_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params, nvimgcdcsFuture_t* future);

    /**
     * @brief Creates generic image encoder.
     *  
     * @param instance [in] The library instance handle the encoder will be used with.
     * @param encoder [in/out] Points a nvimgcdcsEncoder_t handle in which the decoder is returned.
     * @param exec_params [in] Points an execution parameters.
     * @param options [in] String with optional, space separated, list of parameters for specific encoders, in format 
     *                     "<encoder_id>:<parameter_name>=<parameter_value>."
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCreate(
        nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    /**
     * @brief Destroys encoder.
     *  
     * @param encoder [in] The encoder handle to destroy
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderDestroy(nvimgcdcsEncoder_t encoder);

    /**
     * @brief Checks if encoder can encode provided images to given output code streams with specified parameters.
     *  
     * @param encoder [in] The encoder handle to use for checks. 
     * @param images [in] Pointer to input nvimgcdcsImage_t array to check encoding with.
     * @param streams [in] Pointer to output nvimgcdcsCodeStream_t array to check encoding with.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to nvimgcdcsEncodeParams_t struct to check decoding with.
     * @param processing_status [in/out] Points a nvimgcdcsProcessingStatus_t handle in which the processing statuses is returned.
     * @param force_format [in] Valid values 0 or 1. If 1 value, and high priority codec does not support provided format it will 
     *                          fallback to lower priority codec for further checks. For 0 value, when high priority codec does not
     *                          support provided format or parameters but it can process input in general, it will stop check and
     *                          return processing status with flags which shows what format or parameters need to be changed to 
     *                          avoid fallback to lower priority codec.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCanEncode(nvimgcdcsEncoder_t encoder, const nvimgcdcsImage_t* images,
        const nvimgcdcsCodeStream_t* streams, int batch_size, const nvimgcdcsEncodeParams_t* params,
        nvimgcdcsProcessingStatus_t* processing_status, int force_format);

    /**
     * @brief Encode batch of provided images to given output code streams with specified parameters.
     * 
     * @param encoder [in] The encoder handle to use for encoding. 
     * @param images [in] Pointer to input nvimgcdcsImage_t array to encode.
     * @param streams [in] Pointer to output nvimgcdcsCodeStream_t array to encode to.
     * @param batch_size [in] Batch size of provided code streams and images.
     * @param params [in] Pointer to  nvimgcdcsEncodeParams_t struct to encode with.
     * @param future  [in/out] Points a nvimgcdcsFuture_t handle in which the future is returned. 
     *                 The future object can be used to waiting and getting processing statuses.
     * @return nvimgcdcsStatus_t - An error code as specified in {@link nvimgcdcsStatus_t API Return Status Codes} 
     * 
     * @see nvimgcdcsFutureGetProcessingStatus
     * @see nvimgcdcsFutureWaitForAll
     */
    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, const nvimgcdcsImage_t* images,
        const nvimgcdcsCodeStream_t* streams, int batch_size, const nvimgcdcsEncodeParams_t* params, nvimgcdcsFuture_t* future);

#if defined(__cplusplus)
}
#endif

#endif
