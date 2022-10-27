#pragma once

#include <nvimgcdcs_version.h>
#include <nvimgcodecs.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#elif __GNUC__ >= 4
    #define EXPORT __attribute__((visibility("default")))
#else
    #define EXPORT
#endif

#ifndef NVIMGCDCSAPI
    #ifdef __cplusplus
        #define NVIMGCDCSAPI extern "C" EXPORT
    #else
        #define NVIMGCDCSAPI EXPORT
    #endif
#endif

#if defined(__cplusplus)
extern "C"
{
#endif

    struct nvimgcdcsData; // json like
    struct nvimgcdcsEncoder;
    struct nvimgcdcsDecoder;
    struct nvimgcdcsParser;
    struct nvimgcdcsEncoderDesc;
    struct nvimgcdcsDecoderDesc;
    struct nvimgcdcsEncodeState;
    struct nvimgcdcsDecodeState;
    struct nvimgcdcsParseState;
    typedef struct nvimgcdcsData nvimgcdcsData_t;
    typedef struct nvimgcdcsEncoder* nvimgcdcsEncoder_t;
    typedef struct nvimgcdcsDecoder* nvimgcdcsDecoder_t;
    typedef struct nvimgcdcsParser* nvimgcdcsParser_t;
    typedef struct nvimgcdcsEncoderDesc nvimgcdcsEncoderDesc_t;
    typedef struct nvimgcdcsDecoderDesc nvimgcdcsDecoderDesc_t;
    typedef struct nvimgcdcsEncodeState* nvimgcdcsEncodeState_t;
    typedef struct nvimgcdcsDecodeState* nvimgcdcsDecodeState_t;
    typedef struct nvimgcdcsParseState* nvimgcdcsParseState_t;

    typedef enum
    {
        NVIMGCDCS_FRAMEWORK_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_FRAMEWORK_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_FRAMEWORK_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsFrameworkStatus_t;


    struct nvimgcdcsFrameworkDesc
    {
        const char* id; // famework named identifier e.g. nvImageCodecs
        uint32_t version;
        void* instance;
        nvimgcdcsFrameworkStatus_t (*registerEncoder)(
            void* instance, const struct nvimgcdcsEncoderDesc* desc);
        nvimgcdcsFrameworkStatus_t (*registerDecoder)(
            void* instance, const struct nvimgcdcsDecoderDesc* desc);
        nvimgcdcsFrameworkStatus_t (*registerParser)(
            void* instance, const struct nvimgcdcsParserDesc* desc);
    };

    typedef struct nvimgcdcsFrameworkDesc nvimgcdcsFrameworkDesc_t;

    typedef enum
    {
        NVIMGCDCS_MODULE_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_MODULE_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_MODULE_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_MODULE_STATUS_MISSED_DEPENDENCIES          = 3,
        NVIMGCDCS_MODULE_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_MODULE_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_MODULE_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_MODULE_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_MODULE_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_MODULE_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsModuleStatus_t;

    typedef nvimgcdcsModuleStatus_t(nvimgcdcsModuleLoad_t)(nvimgcdcsFrameworkDesc_t* framework);
    typedef uint32_t(nvimgcdcsModuleVersion_t)(void);

    NVIMGCDCSAPI uint32_t nvimgcdcsModuleVersion();
    #define NVIMGCDCS_EXTENSION_MODULE() uint32_t nvimgcdcsModuleVersion()               \
    {                                                   \
        return NVIMGCDCS_VER;                           \
    }

    NVIMGCDCSAPI nvimgcdcsModuleStatus_t nvimgcdcsModuleLoad(nvimgcdcsFrameworkDesc_t* framework);
    NVIMGCDCSAPI void nvimgcdcsModuleUnload(void);

    typedef enum
    {
        NVIMGCDCS_PARSER_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_PARSER_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_PARSER_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_PARSER_STATUS_BAD_BITSTREAM                = 3,
        NVIMGCDCS_PARSER_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_PARSER_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_PARSER_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_PARSER_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_PARSER_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_PARSER_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsParserStatus_t;

    struct nvimgcdcsInputStreamDesc
    {
        void* instance;

        nvimgcdcsParserStatus_t (*read)(
            void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsParserStatus_t (*write)(
            void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsParserStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);
        nvimgcdcsParserStatus_t (*skip)(void* instance, size_t count);
        nvimgcdcsParserStatus_t (*seek)(void* instance, size_t offset, int whence);
        nvimgcdcsParserStatus_t (*tell)(void* instance, size_t* offset);
        nvimgcdcsParserStatus_t (*size)(void* instance, size_t* size);
    };
    typedef struct nvimgcdcsInputStreamDesc* nvimgcdcsIoStreamDesc_t;

    typedef enum
    {
        NVIMGCDCS_CODE_STREAM_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_CODE_STREAM_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_CODE_STREAM_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_CODE_STREAM_STATUS_BAD_BITSTREAM                = 3,
        NVIMGCDCS_CODE_STREAM_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_CODE_STREAM_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_CODE_STREAM_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_CODE_STREAM_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_CODE_STREAM_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_CODE_STREAM_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsCodeStreamStatus_t;

    struct nvimgcdcsCodeStreamDesc
    {
        void* instance;
        const char* codec;
        nvimgcdcsIoStreamDesc_t io_stream;
        nvimgcdcsParseState_t parse_state;
    };
    typedef struct nvimgcdcsCodeStreamDesc* nvimgcdcsCodeStreamDesc_t;

    typedef enum
    {
        NVIMGCDCS_IMAGE_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_IMAGE_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_IMAGE_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_IMAGE_STATUS_BAD_BITSTREAM                = 3,
        NVIMGCDCS_IMAGE_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_IMAGE_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_IMAGE_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_IMAGE_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_IMAGE_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_IMAGE_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsImageStatus_t;

    struct nvimgcdcsImageDesc
    {
        void* instance;
        nvimgcdcsImageStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* result);
        nvimgcdcsImageStatus_t (*getDeviceBuffer)(void* instance, void** buffer, size_t* size);
        nvimgcdcsImageStatus_t (*getHostBuffer)(void* instance, void** buffer, size_t* size);
    };
    typedef struct nvimgcdcsImageDesc* nvimgcdcsImageDesc_t;

    struct nvimgcdcsParserDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsParserStatus_t (*canParse)(
            void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream);
        nvimgcdcsParserStatus_t (*create)(void* instance, nvimgcdcsParser_t* parser);
        nvimgcdcsParserStatus_t (*destroy)(nvimgcdcsParser_t parser);

        nvimgcdcsParserStatus_t (*createParseState)(
            nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state);
        nvimgcdcsParserStatus_t (*destroyParseState)(nvimgcdcsParseState_t parse_state);

        nvimgcdcsParserStatus_t (*getImageInfo)(nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* result,
            nvimgcdcsCodeStreamDesc_t code_stream);
    };

    typedef enum
    {
        NVIMGCDCS_ENCODER_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_ENCODER_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_ENCODER_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_ENCODER_STATUS_BAD_DATA                     = 3,
        NVIMGCDCS_ENCODER_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_ENCODER_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_ENCODER_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_ENCODER_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_ENCODER_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_ENCODER_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsEncoderStatus_t;

    struct nvimgcdcsEncoderDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsEncoderStatus_t (*canEncode)(void* instance, bool* result,
            nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params);

        nvimgcdcsEncoderStatus_t (*create)(
            void* instance, nvimgcdcsEncoder_t* encoder, nvimgcdcsEncodeParams_t* params);
        nvimgcdcsEncoderStatus_t (*destroy)(nvimgcdcsEncoder_t encoder);

        nvimgcdcsEncoderStatus_t (*createEncodeState)(
            nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state);
        nvimgcdcsEncoderStatus_t (*destroyEncodeState)(nvimgcdcsEncodeState_t encode_state);

        nvimgcdcsEncoderStatus_t (*encode)(nvimgcdcsEncoder_t encoder,
            nvimgcdcsEncodeState_t encode_state, nvimgcdcsCodeStreamDesc_t code_stream,
            nvimgcdcsImageDesc_t image, nvimgcdcsEncodeParams_t* params);
    };

    typedef enum
    {
        NVIMGCDCS_DECODER_STATUS_SUCCESS                      = 0,
        NVIMGCDCS_DECODER_STATUS_NOT_INITIALIZED              = 1,
        NVIMGCDCS_DECODER_STATUS_INVALID_PARAMETER            = 2,
        NVIMGCDCS_DECODER_STATUS_BAD_BITSTREAM                = 3,
        NVIMGCDCS_DECODER_STATUS_NOT_SUPPORTED                = 4,
        NVIMGCDCS_DECODER_STATUS_ALLOCATOR_FAILURE            = 5,
        NVIMGCDCS_DECODER_STATUS_EXECUTION_FAILED             = 6,
        NVIMGCDCS_DECODER_STATUS_ARCH_MISMATCH                = 7,
        NVIMGCDCS_DECODER_STATUS_INTERNAL_ERROR               = 8,
        NVIMGCDCS_DECODER_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
    } nvimgcdcsDecoderStatus_t;

    struct nvimgcdcsDecoderDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsDecoderStatus_t (*canDecode)(void* instance, bool* result,
            nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params);

        nvimgcdcsDecoderStatus_t (*create)(
            void* instance, nvimgcdcsDecoder_t* decoder, nvimgcdcsDecodeParams_t* params);
        nvimgcdcsDecoderStatus_t (*destroy)(nvimgcdcsDecoder_t decoder);

        nvimgcdcsDecoderStatus_t (*createDecodeState)(
            nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
        nvimgcdcsDecoderStatus_t (*destroyDecodeState)(nvimgcdcsDecodeState_t decode_state);

        nvimgcdcsDecoderStatus_t (*decode)(nvimgcdcsDecoder_t decoder,
            nvimgcdcsDecodeState_t decode_state, nvimgcdcsCodeStreamDesc_t code_stream,
            nvimgcdcsImageDesc_t image, nvimgcdcsDecodeParams_t* params);
    };



#if defined(__cplusplus)
}
#endif
