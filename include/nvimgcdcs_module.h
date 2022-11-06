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

    struct nvimgcdcsParser;
    typedef struct nvimgcdcsParser* nvimgcdcsParser_t;

    struct nvimgcdcsParseState;
    typedef struct nvimgcdcsParseState* nvimgcdcsParseState_t;
  
    struct nvimgcdcsInputStreamDesc
    {
        void* instance;

        nvimgcdcsStatus_t (*read)(
            void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*write)(
            void* instance, size_t* output_size, void* buf, size_t bytes);
        nvimgcdcsStatus_t (*putc)(void* instance, size_t* output_size, unsigned char ch);
        nvimgcdcsStatus_t (*skip)(void* instance, size_t count);
        nvimgcdcsStatus_t (*seek)(void* instance, size_t offset, int whence);
        nvimgcdcsStatus_t (*tell)(void* instance, size_t* offset);
        nvimgcdcsStatus_t (*size)(void* instance, size_t* size);
    };
    typedef struct nvimgcdcsInputStreamDesc* nvimgcdcsIoStreamDesc_t;

    struct nvimgcdcsCodeStreamDesc
    {
        void* instance;
        const char* codec;
        nvimgcdcsIoStreamDesc_t io_stream;
        nvimgcdcsParseState_t parse_state;
    };
    typedef struct nvimgcdcsCodeStreamDesc* nvimgcdcsCodeStreamDesc_t;

    struct nvimgcdcsImageDesc
    {
        void* instance;
        nvimgcdcsStatus_t (*getImageInfo)(void* instance, nvimgcdcsImageInfo_t* result);
        nvimgcdcsStatus_t (*getDeviceBuffer)(void* instance, void** buffer, size_t* size);
        nvimgcdcsStatus_t (*getHostBuffer)(void* instance, void** buffer, size_t* size);
    };
    typedef struct nvimgcdcsImageDesc* nvimgcdcsImageDesc_t;

    struct nvimgcdcsParserDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canParse)(
            void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream);
        nvimgcdcsStatus_t (*create)(void* instance, nvimgcdcsParser_t* parser);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsParser_t parser);

        nvimgcdcsStatus_t (*createParseState)(
            nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state);
        nvimgcdcsStatus_t (*destroyParseState)(nvimgcdcsParseState_t parse_state);

        nvimgcdcsStatus_t (*getImageInfo)(nvimgcdcsParser_t parser,
            nvimgcdcsImageInfo_t* result, nvimgcdcsCodeStreamDesc_t code_stream);
    };

    struct nvimgcdcsEncoderDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canEncode)(void* instance, bool* result,
            nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params);

        nvimgcdcsStatus_t (*create)(
            void* instance, nvimgcdcsEncoder_t* encoder, nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsEncoder_t encoder);

        nvimgcdcsStatus_t (*createEncodeState)(
            nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state);
        nvimgcdcsStatus_t (*destroyEncodeState)(nvimgcdcsEncodeState_t encode_state);

        nvimgcdcsStatus_t (*encode)(nvimgcdcsEncoder_t encoder,
            nvimgcdcsEncodeState_t encode_state, nvimgcdcsCodeStreamDesc_t code_stream,
            nvimgcdcsImageDesc_t image, nvimgcdcsEncodeParams_t* params);
    };

    typedef struct nvimgcdcsEncoderDesc nvimgcdcsEncoderDesc_t;

    struct nvimgcdcsDecoderDesc
    {
        void* instance; // plugin instance pointer which will be passed back in functions
        const char* id; // named identifier e.g. nvJpeg2000
        uint32_t version;
        const char* codec; // e.g. jpeg2000

        nvimgcdcsStatus_t (*canDecode)(void* instance, bool* result,
            nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params);

        nvimgcdcsStatus_t (*create)(
            void* instance, nvimgcdcsDecoder_t* decoder, nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t (*destroy)(nvimgcdcsDecoder_t decoder);

        nvimgcdcsStatus_t (*createDecodeState)(
            nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state);
        nvimgcdcsStatus_t (*destroyDecodeState)(nvimgcdcsDecodeState_t decode_state);

        nvimgcdcsStatus_t (*decode)(nvimgcdcsDecoder_t decoder,
            nvimgcdcsDecodeState_t decode_state, nvimgcdcsCodeStreamDesc_t code_stream,
            nvimgcdcsImageDesc_t image, nvimgcdcsDecodeParams_t* params);
    };
    
    typedef struct nvimgcdcsDecoderDesc nvimgcdcsDecoderDesc_t;

    struct nvimgcdcsFrameworkDesc
    {
        const char* id; // famework named identifier e.g. nvImageCodecs
        uint32_t version;
        void* instance;
        nvimgcdcsStatus_t (*registerEncoder)(
            void* instance, const struct nvimgcdcsEncoderDesc* desc);
        nvimgcdcsStatus_t (*registerDecoder)(
            void* instance, const struct nvimgcdcsDecoderDesc* desc);
        nvimgcdcsStatus_t (*registerParser)(void* instance, const struct nvimgcdcsParserDesc* desc);
    };

    typedef struct nvimgcdcsFrameworkDesc nvimgcdcsFrameworkDesc_t;

    typedef nvimgcdcsStatus_t(nvimgcdcsModuleLoad_t)(nvimgcdcsFrameworkDesc_t* framework);
    typedef uint32_t(nvimgcdcsModuleVersion_t)(void);

    NVIMGCDCSAPI uint32_t nvimgcdcsModuleVersion();
#define NVIMGCDCS_EXTENSION_MODULE()  \
    uint32_t nvimgcdcsModuleVersion() \
    {                                 \
        return NVIMGCDCS_VER;         \
    }

    NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsModuleLoad(nvimgcdcsFrameworkDesc_t* framework);
    NVIMGCDCSAPI void nvimgcdcsModuleUnload(void);

#if defined(__cplusplus)
}
#endif
