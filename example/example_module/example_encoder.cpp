
#include <nvimgcdcs_module.h>
#include <string>
#include <vector>
#include "exceptions.h"

template <typename D, int SAMPLE_FORMAT = NVIMGCDCS_SAMPLEFORMAT_P_RGB>
int writeBMP(nvimgcdcsIoStreamDesc_t io_stream, const D* d_chanR, size_t pitchR, const D* d_chanG,
    size_t pitchG, const D* d_chanB, size_t pitchB, int width, int height, uint8_t precision,
    bool verbose)
{

    unsigned int headers[13];
    FILE* outfile;
    int extrabytes;
    int paddedsize;
    int x;
    int y;
    int n;
    int red, green, blue;
    std::vector<D> vchanR(height * width);
    std::vector<D> vchanG(height * width);
    std::vector<D> vchanB(height * width);
    D* chanR = vchanR.data();
    D* chanG = vchanG.data();
    D* chanB = vchanB.data();

    if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {

        CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width * sizeof(D), d_chanR, pitchR,
            width * sizeof(D), height, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width * sizeof(D), d_chanG, pitchG,
            width * sizeof(D), height, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width * sizeof(D), d_chanB, pitchB,
            width * sizeof(D), height, cudaMemcpyDeviceToHost));
    } else if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
        vchanR.resize(height * width * 3);
        chanR = vchanR.data();

        CHECK_CUDA(cudaMemcpy(
            chanR, d_chanR, (size_t)width * height * 3 * sizeof(D), cudaMemcpyDeviceToHost));
    }
    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

    headers[7]  = 0;          // biCompression
    headers[8]  = paddedsize; // biSizeImage
    headers[9]  = 0;          // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    size_t written_size;
    std::string bm("BM");
    io_stream->write(io_stream->instance, &written_size, static_cast<void*>(bm.data()), 2);

    for (n = 0; n <= 5; n++) {
        io_stream->putc(io_stream->instance, &written_size, headers[n] & 0x000000FF);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x0000FF00) >> 8);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x00FF0000) >> 16);
        io_stream->putc(
            io_stream->instance, &written_size, (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.
    io_stream->putc(io_stream->instance, &written_size, 1);
    io_stream->putc(io_stream->instance, &written_size, 0);
    io_stream->putc(io_stream->instance, &written_size, 24);
    io_stream->putc(io_stream->instance, &written_size, 0);

    for (n = 7; n <= 12; n++) {
        io_stream->putc(io_stream->instance, &written_size, headers[n] & 0x000000FF);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x0000FF00) >> 8);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x00FF0000) >> 16);
        io_stream->putc(
            io_stream->instance, &written_size, (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    if (verbose && precision > 8) {
        std::cout << "BMP write - truncating " << (int)precision << " bit data to 8 bit"
                  << std::endl;
    }

    //
    // Headers done, now write the data...
    //
    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++) {

            if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
                red   = chanR[y * width + x];
                green = chanG[y * width + x];
                blue  = chanB[y * width + x];
            } else if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
                red   = chanR[3 * (y * width + x)];
                green = chanR[3 * (y * width + x) + 1];
                blue  = chanR[3 * (y * width + x) + 2];
            }
            int scale = precision - 8;
            if (scale > 0) {
                red   = ((red >> scale) + ((red >> (scale - 1)) % 2));
                green = ((green >> scale) + ((green >> (scale - 1)) % 2));
                blue  = ((blue >> scale) + ((blue >> (scale - 1)) % 2));
            }

            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            io_stream->putc(io_stream->instance, &written_size, blue);
            io_stream->putc(io_stream->instance, &written_size, green);
            io_stream->putc(io_stream->instance, &written_size, red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++) {
                io_stream->putc(io_stream->instance, &written_size, 0);
            }
        }
    }
    return 0;
}

static nvimgcdcsEncoderStatus_t example_encoder_can_encode(void* instance, bool* result,
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params)
{
    *result = std::string(params->codec) == "bmp" /*&& (NVIMGCDCS_SAMPLEFORMAT_P_RGB)*/;

    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_encoder_create(
    void* instance, nvimgcdcsEncoder_t* encoder, nvimgcdcsEncodeParams_t* params)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_encoder_destroy(nvimgcdcsEncoder_t encoder)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_create_encode_state(
    nvimgcdcsEncoder_t decoder, nvimgcdcsEncodeState_t* encode_state)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

nvimgcdcsEncoderStatus_t example_destroy_encde_state(nvimgcdcsEncodeState_t encode_state)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_encoder_encode(nvimgcdcsEncoder_t encoder,
    nvimgcdcsEncodeState_t encode_state, nvimgcdcsCodeStreamDesc_t code_stream,
    nvimgcdcsImageDesc_t image, nvimgcdcsEncodeParams_t* params)
{
    nvimgcdcsImageInfo_t image_info;
    image->getImageInfo(image->instance, &image_info);
    unsigned char* dev_image_buffer;
    size_t size;
    image->getDeviceBuffer(image->instance, reinterpret_cast<void**>(&dev_image_buffer), &size);
    if (NVIMGCDCS_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
        writeBMP<unsigned char, NVIMGCDCS_SAMPLEFORMAT_I_RGB>(code_stream->io_stream,
            (unsigned char*)dev_image_buffer, image_info.component_info[0].pitch_in_bytes, NULL, 0,
            NULL, 0, image_info.image_width, image_info.image_height, 8, true);
    } else {
        writeBMP<unsigned char>(code_stream->io_stream, (unsigned char*)dev_image_buffer,
            image_info.component_info[0].pitch_in_bytes,
            (unsigned char*)dev_image_buffer +
                image_info.component_info[0].pitch_in_bytes * image_info.image_height,
            image_info.component_info[1].pitch_in_bytes,
            (unsigned char*)dev_image_buffer +
                +image_info.component_info[0].pitch_in_bytes * image_info.image_height +
                image_info.component_info[1].pitch_in_bytes * image_info.image_height,
            image_info.component_info[2].pitch_in_bytes, image_info.image_width,
            image_info.image_height, 8, true);
    }
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsEncoderDesc example_encoder = {
    NULL,               // instance    
    "example_encoder",  //id
     0x00000100,        // version
    "bmp",              //  codec_type 
    example_encoder_can_encode,
    example_encoder_create,
    example_encoder_destroy, 
    example_create_encode_state, 
    example_destroy_encde_state,
    example_encoder_encode
};
// clang-format on    
