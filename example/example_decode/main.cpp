#include <nvimgcodecs.h>
#include <filesystem>
#include <iostream>
#include <cuda_runtime_api.h>
#define CHECK_CUDA(call)                                                                \
    {                                                                                   \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                         \
            return EXIT_FAILURE;                                                        \
        }                                                                               \
    }
// write bmp, input - RGB, device
template <typename D>
int writeBMP(const char* filename, const D* d_chanR, size_t pitchR, const D* d_chanG, size_t pitchG,
    const D* d_chanB, size_t pitchB, int width, int height, uint8_t precision, bool verbose)
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
    CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width * sizeof(D), d_chanR, pitchR, width * sizeof(D),
        height, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width * sizeof(D), d_chanG, pitchG, width * sizeof(D),
        height, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width * sizeof(D), d_chanB, pitchB, width * sizeof(D),
        height, cudaMemcpyDeviceToHost));

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

    if (!(outfile = fopen(filename, "wb"))) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.

    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);

    for (n = 7; n <= 12; n++) {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
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
            red   = chanR[y * width + x];
            green = chanG[y * width + x];
            blue  = chanB[y * width + x];

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

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++) {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

int main(int argc, const char* argv[])
{
    namespace fs = std::filesystem;
    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next             = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;

    nvimgcdcsInstanceCreate(&instance, instance_create_info);
    nvimgcdcsCodeStream_t code_stream;
    fs::path exe_path(argv[0]);
    fs::path input_file = fs::absolute(exe_path).parent_path() / fs::path("input.j2k");
    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_file.string().c_str());
    nvimgcdcsImageInfo_t image_info;
    nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
    std::cout << "Image info: " << std::endl;
    std::cout << "\t - width:" << image_info.image_width << std::endl;
    std::cout << "\t - height:" << image_info.image_height << std::endl;
    std::cout << "\t - components:" << image_info.num_components << std::endl;
    image_info.sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;//TODO
    int bytes_per_element        = 1; //TODO
   
    image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;

    nvimgcdcsDecodeParams_t decode_params;
    decode_params.backend.useGPU = true;
   
    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, code_stream, &decode_params);

    nvimgcdcsDecodeState_t decode_state;
    nvimgcdcsDecodeStateCreate(decoder, &decode_state);

    unsigned char* image_buffer;

     CHECK_CUDA(cudaMallocPitch((void**)&image_buffer, &image_info.pitch_in_bytes,
        image_info.image_width * bytes_per_element,
        image_info.image_height * image_info.num_components));

    size_t image_buffer_size =
        image_info.pitch_in_bytes * image_info.image_height * image_info.num_components;

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image, &image_info);
    nvimgcdcsImageSetDeviceBuffer(image, image_buffer, image_buffer_size);
    nvimgcdcsImageAttachDecodeState(image, decode_state);

    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);
    cudaDeviceSynchronize();

    writeBMP<unsigned char>("test.bmp", (unsigned char*)image_buffer,
        image_info.pitch_in_bytes, (unsigned char*)image_buffer +image_info.pitch_in_bytes * image_info.image_height , image_info.pitch_in_bytes,
        (unsigned char*)image_buffer +2*image_info.pitch_in_bytes * image_info.image_height  , image_info.pitch_in_bytes, image_info.image_width,  image_info.image_height, 8,
        true);

    nvimgcdcsImageDetachDecodeState(image);
    CHECK_CUDA(cudaFree(image_buffer));
    nvimgcdcsImageDestroy(image);
    nvimgcdcsDecodeStateDestroy(decode_state);
    nvimgcdcsDecoderDestroy(decoder);
    nvimgcdcsCodeStreamDestroy(code_stream);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}