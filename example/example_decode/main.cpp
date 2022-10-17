#include <nvimgcodecs.h>

int main(int argc, const char * argv[])
{    
    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;
    
    nvimgcdcsInstanceCreate(&instance, instance_create_info);
    nvimgcdcsCodeStream_t code_stream;
    const char* input_file = "input.j2k";
   // nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_file);

    //nvimgcdcsCodeStreamDestroy(code_stream);
    nvimgcdcsInstanceDestroy(instance);
    
    return 0;    
}