#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    // Data
    struct nvimgcdcsDataDict;
    struct nvimgcdcsDataList;

    typedef struct nvimgcdcsDataDict* nvimgcdcsDataDict_t;
    typedef struct nvimgcdcsDataList* nvimgcdcsDataList_t;

    typedef enum
    {
        NVIMGCDCS_DATA_NULL,
        NVIMGCDCS_DATA_BOOLEAN,
        NVIMGCDCS_DATA_INT,
        NVIMGCDCS_DATA_DOUBLE,
        NVIMGCDCS_DATA_STRING,
        NVIMGCDCS_DATA_LIST,
        NVIMGCDCS_DATA_DICT
    } nvimgcdcsDataType_t;
#if 0
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_create(nvimgcdcsDataDict_t* data);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_destroy(nvimgcdcsDataDict_t data);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_clear(nvimgcdcsDataDict_t data);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_erase(nvimgcdcsDataDict_t data, const char *name);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_count(nvimgcdcsDataList_t list, size_t size);

nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_string(nvimgcdcsDataDict_t data, const char *name, const char *val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_int(nvimgcdcsDataDict_t data, const char *name, long long val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_double(nvimgcdcsDataDict_t data, const char *name, double val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_bool(nvimgcdcsDataDict_t data, const char *name, bool val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_dict(nvimgcdcsDataDict_t data, const char *name, nvimgcdcsDataDict_t val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_set_list(nvimgcdcsDataDict_t data, const char *name, nvimgcdcsDataList_t val);

nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_item_type(nvimgcdcsDataDict_t *data, const char *name, nvimgcdcsDataType_t* type);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_string(nvimgcdcsDataDict_t *data, const char *name, const char* s);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_int(nvimgcdcsDataDict_t *data, const char *name, long long* i);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_double(nvimgcdcsDataDict_t *data, const char *name, double* d);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_bool(nvimgcdcsDataDict_t *data, const char *name, bool* b);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_dict(nvimgcdcsDataDict_t *data, const char *name, nvimgcdcsDataDict_t dict);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_dict_get_list(nvimgcdcsDataDict_t *data, const char *name, nvimgcdcsDataList_t list);

nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_create(nvimgcdcsDataList_t * list);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_destroy(nvimgcdcsDataList_t list);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_clear(nvimgcdcsDataDict_t data);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_erase(nvimgcdcsDataList_t list, size_t idx);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_count(nvimgcdcsDataList_t list, size_t size);

nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_string(nvimgcdcsDataList_t data, const char val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_int(nvimgcdcsDataList_t data, long long val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_double(nvimgcdcsDataList_t data, double val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_bool(nvimgcdcsDataList_t data, bool val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_dict(nvimgcdcsDataList_t list, nvimgcdcsDataDict_t val);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_push_back_list(nvimgcdcsDataList_t list, nvimgcdcsDataList_t val);

nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_item_type(nvimgcdcsDataList_t *data, size_t idx, nvimgcdcsDataType_t* type);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_string(nvimgcdcsDataList_t *data, const char *name, const char* s);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_int(nvimgcdcsDataList_t *data, const char *name, long long* i);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_double(nvimgcdcsDataList_t *data, const char *name, double* d);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_bool(nvimgcdcsDataList_t *data, const char *name, bool* b);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_dict(nvimgcdcsDataList_t list, size_t idx, nvimgcdcsDataDict_t dict);
nvimgcdcsStatus_t NVIMGCDCSAPI nvimgcdcs_data_list_get_list(nvimgcdcsDataList_t list, size_t idx, nvimgcdcsDataList_t out_array);
#endif

#if defined(__cplusplus)
  }
#endif