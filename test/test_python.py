import os
import numpy as np
import cv2
import cupy as cp
import pytest as t
from nvidia import nvimgcodecs

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../resources"))

MAX_DIFF_THRESHOLD = 4


def compare_image(test_img, ref_img):
    diff = ref_img.astype(np.int32) - test_img.astype(np.int32)
    diff = np.absolute(diff)

    assert test_img.shape == ref_img.shape
    assert diff.max() <= MAX_DIFF_THRESHOLD


def compare_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        cp_test_img = cp.asarray(test_images[i])
        np_test_img = np.asarray(cp.asnumpy(cp_test_img))
        ref_img = cv2.cvtColor(ref_images[i], cv2.COLOR_BGR2RGB)
        ref_img = np.asarray(ref_img)
        compare_image(np_test_img, ref_img)
        

def compare_cv_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        test_img = np.asarray(test_images[i])
        ref_img = np.asarray(ref_images[i])
        compare_image(test_img, ref_img)


@t.mark.parametrize("backends", [None,
                                 [nvimgcodecs.Backend(nvimgcodecs.GPU_ONLY, load_hint=0.5), nvimgcodecs.Backend(
                                     nvimgcodecs.HYBRID_CPU_GPU), nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)],
    [nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)]])
@t.mark.parametrize("decode_data", [True, False])
@t.mark.parametrize(
    "input_img_file",
    [   "bmp/cat-111793_640.bmp",

        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_411.jpg",
        "jpeg/padlock-406986_640_420.jpg",
        "jpeg/padlock-406986_640_422.jpg",
        "jpeg/padlock-406986_640_440.jpg",
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg/ycck_colorspace.jpg",
        "jpeg/cmyk.jpg",
        "jpeg/cmyk-dali.jpg",
        "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_no_orientation.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",

        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-111793_640.jp2",
        "jpeg2k/tiled-cat-1046544_640.jp2",
        "jpeg2k/tiled-cat-111793_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
        "jpeg2k/cat-1245673_640-12bit.jp2",
     ]
)
def test_decode_single_image(tmp_path, input_img_file, decode_data, backends):
    if backends:
        decoder = nvimgcodecs.Decoder(backends = backends, options=":fancy_upsampling=1")
    else:
        decoder = nvimgcodecs.Decoder(options=":fancy_upsampling=1") 

    input_img_path = os.path.join(img_dir_path, input_img_file)
    if decode_data:
        with open(input_img_path, 'rb') as in_file:
            data = in_file.read()
            test_img = decoder.decode(data)
    else:
        test_img = decoder.decode(input_img_path)

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    compare_images([test_img], [ref_img])


@t.mark.parametrize("backends", [None,
                                 [nvimgcodecs.Backend(nvimgcodecs.GPU_ONLY, load_hint=0.5), nvimgcodecs.Backend(
                                     nvimgcodecs.HYBRID_CPU_GPU), nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)],
                                 [nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)]])
@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("decode_data", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [("bmp/cat-111793_640.bmp",

      "jpeg/padlock-406986_640_410.jpg",
      "jpeg/padlock-406986_640_411.jpg",
      "jpeg/padlock-406986_640_420.jpg",
      "jpeg/padlock-406986_640_422.jpg",
      "jpeg/padlock-406986_640_440.jpg",
      "jpeg/padlock-406986_640_444.jpg",
      "jpeg/padlock-406986_640_gray.jpg",
      "jpeg/ycck_colorspace.jpg",
      "jpeg/cmyk.jpg",
      "jpeg/cmyk-dali.jpg",
      "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

      "jpeg/exif/padlock-406986_640_horizontal.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
      "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
      "jpeg/exif/padlock-406986_640_no_orientation.jpg",
      "jpeg/exif/padlock-406986_640_rotate_180.jpg",
      "jpeg/exif/padlock-406986_640_rotate_270.jpg",
      "jpeg/exif/padlock-406986_640_rotate_90.jpg",

      "jpeg2k/cat-1046544_640.jp2",
      "jpeg2k/cat-1046544_640.jp2",
      "jpeg2k/cat-111793_640.jp2",
      "jpeg2k/tiled-cat-1046544_640.jp2",
      "jpeg2k/tiled-cat-111793_640.jp2",
      "jpeg2k/cat-111793_640-16bit.jp2",
      "jpeg2k/cat-1245673_640-12bit.jp2",
      "base/4k_lossless.jp2",
      "base/4k_lossless.jp2",
      "base/4k_lossless.jp2")
     ]
)
def test_decode_batch(tmp_path, input_images_batch, decode_data, backends, cuda_stream):
    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [cv2.imread(img, cv2.IMREAD_COLOR |
                             cv2.IMREAD_ANYDEPTH) for img in input_images]
    if backends:
        decoder = nvimgcodecs.Decoder(backends = backends, options=":fancy_upsampling=1")
    else:
        decoder = nvimgcodecs.Decoder(options=":fancy_upsampling=1") 
    if decode_data:
        data_list = []
        for img in input_images:
            with open(img, 'rb') as in_file:
                data = in_file.read()
                data_list.append(data)
        if cuda_stream:
            test_images = decoder.decode(
                data_list, cuda_stream=cuda_stream.ptr)
        else:
            test_images = decoder.decode(data_list)
    else:
        if cuda_stream:
            test_images = decoder.decode(
                input_images, cuda_stream=cuda_stream.ptr)
        else:
            test_images = decoder.decode(input_images)

    compare_images(test_images, ref_images)


@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_img_file",
    [
        "bmp/cat-111793_640.bmp",

        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_411.jpg",
        "jpeg/padlock-406986_640_420.jpg",
        "jpeg/padlock-406986_640_422.jpg",
        "jpeg/padlock-406986_640_440.jpg",
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg/ycck_colorspace.jpg",
        "jpeg/cmyk.jpg",
        "jpeg/cmyk-dali.jpg",
        "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_no_orientation.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",

        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-111793_640.jp2",
        "jpeg2k/tiled-cat-1046544_640.jp2",
        "jpeg2k/tiled-cat-111793_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
        "jpeg2k/cat-1245673_640-12bit.jp2",
    ]
)
def test_encode_single_image(tmp_path, input_img_file, encode_to_data, cuda_stream):
    encoder = nvimgcodecs.Encoder()

    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodecs.as_image(cp_ref_img)

    if encode_to_data:
        if cuda_stream:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True), cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True))
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + ".jp2")
        if cuda_stream:
            encoder.encode(output_img_path, nv_ref_img,
                           params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True), cuda_stream=cuda_stream.ptr)
        else:
            encoder.encode(output_img_path, nv_ref_img,
                           params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True))
        with open(output_img_path, 'rb') as in_file:
            test_encoded_img = in_file.read()

    test_img = cv2.imdecode(
        np.asarray(bytearray(test_encoded_img)), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    compare_image(np.asarray(test_img), np.asarray(ref_img))


@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [
        ("bmp/cat-111793_640.bmp",

         "jpeg/padlock-406986_640_410.jpg",
         "jpeg/padlock-406986_640_411.jpg",
         "jpeg/padlock-406986_640_420.jpg",
         "jpeg/padlock-406986_640_422.jpg",
         "jpeg/padlock-406986_640_440.jpg",
         "jpeg/padlock-406986_640_444.jpg",
         "jpeg/padlock-406986_640_gray.jpg",
         "jpeg/ycck_colorspace.jpg",
         "jpeg/cmyk.jpg",
         "jpeg/cmyk-dali.jpg",
         "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

         "jpeg/exif/padlock-406986_640_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
         "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
         "jpeg/exif/padlock-406986_640_no_orientation.jpg",
         "jpeg/exif/padlock-406986_640_rotate_180.jpg",
         "jpeg/exif/padlock-406986_640_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_rotate_90.jpg",

         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-111793_640.jp2",
         "jpeg2k/tiled-cat-1046544_640.jp2",
         "jpeg2k/tiled-cat-111793_640.jp2",
         "jpeg2k/cat-111793_640-16bit.jp2",
         "jpeg2k/cat-1245673_640-12bit.jp2",)
    ]
)
def test_encode_batch_image(tmp_path, input_images_batch, encode_to_data, cuda_stream):
    encoder = nvimgcodecs.Encoder()

    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR |
                                          cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = [nvimgcodecs.as_image(cp_ref_img) for cp_ref_img in cp_ref_images]

    if encode_to_data:
        if cuda_stream:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True), cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True))
    else:
        output_img_paths = [os.path.join(tmp_path, os.path.splitext(
            os.path.basename(img))[0] + ".jp2") for img in input_images]
        if cuda_stream:
            encoder.encode(output_img_paths, nv_ref_images,
                           params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True), cuda_stream=cuda_stream.ptr)
        else:
            encoder.encode(output_img_paths, nv_ref_images,
                           params=nvimgcodecs.EncodeParams(jpeg2k_reversible=True))
        test_encoded_images = []
        for out_img_path in output_img_paths:
            with open(out_img_path, 'rb') as in_file:
                test_encoded_img = in_file.read()
                test_encoded_images.append(test_encoded_img)
                
    test_decoded_images = [cv2.cvtColor(cv2.imdecode(
        np.asarray(bytearray(img)), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) for img in test_encoded_images]

    compare_cv_images(test_decoded_images, ref_images)
