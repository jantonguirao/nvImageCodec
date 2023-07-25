from __future__ import annotations
import os
import numpy as np
import cv2
import cupy as cp
import pytest as t
from nvidia import nvimgcodecs

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../resources"))

def is_fancy_upsampling_available():
    return nvimgcodecs.__cuda_version__ >= 12010

def get_default_decoder_options():
    return ":fancy_upsampling=1" if is_fancy_upsampling_available() else ":fancy_upsampling=0"

def get_max_diff_threshold():
    return 4 if is_fancy_upsampling_available() else 44

def compare_image(test_img, ref_img):
    diff = ref_img.astype(np.int32) - test_img.astype(np.int32)
    diff = np.absolute(diff)

    assert test_img.shape == ref_img.shape
    assert diff.max() <= get_max_diff_threshold()


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


def load_single_image(file_path: str, load_mode: str = None):
    """
    Loads a single image to de decoded.
    :param file_path: Path to file with the image to be loaded.
    :param load_mode: In what format the image shall be loaded:
                        "numpy"  - loading using `np.fromfile`,
                        "python" - loading using Python's `open`,
                        "path"   - loading skipped, image path will be returned.
    :return: Encoded image.
    """
    if load_mode == "numpy":
        return np.fromfile(file_path, dtype=np.uint8)
    elif load_mode == "python":
        with open(file_path, 'rb') as in_file:
            return in_file.read()
    elif load_mode == "path":
        return file_path
    else:
        raise RuntimeError(f"Unknown load mode: {load_mode}")


def load_batch(file_paths: list[str], load_mode: str = None):
    return [load_single_image(f, load_mode) for f in file_paths]


def decode_single_image_test(tmp_path, input_img_file, input_format, backends):
    if backends:
        decoder = nvimgcodecs.Decoder(
            backends=backends, options=get_default_decoder_options())
    else:
        decoder = nvimgcodecs.Decoder(options=get_default_decoder_options())

    input_img_path = os.path.join(img_dir_path, input_img_file)

    decoder_input = load_single_image(input_img_path, input_format)

    if input_format == "path":
        test_img = decoder.read(decoder_input)
    else:
        test_img = decoder.decode(decoder_input)

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    compare_images([test_img], [ref_img])

@t.mark.parametrize("backends", [None,
                                 [nvimgcodecs.Backend(nvimgcodecs.GPU_ONLY, load_hint=0.5), nvimgcodecs.Backend(
                                     nvimgcodecs.HYBRID_CPU_GPU), nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)],
                                 [nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_img_file",
    ["bmp/cat-111793_640.bmp",

    "jpeg/padlock-406986_640_410.jpg",
    "jpeg/padlock-406986_640_411.jpg",
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_440.jpg",
    "jpeg/padlock-406986_640_444.jpg",
    "jpeg/padlock-406986_640_gray.jpg",
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
def test_decode_single_image_common(tmp_path, input_img_file, input_format, backends):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends)


@t.mark.parametrize("backends", [None,
                                 [nvimgcodecs.Backend(nvimgcodecs.GPU_ONLY, load_hint=0.5), nvimgcodecs.Backend(
                                     nvimgcodecs.HYBRID_CPU_GPU), nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)],
                                 [nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_img_file",
    ["jpeg/ycck_colorspace.jpg",
     "jpeg/cmyk.jpg",
    ]
)
@t.mark.skipif(nvimgcodecs.__cuda_version__ < 12010,  reason="requires CUDA >= 12.1")
def test_decode_single_image_cuda12_only(tmp_path, input_img_file, input_format, backends):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends)

@t.mark.parametrize("backends", [None,
                                 [nvimgcodecs.Backend(nvimgcodecs.GPU_ONLY, load_hint=0.5), nvimgcodecs.Backend(
                                     nvimgcodecs.HYBRID_CPU_GPU), nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)],
                                 [nvimgcodecs.Backend(nvimgcodecs.CPU_ONLY)]])
@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
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
def test_decode_batch(tmp_path, input_images_batch, input_format, backends, cuda_stream):
    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [cv2.imread(img, cv2.IMREAD_COLOR |
                             cv2.IMREAD_ANYDEPTH) for img in input_images]
    if backends:
        decoder = nvimgcodecs.Decoder(backends=backends, options=get_default_decoder_options())
    else:
        decoder = nvimgcodecs.Decoder(options=get_default_decoder_options())

    encoded_images = load_batch(input_images, input_format)

    if input_format == "path":
        test_images = decoder.read(encoded_images, cuda_stream=0 if cuda_stream is None else cuda_stream.ptr)
    else:
        test_images = decoder.decode(encoded_images, cuda_stream=0 if cuda_stream is None else cuda_stream.ptr)
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
    encode_params = nvimgcodecs.EncodeParams(jpeg2k_encode_params=nvimgcodecs.Jpeg2kEncodeParams(reversible=True))
    
    if encode_to_data:
        if cuda_stream:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params = encode_params, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params = encode_params)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + ".jp2")
        if cuda_stream:
            encoder.write(output_img_path, nv_ref_img,
                          params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_path, nv_ref_img,
                          params=encode_params)
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

    encode_params = nvimgcodecs.EncodeParams(jpeg2k_encode_params=nvimgcodecs.Jpeg2kEncodeParams(reversible=True))

    if encode_to_data:
        if cuda_stream:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=encode_params)
    else:
        output_img_paths = [os.path.join(tmp_path, os.path.splitext(
            os.path.basename(img))[0] + ".jp2") for img in input_images]
        if cuda_stream:
            encoder.write(output_img_paths, nv_ref_images,
                          params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_paths, nv_ref_images,
                          params=encode_params)
        test_encoded_images = []
        for out_img_path in output_img_paths:
            with open(out_img_path, 'rb') as in_file:
                test_encoded_img = in_file.read()
                test_encoded_images.append(test_encoded_img)

    test_decoded_images = [cv2.cvtColor(cv2.imdecode(
        np.asarray(bytearray(img)), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) for img in test_encoded_images]

    compare_cv_images(test_decoded_images, ref_images)
