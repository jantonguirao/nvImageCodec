import os
import numpy as np
import cv2
import cupy as cp
import pytest as t
from nvidia import nvimgcodecs

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))

MAX_DIFF_THRESHOLD = 4


def compare_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        cp_test_img = cp.asarray(test_images[i])
        np_test_img = np.asarray(cp.asnumpy(cp_test_img))
        ref_img = cv2.cvtColor(ref_images[i], cv2.COLOR_BGR2RGB)
        ref_img = np.asarray(ref_img)
        diff = ref_img.astype(np.int32) - np_test_img.astype(np.int32)
        diff = np.absolute(diff)

        assert np_test_img.shape == ref_img.shape
        assert diff.max() <= MAX_DIFF_THRESHOLD

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
def test_decode_single_image(tmp_path, input_img_file, decode_data):
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



@t.mark.parametrize("decode_data", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [   ("bmp/cat-111793_640.bmp",

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
        "jpeg2k/cat-1245673_640-12bit.jp2")
     ]
)
def test_decode_batch(tmp_path, input_images_batch, decode_data):
    decoder = nvimgcodecs.Decoder(options=":fancy_upsampling=1")
    input_images = [os.path.join(img_dir_path, img) for img in input_images_batch]
    if decode_data:
        data_list = []
        for img in input_images:
            with open(img, 'rb') as in_file:
                data = in_file.read()
                data_list.append(data)
        test_images = decoder.decode(data_list)
    else:
        test_images = decoder.decode(input_images)
    
    ref_images = [cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH) for img in input_images]        
    compare_images(test_images, ref_images)
