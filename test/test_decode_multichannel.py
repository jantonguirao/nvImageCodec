from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../resources"))

params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)

def test_decode_single_tiff_multichannel():
    input_img_path = os.path.join(img_dir_path, "tiff/multichannel/cat-1245673_640_multichannel.tif")

    decoder = nvimgcodec.Decoder()

    # By default the decoder works on RGB, ignoring extra channels
    img_RGB = decoder.read(input_img_path)
    assert img_RGB.shape == (423, 640, 3)

    # Using UNCHANGED, we get all 6 channels: R, G, B, Y, Cb, Cr in this image
    params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    img_unchanged = decoder.read(input_img_path, params=params_unchanged)
    assert img_unchanged.shape == (423, 640, 6)

def test_decode_single_tiff_with_alpha_16bit():
    input_img_path = os.path.join(img_dir_path, "tiff/with_alpha_16bit/4ch16bpp.tiff")

    decoder = nvimgcodec.Decoder()

    # By default the decoder works on RGB uint8, ignoring extra channels and bit depth
    img_RGB_u8 = decoder.read(input_img_path)
    assert img_RGB_u8.shape == (497, 497, 3)
    assert img_RGB_u8.dtype == np.uint8

    # Using UNCHANGED, we get all 4 channels, and using allow_any_depth=True we get the original uint16 representation
    params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    img_unchanged = decoder.read(input_img_path, params=params_unchanged)
    assert img_unchanged.shape == (497, 497, 4)
    assert img_unchanged.dtype == np.uint16
