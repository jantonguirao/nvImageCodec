from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec
import pytest as t

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))

params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)
backends_gpu_only=[nvimgcodec.Backend(nvimgcodec.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)]
backends_cpu_only=[nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]

@t.mark.parametrize("path,shape,dtype,backends",
                    [
                        ("tiff/multichannel/cat-1245673_640_multichannel.tif", (423, 640, 6), np.uint8, None),
                        ("tiff/with_alpha_16bit/4ch16bpp.tiff", (497, 497, 4), np.uint16, None),
                        ("png/with_alpha_16bit/4ch16bpp.png", (497, 497, 4), np.uint16, None),
                        ("png/with_alpha/cat-111793_640-alpha.png", (426, 640, 4), np.uint8, None),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", (426, 640, 4), np.uint8, backends_cpu_only),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", (426, 640, 4), np.uint8, backends_gpu_only),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", (497, 497, 4), np.uint16, backends_cpu_only),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", (497, 497, 4), np.uint16, backends_gpu_only),
                    ],
                    )
def test_decode_single_multichannel(path, shape, dtype, backends):
    input_img_path = os.path.join(img_dir_path, path)

    decoder = nvimgcodec.Decoder(backends=backends)

    # By default the decoder works on RGB and uint8, ignoring extra channels
    img_RGB = decoder.read(input_img_path)
    expected_shape_rgb = shape[:2] + (3,)
    assert img_RGB.shape == expected_shape_rgb
    assert img_RGB.dtype == np.uint8

    # Using UNCHANGED
    params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    img_unchanged = decoder.read(input_img_path, params=params_unchanged)
    assert img_unchanged.shape == shape
    assert img_unchanged.dtype == dtype

    # Gray
    params_gray=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY)
    img_gray = decoder.read(input_img_path, params=params_gray)
    expected_shape_gray = shape[:2] + (1,)
    assert img_gray.shape == expected_shape_gray
    assert img_gray.dtype == np.uint8
