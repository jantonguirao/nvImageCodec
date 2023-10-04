from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../resources"))

params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)

def test_decode_single_jpeg2k_16bit():
    input_img_path = os.path.join(img_dir_path, "jpeg2k/cat-1046544_640-16bit.jp2")

    decoder = nvimgcodec.Decoder()

    # First decode to the original bitdepth
    img_u16 = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(allow_any_depth=True))
    assert img_u16.shape == (475, 640, 3)
    assert img_u16.dtype == np.uint16
    data_u16 = np.array(img_u16.cpu())

    # Now decode without extra parameters, meaning we will decode to HWC RGB u8 always
    img_u8 = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(allow_any_depth=False))
    assert img_u8.shape == (475, 640, 3)
    assert img_u8.dtype == np.uint8
    data_u8 = np.array(img_u8.cpu())

    np.testing.assert_allclose(data_u8, (data_u16 * (65535 / 255)).astype(np.uint8))
