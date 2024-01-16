# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
import os
import numpy as np
from nvidia import nvimgcodec
import pytest as t

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))

filenames = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_440.jpg",
]

def test_decode_single_file():
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, filenames[0])
    code_stream0 = nvimgcodec.CodeStream.CreateFromFile(fpath)
    img0 = decoder.decode(code_stream0).cpu()

    code_stream1 = nvimgcodec.CodeStream.CreateFromHostMem(open(fpath, 'rb').read())
    img1 = decoder.decode(code_stream1).cpu()

    code_stream2 = nvimgcodec.CodeStream.CreateFromHostMem(np.fromfile(fpath, dtype=np.uint8))
    img2 = decoder.decode(code_stream2).cpu()

    np.testing.assert_allclose(img0, img1)
    np.testing.assert_allclose(img0, img2)
    np.testing.assert_allclose(img1, img2)

    raw_bytes = open(fpath, 'rb').read()
    img3 = decoder.decode(raw_bytes).cpu()

    np.testing.assert_allclose(img0, img3)
    np.testing.assert_allclose(img1, img3)
    np.testing.assert_allclose(img2, img3)

def test_decode_batch():
    decoder = nvimgcodec.Decoder()
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]

    code_streams0 = [nvimgcodec.CodeStream.CreateFromFile(fpath) for fpath in fpaths]
    imgs0 =[img.cpu() for img in decoder.decode(code_streams0)]
    
    code_streams1 = [nvimgcodec.CodeStream.CreateFromHostMem(open(fpath, 'rb').read()) for fpath in fpaths]
    imgs1 =[img.cpu() for img in decoder.decode(code_streams1)]
    
    code_streams2 = [nvimgcodec.CodeStream.CreateFromHostMem(np.fromfile(fpath, dtype=np.uint8)) for fpath in fpaths]
    imgs2 =[img.cpu() for img in decoder.decode(code_streams2)]

    for img0, img1 in zip(imgs0, imgs1):
        np.testing.assert_allclose(img0, img1)
    for img0, img2 in zip(imgs0, imgs2):
        np.testing.assert_allclose(img0, img2)
    for img1, img2 in zip(imgs1, imgs2):
        np.testing.assert_allclose(img1, img2)

    raw_bytes = [open(fpath, 'rb').read() for fpath in fpaths]
    imgs3 = [img.cpu() for img in decoder.decode(raw_bytes)]

    for img0, img3 in zip(imgs0, imgs3):
        np.testing.assert_allclose(img0, img3)
    for img1, img3 in zip(imgs1, imgs3):
        np.testing.assert_allclose(img1, img3)
    for img2, img3 in zip(imgs2, imgs3):
        np.testing.assert_allclose(img2, img3)
