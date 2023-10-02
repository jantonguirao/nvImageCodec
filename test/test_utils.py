from __future__ import annotations
import numpy as np
import cv2
import cupy as cp
from nvidia import nvimgcodecs

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

def compare_device_with_host_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        cp_test_img = cp.asarray(test_images[i])
        np_test_img = np.asarray(cp.asnumpy(cp_test_img))
        ref_img = cv2.cvtColor(ref_images[i], cv2.COLOR_BGR2RGB)
        ref_img = np.asarray(ref_img)
        compare_image(np_test_img, ref_img)

def compare_host_images(test_images, ref_images):
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
