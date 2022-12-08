import os
import pytest as t
import hashlib
import subprocess

img_dir_path = os.path.abspath("../resources")

def file_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@t.mark.parametrize("exec_dir_path, transcode_exec",
    [(os.path.abspath("../build/bin/bin"), "nvimtrans"),
     (os.path.abspath("../build/bin/bin"), "nvimgcodecs_example_high_level_api")
    ]
)
@t.mark.parametrize(
    "input_img_file,codec,ouput_img_file,params,check_sum",
    [
    ("base/4k_lossless.bmp", "bmp", "4k_lossless-bmp.bmp", "", "22f22e349eb022199a66eaf4d09a2642"),
    ("base/4k_lossless.bmp", "jpeg",  "4k_lossless-bmp.jpg","", "546f0d10710f4357a9d1c6070487c28f"),
    ("base/4k_lossless.bmp", "jpeg2k", "4k_lossless-bmp.jp2","", "69e3a367c872603a8fbc784dcaf10931"),
    ("base/4k_lossless.jp2", "bmp", "4k_lossless-jp2.bmp","", "22f22e349eb022199a66eaf4d09a2642"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2.jpg","", "546f0d10710f4357a9d1c6070487c28f"),
    ("base/4k_lossless.jp2", "jpeg2k", "4k_lossless-jp2.jp2","", "69e3a367c872603a8fbc784dcaf10931"),
    ("base/4k_lossless_q95_444.jpg", "bmp", "4k_lossless-jpg.bmp","", "bc661957135a90af7a74702b06c4b633"),
    ("base/4k_lossless_q95_444.jpg", "jpeg", "4k_lossless-jpg.jpg","", "34e667ecd9df74f8be165bedbac72bb0"),
    ("base/4k_lossless_q95_444.jpg", "jpeg2k", "4k_lossless-jpg.jp2","", "f34e26bdcddd718428a5494b609a1c56"),

    ("base/4k_lossless.bmp", "jpeg",  "4k_lossless-bmp-420.jpg","-chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-420.jpg","-chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-422.jpg","-chroma_subsampling 422", "288ea71d08e74876d90c89c33284c56f"),
    ]
)

def test_imtrans(exec_dir_path, transcode_exec, tmp_path, input_img_file, codec, ouput_img_file, params, check_sum):
    os.chdir(exec_dir_path)
    input_img_path = os.path.join(img_dir_path, input_img_file)
    output_img_path = os.path.join(tmp_path, ouput_img_file)
    subprocess.run(".{}{} -i {} -c {} {} -o {}".format(os.sep, transcode_exec,
                   str(input_img_path), codec, params, str(output_img_path)), shell=True)
    assert check_sum == file_md5(output_img_path)
