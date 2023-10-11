import os
import pytest as t
import hashlib
import subprocess


img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
exec_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin"))
transcode_exec="nvimtrans"


def file_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
    
@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
    [
    ("base/4k_lossless.bmp", "bmp", "4k_lossless-bmp.bmp", "", "22f22e349eb022199a66eaf4d09a2642"),
    ("base/4k_lossless.bmp", "jpeg",  "4k_lossless-bmp.jpg","", "546f0d10710f4357a9d1c6070487c28f"),
    ("base/4k_lossless.bmp", "jpeg2k", "4k_lossless-bmp.jp2", "--enc_color_trans true", "69e3a367c872603a8fbc784dcaf10931"),
    ("base/4k_lossless.jp2", "bmp", "4k_lossless-jp2.bmp","", "22f22e349eb022199a66eaf4d09a2642"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2.jpg","", "546f0d10710f4357a9d1c6070487c28f"),
    ("base/4k_lossless.jp2", "jpeg2k", "4k_lossless-jp2.jp2", "--enc_color_trans true", "69e3a367c872603a8fbc784dcaf10931"),
    ("base/4k_lossless_q95_444.jpg", "bmp", "4k_lossless-jpg.bmp","", "bc661957135a90af7a74702b06c4b633"),
    ("base/4k_lossless_q95_444.jpg", "jpeg", "4k_lossless-jpg.jpg","", "34e667ecd9df74f8be165bedbac72bb0"),
    ("base/4k_lossless_q95_444.jpg", "jpeg2k", "4k_lossless-jpg.jp2", "--enc_color_trans true", "f34e26bdcddd718428a5494b609a1c56"),

    ("base/4k_lossless.bmp", "jpeg",  "4k_lossless-bmp-420.jpg","--chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-420.jpg","--chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-422.jpg","--chroma_subsampling 422", "288ea71d08e74876d90c89c33284c56f"),
    
    #jpeg various input chroma subsampling
    ("jpeg/padlock-406986_640_410.jpg", "bmp",    "padlock-406986_640_410-jpg.bmp", "", "01d3488f82c6422a76db5ab0e89e8a86"),
    ("jpeg/padlock-406986_640_410.jpg", "jpeg",   "padlock-406986_640_410-jpg.jpg", "", "31f72adf6f8553118927df00a1a8adc3"),
    ("jpeg/padlock-406986_640_410.jpg", "jpeg2k", "padlock-406986_640_410-jpg.jp2", "--enc_color_trans true", "06d36693b45d45214ea672a0461b9540"),
    
    ("jpeg/padlock-406986_640_411.jpg", "bmp",    "padlock-406986_640_411-jpg.bmp", "", "925c8465bbd6059570350612abc47ecc"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg",   "padlock-406986_640_411-jpg.jpg", "", "31fcb68b4641c6db3e0464d06dffbbda"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg2k", "padlock-406986_640_411-jpg.jp2", "--enc_color_trans true", "08c7921bf34adb4ea180fa1359fe4cca"),
    
    ("jpeg/padlock-406986_640_420.jpg", "bmp",    "padlock-406986_640_420-jpg.bmp", "", "a04595f50cfbe8d9d5726a8eb3dc70f1"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg",   "padlock-406986_640_420-jpg.jpg", "", "740d7a4ea7237793033a1de88f77e610"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg2k", "padlock-406986_640_420-jpg.jp2", "--enc_color_trans true", "10d7bf042a3fc42c90c762eebce16f09"),

    ("jpeg/padlock-406986_640_422.jpg", "bmp",    "padlock-406986_640_422-jpg.bmp", "", "fca403b759d1ef4173e35c134258719a"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg",   "padlock-406986_640_422-jpg.jpg", "", "b225ca4dcc19412d9c18caf524f9080f"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg2k", "padlock-406986_640_422-jpg.jp2", "--enc_color_trans true", "a9e445c5df6ef56f1f7f695bae42b807"),

    ("jpeg/padlock-406986_640_440.jpg", "bmp",    "padlock-406986_640_440-jpg.bmp", "", "aa26bef4b98a8c56f837426bd33f2eaf"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg",   "padlock-406986_640_440-jpg.jpg", "", "bd29ce0b525a33e8919362c96aee7aa6"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg2k", "padlock-406986_640_440-jpg.jp2", "--enc_color_trans true", "a90b221752545b5f4640494e9d67ff6c"),

    ("jpeg/padlock-406986_640_444.jpg", "bmp",    "padlock-406986_640_444-jpg.bmp", "", "0b5846e30997034867d08843e527b951"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg",   "padlock-406986_640_444-jpg.jpg", "", "8d7e0c2b8a458f6ec4b6af03cc1c9f16"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg2k", "padlock-406986_640_444-jpg.jp2", "--enc_color_trans true", "7c6a2375a8b1387036bd7a02c19e8f8e"),

    #test pnm
    ("base/4k_lossless.bmp", "pnm", "4k_lossless-bmp.ppm","", "0a26f98671a4eba8e2092d2777fc701d"),
    ("base/4k_lossless_q95_444.jpg", "pnm", "4k_lossless-jpg.ppm","", "78521addd2e1d4838faf525ce8704ad2"),
    ("base/4k_lossless.jp2", "pnm", "4k_lossless-jp2.ppm", "", "0a26f98671a4eba8e2092d2777fc701d"),
    
    #test orientation
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_enabled.bmp", "", "70c64d06465f26100d7bcbb6a193404a"),
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_disabled.bmp", "--ignore_orientation true", "a04595f50cfbe8d9d5726a8eb3dc70f1"),
    ]
)
def test_imtrans(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    os.chdir(exec_dir_path)
    input_img_path = os.path.join(img_dir_path, input_img_file)
    output_img_path = os.path.join(tmp_path, output_img_file)
    cmd = ".{}{} -i {} -c {} {} -o {}".format(os.sep, transcode_exec,
                                              str(input_img_path), codec, params, str(output_img_path))
    subprocess.run(cmd, shell=True)
    assert check_sum == file_md5(output_img_path)

