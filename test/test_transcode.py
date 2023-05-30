import os
import pytest as t
import hashlib
import subprocess


img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))

def file_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def imtrans(exec_dir_path, transcode_exec, tmp_path, input_img_file, codec, ouput_img_file, params, check_sum):
    os.chdir(exec_dir_path)
    input_img_path = os.path.join(img_dir_path, input_img_file)
    output_img_path = os.path.join(tmp_path, ouput_img_file)
    cmd = ".{}{} -i {} -c {} {} -o {}".format(os.sep, transcode_exec,
                                              str(input_img_path), codec, params, str(output_img_path))
    subprocess.run(cmd, shell=True)
    assert check_sum == file_md5(output_img_path)
    
@t.mark.parametrize("exec_dir_path, transcode_exec",
                    [(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/bin/bin")), "nvimtrans"),
                     (os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/bin/bin")), "nvimgcodecs_example_high_level_api")
    ]
)
@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
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

    ("base/4k_lossless.bmp", "jpeg",  "4k_lossless-bmp-420.jpg","--chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-420.jpg","--chroma_subsampling 420", "cba05a2ab83cf6a702f2d273dfbbae50"),
    ("base/4k_lossless.jp2", "jpeg", "4k_lossless-jp2-422.jpg","--chroma_subsampling 422", "288ea71d08e74876d90c89c33284c56f"),
    
    #jpeg various input chroma subsampling
    ("base/rgb_25_410.jpg", "bmp",    "rgb_25_410-jpg.bmp", "", "09a9dbb7b75278197c58e7ddcb93c4cb"),
    ("base/rgb_25_410.jpg", "jpeg",   "rgb_25_410-jpg.jpg", "", "d2b17165ebdc89d84d8db39d71524bf5"),
    ("base/rgb_25_410.jpg", "jpeg2k", "rgb_25_410-jpg.jp2", "", "6562455982c8d6a1ee6484cb8c635cb0"),
    
    ("base/rgb_25_411.jpg", "bmp",    "rgb_25_411-jpg.bmp", "", "e9e7261b325d3f2895f0281147f61e0b"),
    ("base/rgb_25_411.jpg", "jpeg",   "rgb_25_411-jpg.jpg", "", "46cc411a9770d1e19e3015cf97c97db5"),
    ("base/rgb_25_411.jpg", "jpeg2k", "rgb_25_411-jpg.jp2", "", "f16e98162f7cd444abae48452cfd8f87"),
    
    ("base/rgb_25_420.jpg", "bmp",    "rgb_25_420-jpg.bmp", "", "4be980aed1726182b873a55611574303"),
    ("base/rgb_25_420.jpg", "jpeg",   "rgb_25_420-jpg.jpg", "", "35893dc324cd29042c13aa8d9ac3374d"),
    ("base/rgb_25_420.jpg", "jpeg2k", "rgb_25_420-jpg.jp2", "", "4e7a8881fe9c4ebaddf8855a43794080"),

    ("base/rgb_25_422.jpg", "bmp",    "rgb_25_422-jpg.bmp", "", "9ab1754b7c192e03090863285dba68ed"),
    ("base/rgb_25_422.jpg", "jpeg",   "rgb_25_422-jpg.jpg", "", "7b71fa9f5359840e004ebee7921fe554"),
    ("base/rgb_25_422.jpg", "jpeg2k", "rgb_25_422-jpg.jp2", "", "80aad59a2e0e1025afeea235daa34559"),

    ("base/rgb_25_440.jpg", "bmp",    "rgb_25_440-jpg.bmp", "", "ba863e3f78be2238300cf2ef5c2d60e9"),
    ("base/rgb_25_440.jpg", "jpeg",   "rgb_25_440-jpg.jpg", "", "57249544fdace822002cbbfaa5988345"),
    ("base/rgb_25_440.jpg", "jpeg2k", "rgb_25_440-jpg.jp2", "", "6c7b06b6bcc03360e05efe2ad8995d66"),

    ("base/rgb_25_444.jpg", "bmp",    "rgb_25_444-jpg.bmp", "", "49f3b3f8ef8c792520bd43436ff76153"),
    ("base/rgb_25_444.jpg", "jpeg",   "rgb_25_444-jpg.jpg", "", "9f9a2c181e981be66d7f0f2a0139935e"),
    ("base/rgb_25_444.jpg", "jpeg2k", "rgb_25_444-jpg.jp2", "", "dda34d8944767382cc532cf334b9abbb"),

    #test pnm
    ("base/4k_lossless.bmp", "pnm", "4k_lossless-bmp.ppm","", "b452d532ba80d9e560baabe8208a04f5"),
    ("base/4k_lossless_q95_444.jpg", "pnm", "4k_lossless-jpg.ppm","", "083f9062bd227cba62471f5ae3b85df1"),
    ("base/4k_lossless.jp2", "pnm", "4k_lossless-jp2.ppm", "", "b452d532ba80d9e560baabe8208a04f5"),
    
    #test orientation
    ("base/f8-exif.jpg", "bmp", "f8-exif_orientation_enabled.bmp", "", "02fa6fe20734b0463b6bed8623d405c4"),
    ("base/f8-exif.jpg", "bmp", "f8-exif_orientation_disabled.bmp", "--ignore_orientation true", "e44b36cc1056cefb8df7473d96dd65e7"),
    ]
)
def test_imtrans(exec_dir_path, transcode_exec, tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    imtrans(exec_dir_path, transcode_exec, tmp_path, input_img_file, codec, output_img_file, params, check_sum)

