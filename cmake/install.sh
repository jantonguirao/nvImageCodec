#!/bin/bash -e
cp ./lib64/libnvimgcodecs.so /usr/lib/x86_64-linux-gnu
mkdir -p /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvjpeg_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvjpeg2k_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvbmp_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/liblibjpeg_turbo_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/liblibtiff_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/libopencv_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
