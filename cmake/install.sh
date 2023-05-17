#!/bin/bash -e
mkdir -p /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvjpeg_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvjpeg2k_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./extensions/libnvbmp_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/extensions
cp ./lib64/libnvimgcodecs.so /usr/lib/x86_64-linux-gnu
#cp ./lib64/nvimgcodecs.cpython* ./bin
