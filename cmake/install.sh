#!/bin/bash -e
mkdir -p /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvjpeg_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvjpeg2k_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvbmp_ext.so.0.1.0.0 /usr/lib/nvimgcodecs/plugins
cp ./lib64/libnvimgcodecs.so /usr/lib/x86_64-linux-gnu
#cp ./lib64/nvimgcodecs.cpython* ./bin
