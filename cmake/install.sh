#!/bin/bash -e
mkdir -p /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvjpeg.so.22.11.0.0 /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvjpeg2k.so.0.6.0.0 /usr/lib/nvimgcodecs/plugins
cp ./plugins/libnvbmp.so.0.1.0.0 /usr/lib/nvimgcodecs/plugins
cp ./lib64/libnvimgcodecs.so /usr/lib/x86_64-linux-gnu
#cp ./lib64/nvimgcodecs.cpython* ./bin