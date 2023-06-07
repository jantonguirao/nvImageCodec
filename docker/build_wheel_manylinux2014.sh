#!/bin/bash

img_name="nvimgcodecs_wheel_builder"
deps_img="urm.nvidia.com/sw-dl-dali-docker-local/dali-ci/main/cu121/x86_64/deps_prebuilt:latest"
docker build --pull -t ${img_name} --build-arg "DEPS_IMAGE_NAME=${deps_img}" -f docker/Dockerfile_builder .

container=$(docker create "${img_name}")
rm -rf wheelhouse && mkdir -p wheelhouse
docker cp "${container}:/wheelhouse/." "wheelhouse"
docker rm -f "${container}"