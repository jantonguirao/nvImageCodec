#!/usr/bin/env bash

# By default build debug version
build_type=Debug

# Use Release build type if:
# * has 'release' in branch name,
# * is master or develop branch,
# * has a tag.
if [[ $CI_COMMIT_REF_NAME == *"release"* ]]; then
    build_type=Release
elif [[ $CI_COMMIT_REF_NAME == "master" ]]; then
    build_type=Release
elif [[ $CI_COMMIT_REF_NAME == "develop" ]]; then
    build_type=Release
elif [[ $CI_COMMIT_TAG != "" ]]; then
    build_type=Release
fi

export NVIMAGECODEC_BUILD_TYPE=$build_type
echo $NVIMAGECODEC_BUILD_TYPE
