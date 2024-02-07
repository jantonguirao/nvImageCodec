#!/bin/bash

set -e
NVIMGCODEC_TMP=


clean_up() {
    status=$?
    if [[ ! -z "${NVIMGCODEC_TMP}" ]]; then
        echo "cleaning up $NVIMGCODEC_TMP ..."
        rm -rf $NVIMGCODEC_TMP
    fi
    exit $status
}
trap clean_up EXIT


copy_from_private_to_public() {
    if [[ -z "${NVIMGCODEC_GITHUB_ROOT}" ]]; then
        echo "The env var NVIMGCODEC_GITHUB_ROOT must be defined"
        exit 1
    fi
    if [[ "$#" != "2" ]]; then
        echo "usage: ./publish_public_repo.sh <private_branch_name> <github_branch_name>"
        exit 1
    fi

    # Clone a fresh NVIMGCODEC public (mirror) repo (from the main branch) to avoid mess
    NVIMGCODEC_TMP=$(mktemp -d)
    echo "Using temp directory: $NVIMGCODEC_TMP"

    NVIMGCODEC_PUBLIC_TMP="$NVIMGCODEC_TMP/NVIMGCODEC"
    echo "Cloning nvimagecodec (public) repo into ${NVIMGCODEC_PUBLIC_TMP} ..."
    git clone --branch $1 ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/nvimagecodec-public.git $NVIMGCODEC_PUBLIC_TMP

    echo "Cloning nvimagecodec-github repo into ${NVIMGCODEC_GITHUB_ROOT} ..."

    git clone https://github.com/NVIDIA/nvImageCodec.git $NVIMGCODEC_GITHUB_ROOT
    
    # Get NVIMGCODEC current commit
    pushd .
    cd $NVIMGCODEC_PUBLIC_TMP
    nvimgcodec_commit=$(git rev-parse --short HEAD)
    popd

    # Sync public repo with private excluding private files and directories 
    rsync -arv \
    --delete-after \
    --exclude=.git \
    --exclude=ca-chain.crt \
    --exclude=coverity.sh \
    --exclude=.gitlab-ci.yml \
    --exclude=.githooks \
    --exclude=.gitlab \
    $NVIMGCODEC_PUBLIC_TMP/ $NVIMGCODEC_GITHUB_ROOT/

    # Prompt for checking in
    echo -e "\nDone. Files are copied from NVIMGCODEC public repo at commit ${nvimgcodec_commit}."
    echo "Once you examine the integraty of the files, please commit as follows:"
    echo ""
    echo "  cd ${NVIMGCODEC_GITHUB_ROOT}"
    echo "  git add -A ."
    echo ""
    echo "Optionally you would need to restore submodules"
    echo "  ./retstore_submodules.sh"
    echo ""
    echo "Modify message as needed and commit"
    echo "  git commit -m 'Adding code for release v0.2.0 (commit ${nvimgcodec_commit})'"
    echo ""    
    echo "Check lfs files"
    echo "  git lfs ls-files"
    echo ""
    echo "Push to github" 
    echo "  git push --set-upstream origin $2"
    echo ""
    echo "Then, remember to create git tag for the new release."
    exit 0
}


copy_from_private_to_public $@
