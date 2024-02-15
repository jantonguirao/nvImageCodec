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
    if [[ -z "${NVIMGCODEC_PUBLIC_ROOT}" ]]; then
        echo "The env var NVIMGCODEC_PUBLIC_ROOT must be defined"
        exit 1
    fi
    if [[ "$#" != "2" ]]; then
        echo "usage: ./populate_public_repo.sh <private_branch_name> <public_branch_name>"
        exit 1
    fi

    # Clone a fresh NVIMGCODEC repo (from the main branch) to avoid mess
    NVIMGCODEC_TMP=$(mktemp -d)
    echo "Using temp directory: $NVIMGCODEC_TMP"

    NVIMGCODEC_PRIVATE_ROOT="$NVIMGCODEC_TMP/NVIMGCODEC"
    echo "Cloning nvimagecodec (private) repo into ${NVIMGCODEC_PRIVATE_ROOT} ..."
    git clone --branch $1 ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/nvimagecodec.git $NVIMGCODEC_PRIVATE_ROOT

    echo "Cloning nvimagecodec-public repo into ${NVIMGCODEC_PUBLIC_ROOT} ..."

    git clone ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/nvimagecodec-public.git $NVIMGCODEC_PUBLIC_ROOT
    
    echo "Creating $2 branch in nvimagecodec-public repo"
    pushd .
    cd $NVIMGCODEC_PUBLIC_ROOT
    git checkout -b $2
    popd

    # Get NVIMGCODEC current commit
    pushd .
    cd $NVIMGCODEC_PRIVATE_ROOT
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
    --exclude=docker/config-docker.sh \
    --exclude=tools/populate_public_repo.sh \
    --exclude=tools/arch_3rd_party_oss.sh \
    --exclude=tools/publish_on_github.sh \
    --exclude=tools/restore_submodules.sh \
    --exclude=docs \
    --exclude=LICENSE.txt \
    --exclude=LICENSE.txt_PUBLIC \
    --exclude=.nspect-vuln-allowlist.toml \
    $NVIMGCODEC_PRIVATE_ROOT/ $NVIMGCODEC_PUBLIC_ROOT/

    # Copy public license file with changing name
    cp -f $NVIMGCODEC_PRIVATE_ROOT/LICENSE.txt_PUBLIC $NVIMGCODEC_PUBLIC_ROOT/LICENSE.txt

    # Make additional scans for potential files or changes to exclude
    echo "Checking for potential keywords and files to exclude ..."
    grep -r --exclude-dir=.git gitlab $NVIMGCODEC_PUBLIC_ROOT || echo "Nothing found"  

    # Prompt for checking in
    echo -e "\nDone. Files are copied from NVIMGCODEC at commit ${nvimgcodec_commit}."
    echo "Once you examine the integraty of the files, please commit as follows:"
    echo ""
    echo "  cd ${NVIMGCODEC_PUBLIC_ROOT}"
    echo "  git add -A ."
    echo "  git commit -m 'sync with internal nvimagecodec repo  $1 branch (commit ${nvimgcodec_commit})'"
    echo "  git push --set-upstream origin $2"
    echo ""
    echo "Then, remember to create and publish a git tag for the new release."
    exit 0
}


copy_from_private_to_public $@
