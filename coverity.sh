#!/bin/bash

URL="https://ipp-coverity-10.nvidia.com:8443"
STREAM="nvImageCodecs"

if ! type cov-configure >/dev/null 2>&1; then
    echo "[WARNING] Coverity is not installed!!!"
    echo "Please use below command to install:"
    echo "    p4 sync //sw/tools/Coverity/2022.6.0/Linux64/..."
    echo "After installation, please add install_dir/bin to PATH environment"
    echo "and configure compilers using following commands:"
    echo "    cov-configure --gcc"
    echo "    cov-configure --cuda"
    echo "    cov-configure --template --compiler c++ --comptype gcc"
    exit 1
fi

# remove previous analysis dir
if [ -d analysis_dir ]; then
    rm -fr analysis_dir
fi

# build
cov-build --dir analysis_dir make -j15 clean all
if [ $? -ne 0 ]; then
    echo "Failed to run cov-build!"
    exit 1
fi

# analyze
cov-analyze --dir analysis_dir
if [ $? -ne 0 ]; then
    echo "Failed to run cov-analyze!"
    exit 1
fi

# cov-format-errors --dir analysis_dir --html-output output

# commit
# cov-commit-defects --dir analysis_dir --url $URL --stream $STREAM --certs ../ca-chain.crt --on-new-cert trust --user reporter --password coverity # Using local certificate
cov-commit-defects --dir analysis_dir --url $URL --stream $STREAM --on-new-cert trust --user reporter --password coverity
if [ $? -ne 0 ]; then
    echo "Failed to run cov-commit-defects!"
    exit 1
fi
