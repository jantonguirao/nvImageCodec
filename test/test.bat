curl -SL --output Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
Miniconda3-latest-Windows-x86_64.exe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /InstallationType=JustMe /S /D=%cd%\miniconda3

set PATH=%cd%/miniconda3/;%cd%/miniconda3/condabin;%cd%/miniconda3/Scripts
call conda create -y -n test_py3.10 python=3.10

call conda activate test_py3.10 || echo Failed to activate conda env
pip install pytest nvidia-nvjpeg-cu%CUDA_VERSION_MAJOR% nvidia-nvjpeg2k-cu%CUDA_VERSION_MAJOR%
pip install nvidia-pyindex
pip install nvidia-cuda-runtime-cu%CUDA_VERSION_MAJOR%
pip install -r requirements_cu%CUDA_VERSION_MAJOR%.txt

REM Uncomment for local testing
REM pip install -I ..\..\python\nvidia_nvimgcodec_cu%CUDA_VERSION_MAJOR%-0.2.0.0-py3-none-win_amd64.whl

set ORIG_PATH=%PATH%
set PATH=%PATH%;%cd%/miniconda3/envs/test_py3.10/Lib/site-packages/nvidia/cuda_runtime/bin
set PATH=%PATH%;%cd%/miniconda3/envs/test_py3.10/Lib/site-packages/nvidia/nvjpeg/bin
set PATH=%PATH%;%cd%/miniconda3/envs/test_py3.10/Lib/site-packages/nvidia/nvjpeg2k/bin
set NVIMGCODEC_EXTENSIONS_PATH=%cd%/../extensions


nvimgcodec_tests.exe --resources_dir ..\resources
pytest -v test_transcode.py
pytest -v .\python

set PATH=%ORIG_PATH%
call conda deactivate
