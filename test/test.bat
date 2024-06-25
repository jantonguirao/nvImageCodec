T:\python\Python3.9\python -m venv .venv
call .venv\Scripts\activate.bat

set CUDA_PATH=%cd%\.venv\Lib\site-packages\nvidia\cuda_runtime
set PATH=%PATH%;%cd%\.venv\Lib\site-packages\nvidia\cuda_runtime\bin

python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements_cu%CUDA_VERSION_MAJOR%.txt

set PATH=%PATH%;%cd%\.venv\Lib\site-packages\nvidia\nvjpeg\bin
set PATH=%PATH%;%cd%\.venv\Lib\site-packages\nvidia\nvjpeg2k\bin

set NVIMGCODEC_EXTENSIONS_PATH=%cd%\..\extensions

for %%G in (".\*.whl") do (
    echo Found and installing: %%G
    pip install -I %%G
)

pytest -v .\python
if %errorlevel% neq 0 exit /b %errorlevel%

REM pytest -v test_transcode.py
REM if %errorlevel% neq 0 exit /b %errorlevel%

nvimgcodec_tests.exe --resources_dir ..\resources
if %errorlevel% neq 0 exit /b %errorlevel%

call deactivate   