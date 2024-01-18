@ECHO OFF
SETLOCAL

REM Usage: build_helper.bat <cmake-build-dir> <cuda-version> [cmake-flags [...]]
REM Example: build_helper.bat ..\build 10.1 -DBUILD_EXAMPLES=FALSE -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release

IF [%1]==[] GOTO :error_build_dir
IF [%2]==[] GOTO :error_cuda_version

SET GENERATOR="Visual Studio 16 2019"

SET "SCRIPT_DIR=%~dp0"
SET "SOURCE_DIR=%SCRIPT_DIR%\.."
SET "BUILD_DIR=%1"
SET "CUDA_VERSION=%2"
SET CMAKE_ARGS=

for /F "tokens=1,2*" %%a in ("%*") do (
  set BUILD_DIR=%%a
  set CUDA_VERSION=%%b
  set CMAKE_ARGS=%%c
)

echo "%SOURCE_DIR%"

set PATH=%PATH%;%SOURCE_DIR%\install\include;%SOURCE_DIR%\install\lib;%SOURCE_DIR%\install\x64\vc16\staticlib

cmake -DNVJPEG2K_LIBRARY=C:\NVIDIA_nvJPEG2K\v0.7\lib\11\nvjpeg2k.lib ^
 -DNVJPEG2K_INCLUDE=C:\NVIDIA_nvJPEG2K\v0.7\include ^
 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
 -S %SOURCE_DIR% ^
 -B %BUILD_DIR% ^
 -G %GENERATOR% ^
 -T host=x64 ^
 -A x64

if %errorlevel% neq 0 exit /b %errorlevel%

pushd %SOURCE_DIR%\build

cmake --build . --config Release

popd

GOTO :eof

:error_build_dir
echo Build directory not specified
GOTO :eof

:error_cuda_version
echo CUDA toolkit version not specified

:eof
