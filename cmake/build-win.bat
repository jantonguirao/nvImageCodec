@ECHO OFF
SETLOCAL

REM Usage: build-win.bat <cmake-build-dir> <cuda-version> [cmake-flags [...]]
REM Example: build-win.bat ..\build 10.1 -DBUILD_EXAMPLES=FALSE -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release

IF [%1]==[] GOTO :error_build_dir
IF [%2]==[] GOTO :error_cuda_version

SET CMAKE=cmake
SET GENERATOR=Ninja

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

%CMAKE% -G %GENERATOR% -S "%SOURCE_DIR%" -B %BUILD_DIR% %CMAKE_ARGS%
if %errorlevel% neq 0 exit /b %errorlevel%


GOTO :eof

:error_build_dir
echo Build directory not specified
GOTO :eof

:error_cuda_version
echo CUDA toolkit version not specified

:eof
