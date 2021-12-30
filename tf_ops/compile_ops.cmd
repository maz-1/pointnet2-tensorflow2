@echo off

cd /D "%~dp0"

SET VSIDE=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
WHERE cl
IF %ERRORLEVEL% NEQ 0 @call "%VSIDE%\VC\Auxiliary\Build\vcvarsall.bat" x64

SET CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1

@FOR /f "delims=" %%i in ('python get_tf_flags.py cflags') DO set TF_CFLAGS=%%i
@FOR /f "delims=" %%i in ('python get_tf_flags.py lflags') DO set TF_LFLAGS=%%i

rem echo %TF_CFLAGS%
rem echo %TF_LFLAGS%

cl.exe /nologo /std:c++14 /Zc:__cplusplus /Zc:preprocessor /LD .\3d_interpolation\tf_interpolate.cpp /I"%CUDA_ROOT%\include" "%CUDA_ROOT%\lib\x64\cudart.lib" %TF_CFLAGS% %TF_LFLAGS% /O2 /link /out:.\3d_interpolation\tf_interpolate_so.dll
