@echo off
rem Build 64-bit ntdata.dll for TDX (needs Visual Studio 2022 C++ tools)
call "D:\Visual Studio\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d "%~dp0"
cl /nologo /utf-8 /O2 /EHsc /std:c++17 /LD /MT ntdata.cpp /Fe:ntdata.dll /link /NOLOGO
del /q ntdata.obj ntdata.exp ntdata.lib 2>nul
