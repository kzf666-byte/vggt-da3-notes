@echo off
REM da3 CLI wrapper: injects --model-dir pointing at the local weights.
REM
REM Why: the CLI default is the HF repo id "depth-anything/DA3NESTED-GIANT-LARGE-1.1".
REM huggingface.co is unreachable here and the HF cache holds only an empty stub,
REM so the default always fails. Pass a local path instead.
REM
REM Usage:
REM   da3.bat image  D:\VGGT\xxx.jpg --export-dir D:\DA3\runs\test
REM   da3.bat images D:\VGGT          --export-dir D:\DA3\runs\test
REM   da3.bat auto   D:\VGGT\xxx.mp4  --export-dir D:\DA3\runs\test
REM
REM Set DA3_MODEL_DIR to use a different checkpoint.
REM NOTE: keep this file ASCII-only -- cmd.exe decodes .bat as GBK on zh-CN
REM Windows and non-ASCII bytes shred the comments into fake commands.

setlocal
if "%DA3_MODEL_DIR%"=="" set "DA3_MODEL_DIR=D:\DA3\models\DA3NESTED-GIANT-LARGE-1.1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "DA3_EXE=D:\pytorch\envs\da3\Scripts\da3.exe"

REM --help / --version / no args are top-level: pass straight through,
REM --model-dir only exists on the subcommands.
if "%~1"==""          "%DA3_EXE%" %* & exit /b %errorlevel%
if "%~1"=="--help"    "%DA3_EXE%" %* & exit /b %errorlevel%
if "%~1"=="-h"        "%DA3_EXE%" %* & exit /b %errorlevel%
if "%~1"=="--version" "%DA3_EXE%" %* & exit /b %errorlevel%

"%DA3_EXE%" %* --model-dir "%DA3_MODEL_DIR%"
