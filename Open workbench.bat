@echo off
rem Opens the Kikuchi workbench (pattern processor + spherical indexing + montage batch). Close this window to stop it.
rem PYTHON = the python.exe of the environment where the kikuchipy fork, CuPy, PyQt5 and pyebsdindex are installed.
rem KPP_DIR = the folder of the kikuchi-pattern-processor clone; leave empty when it sits next to this folder.
set PYTHON=python
set KPP_DIR=
cd /d "%~dp0"
"%PYTHON%" kikuchi_workbench.py %*
if errorlevel 1 pause
