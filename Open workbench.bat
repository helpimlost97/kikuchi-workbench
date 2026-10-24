@echo off
rem Opens the Kikuchi workbench (pattern processor + spherical indexing + montage batch). Close this window to stop it.
rem PYTHON = the python.exe of the environment where the kikuchipy fork, CuPy, PyQt5 and pyebsdindex are installed.
set PYTHON=python
cd /d "%~dp0"
"%PYTHON%" kikuchi_workbench.py %*
if errorlevel 1 pause
