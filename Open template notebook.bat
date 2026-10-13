@echo off
rem Opens spherical_indexing_template.ipynb in JupyterLab. Close this window to stop Jupyter.
rem PYTHON = the python.exe of the environment where kikuchipy and jupyterlab are installed.
rem "python" uses the one on PATH; otherwise put the full path, e.g. C:\path\to\venv\Scripts\python.exe
set PYTHON=python
cd /d "%~dp0"
"%PYTHON%" -m jupyter lab "spherical_indexing_template.ipynb"
if errorlevel 1 pause
