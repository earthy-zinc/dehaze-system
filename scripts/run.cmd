@echo off
REM Windows cmd / PowerShell 薄壳，透传所有参数到 run.py
python "%~dp0run.py" %*
