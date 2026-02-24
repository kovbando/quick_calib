@echo off
setlocal

call .\venv\Scripts\activate.bat

python quick_calib.py "G:\20260224\flircalib_1\image_0_c"
if errorlevel 1 exit /b 1

python quick_calib.py "G:\20260224\flircalib_1\image_1_c"
if errorlevel 1 exit /b 1

python quick_calib.py "G:\20260224\flircalib_2\image_0_c"
if errorlevel 1 exit /b 1

python quick_calib.py "G:\20260224\flircalib_2\image_1_c"
if errorlevel 1 exit /b 1

echo Done.
endlocal