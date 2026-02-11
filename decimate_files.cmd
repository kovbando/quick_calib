@echo off
REM ==============================================================
REM  Deletes all but every Nth image file (default: keep every 10th)
REM  Step 1: Preview deletions using -WhatIf
REM  Step 2: Ask for confirmation before actually deleting
REM ==============================================================

setlocal

REM --- CONFIGURATION ---
set "KEEP_EVERY=2"
set "PATTERN=frame_*.jpg"

echo.
echo ==============================================================
echo  Keep every %KEEP_EVERY%th file matching "%PATTERN%" in:
echo  %cd%
echo ==============================================================
echo.
echo  Step 1: Previewing which files *would* be deleted...
echo --------------------------------------------------------------
echo.

powershell -NoProfile -Command ^
  "$i=0; Get-ChildItem -Filter '%PATTERN%' | Sort-Object Name | ForEach-Object { if ($i %% %KEEP_EVERY% -ne 0) { Remove-Item $_ -WhatIf }; $i++ }"

echo.
echo --------------------------------------------------------------
echo  Above is the preview list (no deletions yet).
echo.

choice /M "Do you want to proceed and actually delete those files?"

if errorlevel 2 (
    echo.
    echo Aborted. No files were deleted.
    goto :end
)

echo.
echo Proceeding with deletion...
echo --------------------------------------------------------------
powershell -NoProfile -Command ^
  "$i=0; Get-ChildItem -Filter '%PATTERN%' | Sort-Object Name | ForEach-Object { if ($i %% %KEEP_EVERY% -ne 0) { Remove-Item $_ }; $i++ }"

echo.
echo --------------------------------------------------------------
echo Done! All unnecessary files were deleted.
echo.

:end
pause
endlocal
