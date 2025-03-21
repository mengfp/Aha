cd /d %~dp0
xcopy include\aha.h "C:\Program Files\Aha\include\" /I /Y
xcopy .\x64\Release\aha.lib "C:\Program Files\Aha\lib\" /I /Y
xcopy .\eigen\Eigen "C:\Program Files\Aha\include\Eigen" /I /Y /E
pause