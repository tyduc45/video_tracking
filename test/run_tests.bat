@echo off
REM Windows 批处理测试运行脚本

echo ================================
echo 推理系统完整测试
echo ================================

cd /d "%~dp0.."

REM 检查依赖
echo.
echo 📦 检查依赖...
python -m pip install pytest pytest-cov -q

REM 运行测试
echo.
echo 🧪 运行测试...
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
python -m pytest test/ -v --tb=short --junit-xml=test_reports/report_%mydate%_%mytime%.xml

exit /b %errorlevel%
