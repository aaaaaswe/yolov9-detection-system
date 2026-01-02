@echo off
REM Windows 环境修复脚本

echo ========================================
echo   YOLOv9 Web 应用环境修复
echo ========================================
echo.

echo [1/3] 检查 Python 安装...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    echo 请从 https://www.python.org/downloads/ 下载并安装 Python
    pause
    exit /b 1
)
echo Python 已安装
python --version
echo.

echo [2/3] 检查并安装依赖...
echo 安装 Streamlit...
pip install streamlit
if errorlevel 1 (
    echo Streamlit 安装失败
    pause
    exit /b 1
)

echo.
echo 安装 Ultralytics...
pip install ultralytics
if errorlevel 1 (
    echo Ultralytics 安装失败
    pause
    exit /b 1
)

echo.
echo 安装其他依赖...
pip install opencv-python numpy pillow
if errorlevel 1 (
    echo 其他依赖安装失败
    pause
    exit /b 1
)

echo.
echo [3/3] 检查项目文件...
if not exist "detect.py" (
    echo 警告: detect.py 不存在
    echo 请确保在项目根目录下运行此脚本
)

echo.
echo ========================================
echo   环境检查完成
echo ========================================
echo.
echo 现在可以启动 Web 应用了:
echo.
echo 方式 1 (使用完整路径):
echo   C:\Python\Python310\Scripts\streamlit.exe run web_app\app.py
echo.
echo 方式 2 (如果 Streamlit 已添加到 PATH):
echo   streamlit run web_app\app.py
echo.
echo 方式 3 (使用启动脚本):
echo   cd web_app
echo   run.bat
echo.

echo 按任意键继续...
pause >nul
