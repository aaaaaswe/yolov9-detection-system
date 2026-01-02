# Streamlit Web 应用启动指南

## ⚠️ 重要提示

**不要直接使用 `python app.py` 运行！**

Streamlit 应用必须使用 `streamlit run` 命令启动，否则会出现大量警告且界面无法正常显示。

## 正确的启动方法

### 方法 1：使用启动脚本（推荐）

#### Linux/Mac

```bash
cd yolov9_detection/web_app
chmod +x run.sh
./run.sh
```

#### Windows

```cmd
cd yolov9_detection\web_app
run.bat
```

### 方法 2：手动启动

```bash
cd yolov9_detection/web_app
streamlit run app.py
```

### 方法 3：指定端口

如果默认端口（8501）被占用，可以指定其他端口：

```bash
streamlit run app.py --server.port 8502
```

### 方法 4：局域网访问

如果需要在局域网内访问（其他设备访问）：

```bash
streamlit run app.py --server.address 0.0.0.0
```

然后其他设备通过 `http://你的IP:8501` 访问。

## 常见问题

### 1. 错误：Command not found: streamlit

**原因**：未安装 Streamlit

**解决方法**：
```bash
pip install streamlit
```

### 2. 警告：Thread 'MainThread': missing ScriptRunContext!

**原因**：使用了 `python app.py` 而不是 `streamlit run app.py`

**解决方法**：
```bash
# 错误 ❌
python app.py

# 正确 ✅
streamlit run app.py
```

### 3. 端口被占用

**错误信息**：
```
Network error: Address already in use
```

**解决方法**：更换端口
```bash
streamlit run app.py --server.port 8502
```

### 4. 首次启动慢

**原因**：Streamlit 首次启动需要初始化环境

**解决方法**：耐心等待，后续启动会更快

### 5. 浏览器无法访问

**检查清单**：
- [ ] 确认 Streamlit 正在运行（查看终端输出）
- [ ] 检查防火墙设置
- [ ] 尝试使用 `http://localhost:8501` 访问
- [ ] 查看终端显示的实际访问地址

## 启动后的访问

启动成功后，终端会显示类似以下信息：

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

For better performance, install the Watchdog module:

  $ pip install watchdog
```

**本地访问**：点击或访问 `http://localhost:8501`
**局域网访问**：访问 `http://192.168.x.x:8501`（显示的 Network URL）

## 优化建议

### 1. 安装 Watchdog（提高性能）

```bash
pip install watchdog
```

安装后，Streamlit 会更快速地检测文件变化并重新加载。

### 2. 禁用文件监视（提高稳定性）

如果不需要自动重新加载：

```bash
streamlit run app.py --server.runOnSave false
```

### 3. 开启调试模式

```bash
streamlit run app.py --logger.level debug
```

### 4. 自定义主题

创建 `.streamlit/config.toml` 文件：

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 生产环境部署

### 使用 Streamlit Cloud

1. 将代码推送到 GitHub
2. 访问 https://share.streamlit.io/
3. 点击 "New app"
4. 连接你的 GitHub 仓库
5. 配置完成后自动部署

### 使用 Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

构建并运行：

```bash
docker build -t yolov9-web .
docker run -p 8501:8501 yolov9-web
```

### 使用 Nginx 反向代理

Nginx 配置示例：

```nginx
location / {
    proxy_pass http://localhost:8501;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # WebSocket 支持
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## 功能说明

### 📷 图片检测
- 上传图片文件（JPG, PNG, BMP 等）
- 实时显示检测结果
- 下载标注后的图片

### 🎬 视频检测
- 上传视频文件（MP4, AVI, MOV 等）
- 设置最大检测帧数和跳帧参数
- 下载标注后的视频

### 📁 批量检测
- 一次上传多张图片
- 批量处理并显示结果
- 逐个下载检测结果

### 📦 项目资源
- 查看和下载 `.gitignore` 文件
- 下载 `requirements.txt`
- 下载 `README.md`
- 查看项目结构说明

## 配置文件

### .streamlit/config.toml

项目已包含配置文件，主要配置：

```toml
[client]
showErrorDetails = true

[server]
port = 8501
headless = false
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 故障排除

### 日志查看

如果遇到问题，查看 Streamlit 日志：

```bash
streamlit run app.py --logger.level debug
```

### 清除缓存

```bash
streamlit cache clear
```

### 重置配置

删除 `.streamlit` 目录下的配置文件重新生成。

## 技术支持

如有问题，请查看：
- Streamlit 官方文档：https://docs.streamlit.io/
- GitHub Issues：https://github.com/aaaaaswe/yolov9-detection-system/issues

---

**记住：使用 `streamlit run app.py` 启动应用，而不是 `python app.py`！**
