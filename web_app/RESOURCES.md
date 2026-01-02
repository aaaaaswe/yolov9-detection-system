# 项目资源功能说明

## 功能概述

在 YOLOv9 检测系统的 Web 应用中新增了"项目资源"页面，可以方便地查看和下载项目配置文件。

## 主要功能

### 1. .gitignore 文件展示
- 完整显示项目 `.gitignore` 文件内容
- 提供一键下载功能
- 包含详细的使用说明和配置说明

### 2. 其他资源下载
- requirements.txt（Python 依赖包列表）
- README.md（项目说明文档）

### 3. 项目结构展示
- 清晰的项目目录结构说明
- 便于理解项目组织方式

## 使用方法

### 启动 Web 应用
```bash
cd yolov9_detection/web_app
streamlit run app.py
```

### 访问项目资源页面
1. 在左侧侧边栏找到"📋 功能导航"
2. 点击"📦 项目资源"
3. 查看和下载需要的文件

### 下载 .gitignore 文件
1. 在"项目资源"页面中找到 `.gitignore` 部分
2. 点击"⬇️ 下载 .gitignore 文件"按钮
3. 文件将自动下载到你的本地

## .gitignore 文件包含的内容

该文件已针对 YOLOv9 项目进行了优化，包含以下配置：

- **Python 相关**: `__pycache__/`, `*.pyc`, 虚拟环境等
- **PyTorch 相关**: `*.pt`, `*.pth`, `runs/` 等
- **YOLO 特定**: 模型权重、数据集、训练结果等
- **IDE 配置**: VSCode, PyCharm 等
- **数据文件**: 数据集、结果输出、日志等

## Git 下载项目

如果你想通过 Git 克隆整个项目，使用以下命令：

```bash
git clone <你的仓库地址>
cd yolov9_detection
```

克隆后，`.gitignore` 文件会自动生效，忽略配置中指定的文件。

## 注意事项

1. 确保 `.gitignore` 文件位于项目根目录
2. 修改 `.gitignore` 后，已跟踪的文件需要手动使用 `git rm --cached` 移除
3. 下载的文件与项目中的文件完全一致

## 示例：更新 .gitignore 后的清理命令

如果你修改了 `.gitignore` 并希望清理已跟踪的文件：

```bash
# 从 Git 缓存中移除所有文件
git rm -r --cached .

# 重新添加所有文件（应用新的 .gitignore 规则）
git add .

# 提交更改
git commit -m "Update .gitignore"
```
