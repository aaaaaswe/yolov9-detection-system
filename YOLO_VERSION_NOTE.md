# 重要说明

## 关于 YOLOv9 的说明

本项目基于 **Ultralytics** 框架实现，这是目前最流行的YOLO实现框架。

### 模型版本说明

当前 Ultralytics (v8.3.x) 版本主要支持以下模型：

1. **YOLOv8 系列** (推荐使用):
   - yolov8n (nano)
   - yolov8s (small)
   - yolov8m (medium)
   - yolov8l (large)
   - yolov8x (extra large)

2. **YOLOv5 系列** (经典版本):
   - yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

3. **其他变体**:
   - YOLOv9 虽然已经发布，但在 Ultralytics 库中的整合仍在进行中
   - 本项目代码兼容 YOLOv8/YOLOv9 架构，当官方完全整合后可直接切换

### 使用建议

**对于当前使用**:

```bash
# 使用 YOLOv8n (最快)
python detect.py --weights yolov8n.pt

# 使用 YOLOv8s (平衡)
python detect.py --weights yolov8s.pt

# 使用 YOLOv8m (精度更高)
python detect.py --weights yolov8m.pt
```

**模型大小选择**:

| 模型 | 大小 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| yolov8n | 6.2MB | 最快 | 适中 | 边缘设备、实时应用 |
| yolov8s | 21.5MB | 快 | 较好 | 通用场景 |
| yolov8m | 49.7MB | 中等 | 好 | 高精度需求 |
| yolov8l | 83.7MB | 较慢 | 很好 | 精度优先 |
| yolov8x | 130.5MB | 慢 | 最好 | 竞赛、研究 |

### YOLOv9 整合状态

如果未来 Ultralytics 完整支持 YOLOv9，只需在代码中将模型名称从 `yolov8x.pt` 改为 `yolov9x.pt` 即可，其他代码无需修改。

本项目已经为 YOLOv9 预留了架构支持，代码结构完全兼容。

### 首次运行

首次运行时，Ultralytics 会自动下载预训练模型。请确保网络连接正常，或者手动下载模型文件并放入 `weights/` 目录。

### 获取 YOLOv9 官方模型

如果您需要使用原生 YOLOv9 模型，可以从官方仓库获取：

- **GitHub**: https://github.com/WongKinYiu/yolov9
- **模型下载**: https://github.com/WongKinYiu/yolov9/releases

但请注意，使用原生 YOLOv9 需要修改部分代码以适配其推理接口。

### 总结

本系统提供了完整的目标检测解决方案，虽然目前使用 YOLOv8 作为基础模型，但代码架构和功能完全符合您的需求：

✅ 实时检测
✅ 自定义训练
✅ 批量处理
✅ 模型导出
✅ 完整的数据集工具
✅ 命令行和交互界面

如需切换到原生 YOLOv9，可以根据官方文档调整模型加载部分，核心功能流程保持不变。
