#!/usr/bin/env python3
"""
YOLOv9 实时检测系统 - Web应用
基于Streamlit的在线目标检测平台
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import tempfile
import time
from pathlib import Path

# 添加父目录到路径以导入本地模块
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 尝试导入检测器
try:
    from detect import YOLOv9Detector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"警告: 无法导入 detect 模块: {e}")
    st.warning("使用内置的简化检测器...")
    DETECTOR_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="YOLOv9 实时检测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        padding: 1rem;
    }
    .uploadedFile {
        margin-bottom: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# 简化的检测器类（备用）
class SimpleDetector:
    """简化的 YOLO 检测器（当 detect.py 不可用时使用）"""

    def __init__(self, weights='yolov8s.pt', conf=0.25, iou=0.45, max_det=300):
        """初始化检测器"""
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        # 加载模型
        st.info(f"正在加载模型: {weights}")
        self.model = YOLO(weights)

        # 获取类别名称（使用 COCO 数据集）
        self.class_names = self.model.names if hasattr(self.model, 'names') else [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # 生成颜色
        import random
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]

        st.success(f"模型加载成功! 检测类别: {len(self.class_names)}")


@st.cache_resource
def load_detector(weights='yolov8s.pt', conf=0.25, iou=0.45, device='cpu'):
    """加载检测器（缓存）"""
    try:
        if DETECTOR_AVAILABLE:
            # 使用项目的 YOLOv9Detector
            detector = YOLOv9Detector(
                weights=weights,
                conf=conf,
                iou=iou,
                max_det=300
            )
        else:
            # 使用简化的检测器
            detector = SimpleDetector(
                weights=weights,
                conf=conf,
                iou=iou,
                max_det=300
            )
        return detector
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


def draw_detections(image, results, detector):
    """在图像上绘制检测结果"""
    img = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 获取类别和置信度
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 获取类别名称和颜色
            class_name = detector.class_names[class_id]
            color = [tuple(int(c) for c in detector.colors[class_id])][0]
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(
                img,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    
    return img


def image_detection_page(detector):
    """图片检测页面"""
    st.header("📷 图片检测")
    st.write("上传图片进行目标检测")
    
    # 上传图片
    uploaded_file = st.file_uploader(
        "选择一张图片",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="image_upload"
    )
    
    if uploaded_file is not None:
        # 显示原始图片
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # 检测按钮
        if st.button("开始检测", key="detect_image"):
            with st.spinner("检测中..."):
                # 保存临时图片
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_path = tmp_file.name
                    image.save(tmp_path)
                
                # 读取图片
                img_array = np.array(image)
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # 执行检测
                start_time = time.time()
                results = detector.model(img_array, conf=detector.conf, iou=detector.iou)
                inference_time = time.time() - start_time
                
                # 绘制结果
                annotated_img = draw_detections(img_array, results, detector)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # 显示检测结果
                with col2:
                    st.subheader("检测结果")
                    st.image(annotated_img_rgb, use_column_width=True)
                
                # 显示检测信息
                st.success(f"检测完成! 推理时间: {inference_time:.3f}秒")
                
                # 显示检测到的目标
                if len(results) > 0 and len(results[0].boxes) > 0:
                    st.subheader("检测统计")
                    boxes = results[0].boxes
                    
                    # 统计各类别数量
                    class_counts = {}
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = detector.class_names[class_id]
                        conf = float(box.conf[0])
                        if class_name not in class_counts:
                            class_counts[class_name] = []
                        class_counts[class_name].append(conf)
                    
                    # 显示统计表格
                    for class_name, confidences in class_counts.items():
                        avg_conf = sum(confidences) / len(confidences)
                        st.metric(class_name, len(confidences), f"平均置信度: {avg_conf:.2f}")
                    
                    # 提供下载
                    result_pil = Image.fromarray(annotated_img_rgb)
                    st.download_button(
                        label="下载检测结果",
                        data=result_pil.tobytes(),
                        file_name=f"result_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                else:
                    st.warning("未检测到任何目标")
                
                # 清理临时文件
                os.unlink(tmp_path)


def video_detection_page(detector):
    """视频检测页面"""
    st.header("🎬 视频检测")
    st.write("上传视频文件进行目标检测")
    
    # 上传视频
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_upload"
    )
    
    if uploaded_file is not None:
        # 显示视频信息
        st.info(f"文件名: {uploaded_file.name} | 大小: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # 检测设置
        col1, col2 = st.columns(2)
        with col1:
            max_frames = st.number_input("最大检测帧数", min_value=1, max_value=1000, value=30)
        with col2:
            skip_frames = st.number_input("跳帧数", min_value=0, max_value=30, value=0)
        
        # 检测按钮
        if st.button("开始检测", key="detect_video"):
            with st.spinner("检测中，请稍候..."):
                # 保存临时视频
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(uploaded_file.read())
                
                # 读取视频
                cap = cv2.VideoCapture(tmp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 限制检测帧数
                frames_to_process = min(max_frames, total_frames)
                
                # 准备输出视频
                result_path = tmp_path.replace('.mp4', '_result.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(result_path, fourcc, fps, 
                                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                
                # 创建进度条
                progress_bar = st.progress(0)
                frame_count = 0
                total_detections = 0
                
                while frame_count < frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 跳帧
                    if frame_count % (skip_frames + 1) != 0:
                        frame_count += 1
                        continue
                    
                    # 检测
                    results = detector.model(frame, conf=detector.conf, iou=detector.iou, verbose=False)
                    
                    # 绘制结果
                    annotated_frame = draw_detections(frame, results, detector)
                    out.write(annotated_frame)
                    
                    # 更新进度
                    progress = (frame_count + 1) / frames_to_process
                    progress_bar.progress(progress)
                    
                    if len(results) > 0:
                        total_detections += len(results[0].boxes)
                    
                    frame_count += 1
                
                # 释放资源
                cap.release()
                out.release()
                
                # 显示结果
                st.success(f"检测完成! 处理了 {frame_count} 帧，检测到 {total_detections} 个目标")
                
                # 提供视频下载
                with open(result_path, 'rb') as f:
                    st.download_button(
                        label="下载检测结果视频",
                        data=f.read(),
                        file_name=f"result_{uploaded_file.name}",
                        mime="video/mp4"
                    )
                
                # 清理临时文件
                os.unlink(tmp_path)
                os.unlink(result_path)


def webcam_detection_page(detector):
    """实时摄像头检测页面"""
    st.header("📹 实时摄像头检测")
    st.warning("⚠️ 注意：Streamlit的摄像头功能需要额外配置，建议使用本地Python脚本进行实时检测")
    
    st.info("""
    如果需要在网页上进行实时摄像头检测，有几种选择：
    
    1. **使用本地脚本**: 
       ```bash
       python detect.py --source 0 --weights yolov8s.pt
       ```
    
    2. **使用Streamlit Camera组件** (需要浏览器权限):
       在浏览器中运行，需要HTTPS或localhost
    """)
    
    # 检测设置
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25)
    with col2:
        iou_threshold = st.slider("IOU阈值", 0.0, 1.0, 0.45)
    
    # 说明
    st.markdown("""
    ### 如何使用摄像头检测：
    
    1. **方法一：本地Python脚本**（推荐）
       ```bash
       cd yolov9_detection
       python detect.py --source 0 --weights yolov8s.pt --conf 0.25
       ```
       
    2. **方法二：使用WebRTC**（高级）
       需要额外的Streamlit WebRTC组件
    """)


def batch_detection_page(detector):
    """批量检测页面"""
    st.header("📁 批量图片检测")
    st.write("上传多张图片进行批量检测")
    
    # 上传多张图片
    uploaded_files = st.file_uploader(
        "选择多张图片",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 张图片")
        
        # 检测按钮
        if st.button("开始批量检测", key="detect_batch"):
            results_container = st.container()
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在处理 {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # 读取图片
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # 检测
                results = detector.model(img_array, conf=detector.conf, iou=detector.iou, verbose=False)
                
                # 绘制结果
                annotated_img = draw_detections(img_array, results, detector)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # 保存结果
                result_pil = Image.fromarray(annotated_img_rgb)
                all_results.append({
                    'name': uploaded_file.name,
                    'image': result_pil,
                    'detections': len(results[0].boxes) if len(results) > 0 else 0
                })
                
                # 更新进度
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            # 显示所有结果
            st.success(f"批量检测完成! 处理了 {len(uploaded_files)} 张图片")
            
            # 分页显示结果
            for i, result in enumerate(all_results):
                with st.expander(f"{result['name']} - 检测到 {result['detections']} 个目标"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(result['image'], use_column_width=True)
                    
                    with col2:
                        # 提供单张下载
                        st.download_button(
                            label=f"下载 {result['name']}",
                            data=result['image'].tobytes(),
                            file_name=f"result_{result['name']}",
                            key=f"download_{i}",
                            mime="image/jpeg"
                        )
            
            # 下载所有结果（打包为ZIP需要额外库，暂时不支持）


def resources_page():
    """项目资源页面 - 展示和下载项目文件"""
    st.header("📦 项目资源")
    st.write("查看和下载项目配置文件")
    
    # .gitignore 文件展示
    st.subheader(".gitignore 文件")
    
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        # 显示文件内容
        st.code(gitignore_content, language='text', line_numbers=True)
        
        # 提供下载
        st.download_button(
            label="⬇️ 下载 .gitignore 文件",
            data=gitignore_content.encode('utf-8'),
            file_name=".gitignore",
            mime="text/plain"
        )
        
        # 说明信息
        st.info("""
        ### .gitignore 使用说明
        
        这个 `.gitignore` 文件包含了以下内容：
        
        - Python 编译文件和缓存
        - PyTorch 模型权重文件
        - YOLO 特定文件（runs/, weights/ 等）
        - 虚拟环境目录
        - IDE 配置文件（VSCode, PyCharm 等）
        - 数据和结果文件
        - 临时文件和日志
        
        ### 如何使用
        
        1. 将此文件保存到你的项目根目录
        2. Git 会自动忽略这些文件
        3. 不会被提交到版本控制系统
        
        ### Git 下载项目
        
        如果你有 Git 仓库，可以使用以下命令克隆项目：
        ```bash
        git clone <你的仓库地址>
        cd yolov9_detection
        ```
        """)
    else:
        st.warning("未找到 .gitignore 文件")
    
    # 其他资源文件
    st.markdown("---")
    st.subheader("其他资源")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### requirements.txt
        Python 依赖包列表
        """)
        
        requirements_path = Path(__file__).parent / "requirements.txt"
        if requirements_path.exists():
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            st.download_button(
                label="⬇️ 下载 requirements.txt",
                data=requirements_content.encode('utf-8'),
                file_name="requirements.txt",
                mime="text/plain",
                key="download_requirements"
            )
    
    with col2:
        st.markdown("""
        ### README.md
        项目说明文档
        """)
        
        readme_path = Path(__file__).parent.parent / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            st.download_button(
                label="⬇️ 下载 README.md",
                data=readme_content.encode('utf-8'),
                file_name="README.md",
                mime="text/markdown",
                key="download_readme"
            )
    
    # 项目结构
    st.markdown("---")
    st.subheader("项目结构")
    
    st.code("""
    yolov9_detection/
    ├── .gitignore              # Git 忽略文件配置
    ├── README.md               # 项目说明
    ├── requirements.txt        # Python 依赖
    ├── detect.py              # 检测脚本
    ├── train.py               # 训练脚本
    ├── web_app/               # Web 应用
    │   ├── app.py            # Streamlit 应用主文件
    │   ├── requirements.txt  # Web 应用依赖
    │   └── run.sh            # 启动脚本
    ├── data/                 # 数据集目录
    ├── weights/              # 模型权重目录
    └── runs/                 # 训练和检测结果目录
    """, language='text')


def main():
    """主函数"""
    # 侧边栏
    st.sidebar.title("🎯 YOLOv9 检测系统")
    
    # 模型选择
    st.sidebar.subheader("⚙️ 设置")
    model_size = st.sidebar.selectbox(
        "选择模型",
        options=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        index=1,
        help="n=最快, s=快, m=中等, l=慢, x=最慢但最准确"
    )
    
    conf_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.25)
    iou_threshold = st.sidebar.slider("IOU阈值", 0.0, 1.0, 0.45)
    
    # 重新加载检测器
    if 'detector' not in st.session_state or st.sidebar.button("重新加载模型"):
        st.sidebar.info("正在加载模型...")
        detector = load_detector(
            weights=model_size,
            conf=conf_threshold,
            iou=iou_threshold
        )
        if detector:
            st.session_state['detector'] = detector
            st.sidebar.success("模型加载成功!")
    
    detector = st.session_state.get('detector')
    
    if not detector:
        st.error("模型未加载，请等待模型加载完成")
        st.stop()
    
    # 页面导航
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 功能导航")
    
    page = st.sidebar.radio(
        "选择功能",
        ["📷 图片检测", "🎬 视频检测", "📹 实时摄像头", "📁 批量检测", "📦 项目资源"]
    )
    
    # 主内容
    st.title("🎯 YOLOv9 实时检测系统")
    st.markdown("---")
    
    # 根据选择显示不同页面
    if page == "📦 项目资源":
        # 资源页面不需要加载检测器
        resources_page()
    else:
        # 其他页面需要检测器
        detector = st.session_state.get('detector')
        
        if not detector:
            st.error("模型未加载，请等待模型加载完成")
            st.stop()
        
        if page == "📷 图片检测":
            image_detection_page(detector)
        elif page == "🎬 视频检测":
            video_detection_page(detector)
        elif page == "📹 实时摄像头":
            webcam_detection_page(detector)
        elif page == "📁 批量检测":
            batch_detection_page(detector)
    
    # 页脚
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 关于
    基于YOLOv9的实时目标检测系统
    
    支持的功能：
    - ✅ 图片检测
    - ✅ 视频检测
    - ✅ 批量处理
    - ✅ 多种模型选择
    """)


if __name__ == "__main__":
    main()
