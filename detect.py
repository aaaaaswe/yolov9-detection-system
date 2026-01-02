#!/usr/bin/env python3
"""
YOLOv9实时检测脚本
支持视频流、摄像头、图片和文件夹检测
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import threading
import queue


class YOLOv9Detector:
    """YOLOv9检测器类"""
    
    def __init__(self, weights, device=None, conf=0.25, iou=0.45, max_det=300):
        """
        初始化检测器
        
        Args:
            weights: 模型权重路径
            device: 运行设备
            conf: 置信度阈值
            iou: IOU阈值
            max_det: 最大检测数量
        """
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        
        # 检查设备
        import torch
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"加载模型: {weights}")
        self.model = YOLO(weights)
        print(f"模型加载成功! 设备: {self.device}")
        
        # 获取类别名称
        self.class_names = self.model.names
        
        # 生成随机颜色
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
    
    def preprocess(self, image):
        """预处理图像"""
        return image
    
    def draw_detections(self, image, results, show_labels=True, show_conf=True):
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            results: 检测结果
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
        """
        img = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 获取类别和置信度
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 跳过低置信度检测
                if conf < self.conf:
                    continue
                
                # 获取类别名称和颜色
                class_name = self.class_names[class_id]
                color = self.colors[class_id].tolist()
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if show_labels:
                    label = f"{class_name}"
                    if show_conf:
                        label += f" {conf:.2f}"
                    
                    # 计算标签位置
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
    
    def detect_image(self, image_path, output_path=None, show=False):
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            output_path: 输出路径
            show: 是否显示结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return None
        
        print(f"检测图片: {image_path}")
        
        # 检测
        start_time = time.time()
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device
        )
        inference_time = time.time() - start_time
        
        # 绘制结果
        annotated_image = self.draw_detections(image, results)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"结果已保存: {output_path}")
        
        # 显示结果
        if show:
            cv2.imshow('Detection Result', annotated_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 打印统计信息
        for result in results:
            print(f"检测到 {len(result.boxes)} 个目标")
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  - {self.class_names[class_id]}: {conf:.2f}")
        
        print(f"推理时间: {inference_time:.3f}s")
        
        return results
    
    def detect_video(self, video_path, output_path=None, show=True, save=False):
        """
        检测视频
        
        Args:
            video_path: 视频路径或摄像头索引 (0, 1, ...)
            output_path: 输出视频路径
            show: 是否显示结果
            save: 是否保存结果
        """
        # 打开视频
        if isinstance(video_path, int) or video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
            print(f"打开摄像头: {video_path}")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"打开视频: {video_path}")
        
        if not cap.isOpened():
            print("错误: 无法打开视频流")
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        
        # 创建视频写入器
        if save and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"保存视频到: {output_path}")
        else:
            out = None
        
        # 统计信息
        frame_count = 0
        total_time = 0
        
        print("\n开始检测，按 'q' 退出...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频结束或无法读取帧")
                break
            
            frame_count += 1
            
            # 检测
            start_time = time.time()
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 绘制结果
            annotated_frame = self.draw_detections(frame, results)
            
            # 显示FPS
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示结果
            if show:
                cv2.imshow('YOLOv9 Detection', annotated_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户退出检测")
                    break
            
            # 保存结果
            if out:
                out.write(annotated_frame)
            
            # 打印进度
            if frame_count % 30 == 0:
                print(f"进度: {frame_count}/{total_frames} 帧, 平均FPS: {frame_count/total_time:.1f}")
        
        # 释放资源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n检测完成!")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均FPS: {avg_fps:.1f}")
    
    def detect_folder(self, input_folder, output_folder=None):
        """
        检测文件夹中的所有图片
        
        Args:
            input_folder: 输入文件夹
            output_folder: 输出文件夹
        """
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(input_folder).glob(ext))
        
        if not image_files:
            print(f"错误: {input_folder} 中没有找到图片")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 创建输出文件夹
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # 检测每张图片
        for i, image_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 检测: {image_file.name}")
            
            output_path = None
            if output_folder:
                output_path = os.path.join(output_folder, image_file.name)
            
            self.detect_image(
                str(image_file),
                output_path=output_path,
                show=False
            )
        
        print(f"\n文件夹检测完成!")


def detect_camera(args):
    """摄像头检测"""
    print("=" * 60)
    print("模式: 摄像头检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det
    )
    
    detector.detect_video(
        video_path=args.source if args.source else 0,
        output_path=args.output if args.save else None,
        show=not args.hide,
        save=args.save
    )


def detect_video(args):
    """视频检测"""
    print("=" * 60)
    print("模式: 视频检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det
    )
    
    detector.detect_video(
        video_path=args.source,
        output_path=args.output if args.save else None,
        show=not args.hide,
        save=args.save
    )


def detect_image(args):
    """图片检测"""
    print("=" * 60)
    print("模式: 图片检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det
    )
    
    detector.detect_image(
        image_path=args.source,
        output_path=args.output,
        show=not args.hide
    )


def detect_folder(args):
    """文件夹检测"""
    print("=" * 60)
    print("模式: 批量检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det
    )
    
    output_folder = None
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.join(os.path.dirname(args.source), 'results')
    
    detector.detect_folder(
        input_folder=args.source,
        output_folder=output_folder
    )


def main():
    parser = argparse.ArgumentParser(description='YOLOv9检测工具')
    
    # 输入输出参数
    parser.add_argument('--source', type=str, help='输入源 (摄像头索引/视频/图片/文件夹)')
    parser.add_argument('--output', type=str, help='输出路径')
    
    # 模型参数
    parser.add_argument('--weights', type=str, default='yolov8s.pt',
                       help='模型权重路径')
    parser.add_argument('--device', type=str, help='运行设备 (auto, cpu, 0, 1, ...)')
    
    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--max_det', type=int, default=300, help='最大检测数量')
    
    # 显示和保存参数
    parser.add_argument('--hide', action='store_true', help='不显示结果')
    parser.add_argument('--save', action='store_true', help='保存结果')
    
    args = parser.parse_args()
    
    # 自动检测输入类型
    if not args.source:
        # 默认使用摄像头
        detect_camera(args)
    elif os.path.isfile(args.source):
        # 检查文件类型
        ext = os.path.splitext(args.source)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            detect_video(args)
        else:
            detect_image(args)
    elif os.path.isdir(args.source):
        detect_folder(args)
    else:
        # 尝试作为摄像头索引
        try:
            idx = int(args.source)
            detect_camera(args)
        except ValueError:
            print(f"错误: 无效的输入源 {args.source}")
            return


if __name__ == '__main__':
    main()
