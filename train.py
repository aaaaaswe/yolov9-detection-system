#!/usr/bin/env python3
"""
YOLOv9训练脚本
支持自定义数据集训练和预训练模型微调
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import yaml
from ultralytics import YOLO
from datetime import datetime
import shutil


class YOLOv9Trainer:
    """YOLOv9训练器类"""
    
    def __init__(self, model_size='y', data_yaml=None, device=None):
        """
        初始化训练器
        
        Args:
            model_size: 模型大小 (n, s, m, l, x)
            data_yaml: 数据集配置文件路径
            device: 训练设备 (auto, cpu, 0, 1, ...)
        """
        self.model_size = model_size.lower()
        self.data_yaml = data_yaml
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型映射 (当前使用YOLOv8，待YOLOv9完全整合后可切换)
        self.model_map = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt',
            't': 'yolov8t.pt'
        }
        
        # 创建输出目录
        self.runs_dir = Path('runs/train')
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化训练器:")
        print(f"  模型大小: {model_size}")
        print(f"  数据集配置: {data_yaml}")
        print(f"  设备: {self.device}")
    
    def load_model(self, weights=None):
        """
        加载YOLOv9模型
        
        Args:
            weights: 预训练权重路径，如果为None则使用官方权重
        """
        if weights and os.path.exists(weights):
            print(f"加载自定义权重: {weights}")
            self.model = YOLO(weights)
        else:
            model_name = self.model_map.get(self.model_size, 'yolov8s.pt')
            print(f"加载官方模型: {model_name}")
            self.model = YOLO(model_name)
        
        return self.model
    
    def validate_dataset(self):
        """验证数据集配置"""
        if not self.data_yaml or not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"数据集配置文件不存在: {self.data_yaml}")
        
        # 读取配置
        with open(self.data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 验证路径
        dataset_path = data_config.get('path')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 验证图像和标签目录
        for split in ['train', 'val']:
            img_dir = os.path.join(dataset_path, data_config[split])
            if not os.path.exists(img_dir):
                print(f"警告: {split}图像目录不存在: {img_dir}")
        
        print(f"数据集验证通过:")
        print(f"  类别数: {data_config.get('nc')}")
        print(f"  类别名称: {data_config.get('names')}")
        print(f"  数据集路径: {dataset_path}")
        
        return data_config
    
    def train(self, epochs=100, batch=16, imgsz=640, 
              optimizer='auto', lr0=0.01, patience=50, 
              save=True, project=None, name=None, resume=None):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 图像大小
            optimizer: 优化器 (SGD, Adam, AdamW, auto)
            lr0: 初始学习率
            patience: 早停耐心值
            save: 是否保存模型
            project: 项目名称
            name: 运行名称
            resume: 恢复训练路径
        """
        # 验证数据集
        self.validate_dataset()
        
        # 加载模型
        if not hasattr(self, 'model'):
            self.load_model()
        
        # 设置运行名称
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not name:
            name = f'yolov9_{self.model_size}_{timestamp}'
        
        print(f"\n开始训练...")
        print(f"  轮数: {epochs}")
        print(f"  批次: {batch}")
        print(f"  图像大小: {imgsz}")
        print(f"  优化器: {optimizer}")
        print(f"  学习率: {lr0}")
        print(f"  运行名称: {name}")
        
        # 训练参数
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'optimizer': optimizer,
            'lr0': lr0,
            'patience': patience,
            'save': save,
            'project': project if project else str(self.runs_dir),
            'name': name,
            'device': self.device,
            'verbose': True,
            'plots': True,
            'save_period': 10,  # 每10轮保存一次
        }
        
        # 如果恢复训练
        if resume:
            print(f"从 {resume} 恢复训练")
            train_args['resume'] = resume
        
        # 开始训练
        results = self.model.train(**train_args)
        
        print(f"\n训练完成!")
        print(f"最佳模型保存位置: {self.runs_dir / name}")
        
        # 返回训练结果
        return results
    
    def export_model(self, weights_path, export_format='onnx'):
        """
        导出模型
        
        Args:
            weights_path: 模型权重路径
            export_format: 导出格式 (onnx, torchscript, coreml, tflite, etc.)
        """
        print(f"导出模型为 {export_format} 格式...")
        
        # 加载模型
        model = YOLO(weights_path)
        
        # 导出
        export_path = model.export(format=export_format)
        
        print(f"模型已导出到: {export_path}")
        return export_path
    
    def hyperparameter_tuning(self, iterations=30, epochs=30, space=None):
        """
        超参数调优
        
        Args:
            iterations: 迭代次数
            epochs: 每次训练的轮数
            space: 超参数搜索空间
        """
        print(f"开始超参数调优...")
        
        # 加载模型
        if not hasattr(self, 'model'):
            self.load_model()
        
        # 默认超参数空间
        default_space = {
            'lr0': [0.001, 0.01, 0.1],
            'lrf': [0.01, 0.1],
            'momentum': [0.8, 0.937, 0.99],
            'weight_decay': [0.0005, 0.001],
            'warmup_epochs': [3.0, 5.0],
            'warmup_momentum': [0.0, 0.8],
            'box': [7.5, 10.0],
            'cls': [0.5, 1.0],
        }
        
        space = space if space else default_space
        
        # 超参数调优
        results = self.model.tune(
            data=self.data_yaml,
            space=space,
            iterations=iterations,
            epochs=epochs,
            device=self.device
        )
        
        print(f"超参数调优完成!")
        print(f"最佳超参数: {results}")
        
        return results


def train_from_scratch(args):
    """从头训练"""
    print("=" * 60)
    print("模式: 从头训练")
    print("=" * 60)
    
    trainer = YOLOv9Trainer(
        model_size=args.model_size,
        data_yaml=args.data,
        device=args.device
    )
    
    # 加载模型
    if args.weights:
        trainer.load_model(args.weights)
    else:
        trainer.load_model()
    
    # 开始训练
    results = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr,
        patience=args.patience,
        save_period=args.save_period,
        project=args.project,
        name=args.name
    )
    
    return results


def resume_training(args):
    """恢复训练"""
    print("=" * 60)
    print("模式: 恢复训练")
    print("=" * 60)
    
    if not args.resume:
        print("错误: 需要指定 --resume 参数")
        return
    
    trainer = YOLOv9Trainer(
        model_size=args.model_size,
        data_yaml=args.data,
        device=args.device
    )
    
    results = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    return results


def export_model(args):
    """导出模型"""
    print("=" * 60)
    print("模式: 导出模型")
    print("=" * 60)
    
    trainer = YOLOv9Trainer()
    export_path = trainer.export_model(args.weights, args.format)
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv9训练工具')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'resume', 'export', 'tune'],
                       help='运行模式: train(训练), resume(恢复), export(导出), tune(超参数调优)')
    
    # 数据集参数
    parser.add_argument('--data', type=str, default='data/custom_dataset/data.yaml',
                       help='数据集配置文件路径')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='y',
                       choices=['n', 's', 'm', 'l', 'x', 't'],
                       help='模型大小')
    parser.add_argument('--weights', type=str, help='预训练权重路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='优化器')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--save_period', type=int, default=10, help='保存周期')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, help='恢复训练路径')
    
    # 导出参数
    parser.add_argument('--format', type=str, default='onnx',
                       help='导出格式')
    
    # 超参数调优
    parser.add_argument('--tune_iterations', type=int, default=30, help='调优迭代次数')
    
    # 其他参数
    parser.add_argument('--device', type=str, help='训练设备 (auto, cpu, 0, 1, ...)')
    parser.add_argument('--project', type=str, help='项目名称')
    parser.add_argument('--name', type=str, help='运行名称')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_from_scratch(args)
    elif args.mode == 'resume':
        resume_training(args)
    elif args.mode == 'export':
        export_model(args)
    elif args.mode == 'tune':
        print("超参数调优功能暂未完全实现，请使用 train 模式")


if __name__ == '__main__':
    main()
