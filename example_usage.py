#!/usr/bin/env python3
"""
YOLOv9 使用示例
展示常用功能的使用方法
"""

from detect import YOLOv9Detector
from train import YOLOv9Trainer
from prepare_dataset import create_yolo_dataset_structure, split_dataset
import cv2


def example_1_create_dataset():
    """示例1: 创建数据集结构"""
    print("\n" + "=" * 60)
    print("示例1: 创建数据集结构")
    print("=" * 60)
    
    classes = ['person', 'car', 'bicycle', 'dog', 'cat']
    dataset_path = 'data/example_dataset'
    
    create_yolo_dataset_structure(dataset_path, classes)
    
    print(f"数据集结构已创建在: {dataset_path}")
    print(f"类别: {classes}")


def example_2_train_model():
    """示例2: 训练模型"""
    print("\n" + "=" * 60)
    print("示例2: 训练模型")
    print("=" * 60)
    
    trainer = YOLOv9Trainer(
        model_size='s',  # 使用小模型快速训练
        data_yaml='data/custom_dataset/data.yaml',
        device='auto'
    )
    
    # 加载预训练模型
    trainer.load_model()
    
    # 开始训练 (示例使用较少的epoch)
    results = trainer.train(
        epochs=10,      # 实际训练时使用更多epoch
        batch=8,
        imgsz=640,
        optimizer='auto',
        lr0=0.01,
        save_period=5
    )
    
    print("训练完成!")


def example_3_detect_image():
    """示例3: 检测单张图片"""
    print("\n" + "=" * 60)
    print("示例3: 检测单张图片")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights='yolov9s.pt',
        conf=0.25,
        iou=0.45
    )
    
    # 检测图片
    results = detector.detect_image(
        image_path='test_image.jpg',
        output_path='result_image.jpg',
        show=False
    )
    
    print("图片检测完成!")


def example_4_detect_webcam():
    """示例4: 摄像头实时检测"""
    print("\n" + "=" * 60)
    print("示例4: 摄像头实时检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights='yolov8s.pt',
        conf=0.30,
        iou=0.45
    )
    
    # 摄像头检测 (按 'q' 退出)
    print("启动摄像头检测，按 'q' 退出...")
    detector.detect_video(
        video_path=0,  # 0 表示默认摄像头
        show=True
    )


def example_5_detect_video():
    """示例5: 视频文件检测"""
    print("\n" + "=" * 60)
    print("示例5: 视频文件检测")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights='yolov9s.pt',
        conf=0.25,
        iou=0.45
    )
    
    # 检测视频
    detector.detect_video(
        video_path='input_video.mp4',
        output_path='output_video.mp4',
        show=False,
        save=True
    )
    
    print("视频检测完成!")


def example_6_batch_detect():
    """示例6: 批量检测文件夹"""
    print("\n" + "=" * 60)
    print("示例6: 批量检测文件夹")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights='yolov9s.pt',
        conf=0.25,
        iou=0.45
    )
    
    # 批量检测
    detector.detect_folder(
        input_folder='test_images/',
        output_folder='results/'
    )
    
    print("批量检测完成!")


def example_7_custom_detect():
    """示例7: 自定义检测逻辑"""
    print("\n" + "=" * 60)
    print("示例7: 自定义检测逻辑")
    print("=" * 60)
    
    detector = YOLOv9Detector(
        weights='yolov9s.pt',
        conf=0.25
    )
    
    # 读取图片
    image = cv2.imread('test_image.jpg')
    
    # 检测
    results = detector.model(image, conf=detector.conf)
    
    # 自定义处理检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            print(f"检测到: {detector.class_names[class_id]}, "
                  f"置信度: {conf:.2f}, "
                  f"位置: ({x1}, {y1}, {x2}, {y2})")
            
            # 自定义逻辑: 只检测人
            if detector.class_names[class_id] == 'person':
                print("  -> 这是一个人!")


def example_8_train_with_custom_weights():
    """示例8: 使用自定义权重训练"""
    print("\n" + "=" * 60)
    print("示例8: 使用自定义权重训练")
    print("=" * 60)
    
    trainer = YOLOv9Trainer(
        model_size='s',
        data_yaml='data/custom_dataset/data.yaml'
    )
    
    # 从上次训练继续
    trainer.load_model(weights='runs/train/exp1/weights/last.pt')
    
    # 继续训练
    results = trainer.train(
        epochs=50,
        batch=16
    )
    
    print("继续训练完成!")


def example_9_export_model():
    """示例9: 导出模型"""
    print("\n" + "=" * 60)
    print("示例9: 导出模型")
    print("=" * 60)
    
    trainer = YOLOv9Trainer()
    
    # 导出为ONNX
    export_path = trainer.export_model(
        weights='yolov9s.pt',
        export_format='onnx'
    )
    
    print(f"模型已导出到: {export_path}")


def example_10_multi_model_ensemble():
    """示例10: 多模型集成"""
    print("\n" + "=" * 60)
    print("示例10: 多模型集成")
    print("=" * 60)
    
    from ultralytics import YOLO
    import numpy as np
    
    # 加载多个模型
    model1 = YOLO('yolov9s.pt')
    model2 = YOLO('yolov9m.pt')
    
    # 读取图片
    image = cv2.imread('test_image.jpg')
    
    # 多个模型预测
    results1 = model1(image, conf=0.25)
    results2 = model2(image, conf=0.25)
    
    # 简单的集成: 只保留两个模型都检测到的目标
    boxes1 = results1[0].boxes.xyxy.cpu().numpy()
    boxes2 = results2[0].boxes.xyxy.cpu().numpy()
    
    # 计算IOU并过滤
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    # 合并结果 (简化示例)
    print(f"模型1检测到 {len(boxes1)} 个目标")
    print(f"模型2检测到 {len(boxes2)} 个目标")
    
    print("多模型集成完成!")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("YOLOv9 使用示例")
    print("=" * 80)
    
    examples = {
        '1': ('创建数据集结构', example_1_create_dataset),
        '2': ('训练模型', example_2_train_model),
        '3': ('检测单张图片', example_3_detect_image),
        '4': ('摄像头实时检测', example_4_detect_webcam),
        '5': ('视频文件检测', example_5_detect_video),
        '6': ('批量检测文件夹', example_6_batch_detect),
        '7': ('自定义检测逻辑', example_7_custom_detect),
        '8': ('使用自定义权重训练', example_8_train_with_custom_weights),
        '9': ('导出模型', example_9_export_model),
        '10': ('多模型集成', example_10_multi_model_ensemble),
    }
    
    print("\n可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请选择要运行的示例 (1-10) 或输入 'all' 运行所有示例: ").strip()
    
    if choice.lower() == 'all':
        for key, (name, func) in examples.items():
            print(f"\n运行示例 {key}: {name}")
            try:
                func()
            except Exception as e:
                print(f"示例 {key} 运行失败: {e}")
    elif choice in examples:
        name, func = examples[choice]
        print(f"\n运行示例: {name}")
        try:
            func()
        except Exception as e:
            print(f"示例运行失败: {e}")
    else:
        print("无效的选择!")


if __name__ == '__main__':
    main()
