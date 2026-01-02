#!/usr/bin/env python3
"""
YOLOv9 图形化界面
基于 PyQt6 的桌面应用程序
"""

import sys
import os
import subprocess
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QFileDialog, QGroupBox, QFormLayout,
    QMessageBox, QProgressBar, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon


class WorkerThread(QThread):
    """工作线程 - 执行耗时操作"""
    
    output_signal = pyqtSignal(str)  # 输出信号
    finished_signal = pyqtSignal(int, str)  # 完成信号 (状态码, 消息)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
    
    def run(self):
        """执行命令"""
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取输出
            for line in process.stdout:
                self.output_signal.emit(line.rstrip())
            
            process.wait()
            self.finished_signal.emit(process.returncode, "执行完成")
            
        except Exception as e:
            self.output_signal.emit(f"错误: {str(e)}")
            self.finished_signal.emit(-1, str(e))


class DatasetPrepareWidget(QWidget):
    """数据集准备页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 操作模式选择
        mode_group = QGroupBox("操作模式")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["create", "split", "visualize"])
        mode_layout.addWidget(QLabel("模式:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 路径设置
        path_group = QGroupBox("路径设置")
        path_layout = QFormLayout()
        
        self.dataset_path = QLineEdit("data/custom_dataset")
        self.dataset_path_btn = QPushButton("浏览...")
        self.dataset_path_btn.clicked.connect(self.select_dataset_path)
        
        path_layout.addRow("数据集路径:", self.create_path_row(self.dataset_path, self.dataset_path_btn))
        
        self.image_dir = QLineEdit()
        self.image_dir_btn = QPushButton("浏览...")
        self.image_dir_btn.clicked.connect(self.select_image_dir)
        
        path_layout.addRow("图像目录:", self.create_path_row(self.image_dir, self.image_dir_btn))
        
        self.label_dir = QLineEdit()
        self.label_dir_btn = QPushButton("浏览...")
        self.label_dir_btn.clicked.connect(self.select_label_dir)
        
        path_layout.addRow("标签目录:", self.create_path_row(self.label_dir, self.label_dir_btn))
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 类别设置（仅 create 模式）
        class_group = QGroupBox("类别设置")
        class_layout = QFormLayout()
        
        self.classes_edit = QLineEdit("person car dog")
        class_layout.addRow("类别名称 (空格分隔):", self.classes_edit)
        
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)
        
        # 数据划分（仅 split 模式）
        split_group = QGroupBox("数据划分")
        split_layout = QFormLayout()
        
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0, 1)
        self.train_ratio.setSingleStep(0.1)
        self.train_ratio.setValue(0.7)
        
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0, 1)
        self.val_ratio.setSingleStep(0.1)
        self.val_ratio.setValue(0.2)
        
        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0, 1)
        self.test_ratio.setSingleStep(0.1)
        self.test_ratio.setValue(0.1)
        
        split_layout.addRow("训练集比例:", self.train_ratio)
        split_layout.addRow("验证集比例:", self.val_ratio)
        split_layout.addRow("测试集比例:", self.test_ratio)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # 执行按钮
        btn_layout = QHBoxLayout()
        self.execute_btn = QPushButton("执行")
        self.execute_btn.clicked.connect(self.execute)
        self.execute_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        btn_layout.addWidget(self.execute_btn)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_path_row(self, line_edit, button):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        widget.setLayout(layout)
        return widget
    
    def select_dataset_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集路径")
        if path:
            self.dataset_path.setText(path)
    
    def select_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if path:
            self.image_dir.setText(path)
    
    def select_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标签目录")
        if path:
            self.label_dir.setText(path)
    
    def execute(self):
        """执行数据集准备"""
        mode = self.mode_combo.currentText()
        dataset_path = self.dataset_path.text()
        
        cmd = [
            sys.executable, 'prepare_dataset.py',
            '--mode', mode,
            '--dataset_path', dataset_path
        ]
        
        if mode == 'create':
            classes = self.classes_edit.text()
            if classes:
                cmd.extend(['--classes'] + classes.split())
        elif mode == 'split':
            image_dir = self.image_dir.text()
            label_dir = self.label_dir.text()
            if image_dir:
                cmd.extend(['--image_dir', image_dir])
            if label_dir:
                cmd.extend(['--label_dir', label_dir])
            cmd.extend([
                '--train_ratio', str(self.train_ratio.value()),
                '--val_ratio', str(self.val_ratio.value()),
                '--test_ratio', str(self.test_ratio.value())
            ])
        
        self.parent_window.execute_command(cmd)


class TrainWidget(QWidget):
    """训练页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        self.model_size = QComboBox()
        self.model_size.addItems(["n", "s", "m", "l", "x"])
        self.model_size.setCurrentText("y")
        
        self.weights = QLineEdit("yolov8s.pt")
        self.weights_btn = QPushButton("浏览...")
        self.weights_btn.clicked.connect(self.select_weights)
        
        model_layout.addRow("模型大小:", self.model_size)
        model_layout.addRow("预训练权重:", self.create_path_row(self.weights, self.weights_btn))
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 数据集设置
        data_group = QGroupBox("数据集设置")
        data_layout = QFormLayout()
        
        self.data_path = QLineEdit("data/custom_dataset/data.yaml")
        self.data_path_btn = QPushButton("浏览...")
        self.data_path_btn.clicked.connect(self.select_data_path)
        
        data_layout.addRow("数据集配置:", self.create_path_row(self.data_path, self.data_path_btn))
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 训练参数
        param_group = QGroupBox("训练参数")
        param_layout = QFormLayout()
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(16)
        
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 2048)
        self.imgsz.setValue(640)
        
        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "Adam", "AdamW"])
        
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.0001, 1.0)
        self.lr.setDecimals(4)
        self.lr.setValue(0.01)
        
        self.patience = QSpinBox()
        self.patience.setRange(1, 100)
        self.patience.setValue(50)
        
        self.device = QLineEdit("0")
        
        param_layout.addRow("训练轮数:", self.epochs)
        param_layout.addRow("批次大小:", self.batch)
        param_layout.addRow("图像尺寸:", self.imgsz)
        param_layout.addRow("优化器:", self.optimizer)
        param_layout.addRow("学习率:", self.lr)
        param_layout.addRow("早停轮数:", self.patience)
        param_layout.addRow("设备 (CPU:cpu GPU:0):", self.device)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 执行按钮
        btn_layout = QHBoxLayout()
        self.execute_btn = QPushButton("开始训练")
        self.execute_btn.clicked.connect(self.execute)
        self.execute_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        btn_layout.addWidget(self.execute_btn)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_path_row(self, line_edit, button):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        widget.setLayout(layout)
        return widget
    
    def select_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "*.pt")
        if path:
            self.weights.setText(path)
    
    def select_data_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集配置", "", "*.yaml")
        if path:
            self.data_path.setText(path)
    
    def execute(self):
        """执行训练"""
        cmd = [
            sys.executable, 'train.py',
            '--mode', 'train',
            '--data', self.data_path.text(),
            '--model_size', self.model_size.currentText(),
            '--epochs', str(self.epochs.value()),
            '--batch', str(self.batch.value()),
            '--imgsz', str(self.imgsz.value()),
            '--optimizer', self.optimizer.currentText(),
            '--lr', str(self.lr.value()),
            '--patience', str(self.patience.value()),
            '--device', self.device.text()
        ]
        
        weights_text = self.weights.text()
        if weights_text:
            cmd.extend(['--weights', weights_text])
        
        self.parent_window.execute_command(cmd)


class DetectWidget(QWidget):
    """检测页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 输入源设置
        source_group = QGroupBox("输入源设置")
        source_layout = QFormLayout()
        
        self.source = QLineEdit("0")
        self.source_btn = QPushButton("浏览...")
        self.source_btn.clicked.connect(self.select_source)
        
        source_layout.addRow("输入源 (0/文件路径):", self.create_path_row(self.source, self.source_btn))
        
        self.weights = QLineEdit("yolov8s.pt")
        self.weights_btn = QPushButton("浏览...")
        self.weights_btn.clicked.connect(self.select_weights)
        
        source_layout.addRow("模型权重:", self.create_path_row(self.weights, self.weights_btn))
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # 检测参数
        param_group = QGroupBox("检测参数")
        param_layout = QFormLayout()
        
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0, 1)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0, 1)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.45)
        
        self.max_det = QSpinBox()
        self.max_det.setRange(1, 1000)
        self.max_det.setValue(300)
        
        self.device = QLineEdit("0")
        
        self.output = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.select_output)
        
        param_layout.addRow("置信度阈值:", self.conf)
        param_layout.addRow("IOU 阈值:", self.iou)
        param_layout.addRow("最大检测数:", self.max_det)
        param_layout.addRow("设备 (CPU:cpu GPU:0):", self.device)
        param_layout.addRow("输出路径:", self.create_path_row(self.output, self.output_btn))
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 选项
        option_layout = QHBoxLayout()
        self.save_check = QCheckBox("保存结果")
        self.hide_check = QCheckBox("隐藏窗口")
        option_layout.addWidget(self.save_check)
        option_layout.addWidget(self.hide_check)
        layout.addLayout(option_layout)
        
        # 执行按钮
        btn_layout = QHBoxLayout()
        self.execute_btn = QPushButton("开始检测")
        self.execute_btn.clicked.connect(self.execute)
        self.execute_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        btn_layout.addWidget(self.execute_btn)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_path_row(self, line_edit, button):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        widget.setLayout(layout)
        return widget
    
    def select_source(self):
        # 检查是否是文件
        path, _ = QFileDialog.getOpenFileName(self, "选择输入源", "", "所有文件 (*.*)")
        if path:
            self.source.setText(path)
    
    def select_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "*.pt")
        if path:
            self.weights.setText(path)
    
    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output.setText(path)
    
    def execute(self):
        """执行检测"""
        cmd = [
            sys.executable, 'detect.py',
            '--source', self.source.text(),
            '--weights', self.weights.text(),
            '--conf', str(self.conf.value()),
            '--iou', str(self.iou.value()),
            '--max_det', str(self.max_det.value()),
            '--device', self.device.text()
        ]
        
        if self.save_check.isChecked():
            cmd.append('--save')
        
        if self.hide_check.isChecked():
            cmd.append('--hide')
        
        output_text = self.output.text()
        if output_text:
            cmd.extend(['--output', output_text])
        
        self.parent_window.execute_command(cmd)


class ExportWidget(QWidget):
    """导出页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        self.weights = QLineEdit()
        self.weights_btn = QPushButton("浏览...")
        self.weights_btn.clicked.connect(self.select_weights)
        
        model_layout.addRow("模型权重:", self.create_path_row(self.weights, self.weights_btn))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["onnx", "torchscript", "coreml", "tflite"])
        
        model_layout.addRow("导出格式:", self.format_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 执行按钮
        btn_layout = QHBoxLayout()
        self.execute_btn = QPushButton("导出模型")
        self.execute_btn.clicked.connect(self.execute)
        self.execute_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        btn_layout.addWidget(self.execute_btn)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_path_row(self, line_edit, button):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        widget.setLayout(layout)
        return widget
    
    def select_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "*.pt")
        if path:
            self.weights.setText(path)
    
    def execute(self):
        """执行导出"""
        cmd = [
            sys.executable, 'train.py',
            '--mode', 'export',
            '--weights', self.weights.text(),
            '--format', self.format_combo.currentText()
        ]
        
        self.parent_window.execute_command(cmd)


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("YOLOv9 图形化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("YOLOv9 实时检测系统")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        self.dataset_widget = DatasetPrepareWidget(self)
        self.train_widget = TrainWidget(self)
        self.detect_widget = DetectWidget(self)
        self.export_widget = ExportWidget(self)
        
        self.tab_widget.addTab(self.dataset_widget, "📁 数据集准备")
        self.tab_widget.addTab(self.train_widget, "🚀 训练模型")
        self.tab_widget.addTab(self.detect_widget, "🎯 目标检测")
        self.tab_widget.addTab(self.export_widget, "📦 导出模型")
        
        main_layout.addWidget(self.tab_widget)
        
        # 日志输出区域
        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet("font-family: monospace; background-color: #1e1e1e; color: #d4d4d4;")
        
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def execute_command(self, command):
        """执行命令"""
        # 检查是否有正在运行的任务
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "警告", "正在执行任务，请等待完成")
            return
        
        # 清空日志
        self.log_output.clear()
        
        # 显示命令
        self.log_output.append(f"执行命令: {' '.join(command)}")
        self.log_output.append("-" * 80)
        
        # 创建并启动工作线程
        self.worker_thread = WorkerThread(command)
        self.worker_thread.output_signal.connect(self.append_log)
        self.worker_thread.finished_signal.connect(self.on_command_finished)
        self.worker_thread.start()
        
        self.statusBar().showMessage("执行中...")
    
    def append_log(self, text):
        """追加日志"""
        self.log_output.append(text)
        # 自动滚动到底部
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_command_finished(self, returncode, message):
        """命令完成处理"""
        if returncode == 0:
            self.log_output.append("-" * 80)
            self.log_output.append(f"✓ {message}")
            self.statusBar().showMessage("完成")
        else:
            self.log_output.append("-" * 80)
            self.log_output.append(f"✗ {message}")
            self.statusBar().showMessage("失败")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "任务正在执行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置应用图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
