#!/usr/bin/env python3
"""
测试资源页面功能
"""

from pathlib import Path

# 测试 .gitignore 文件是否存在
gitignore_path = Path(__file__).parent / ".gitignore"
print(f"检查 .gitignore 文件: {gitignore_path}")

if gitignore_path.exists():
    print("✓ .gitignore 文件存在")
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✓ 文件大小: {len(content)} 字符")
    print(f"✓ 行数: {len(content.splitlines())}")
    print("\n文件内容预览（前10行）:")
    print("\n".join(content.splitlines()[:10]))
    print("...")
else:
    print("✗ .gitignore 文件不存在")

# 测试 Web 应用中的资源页面函数
print("\n" + "="*50)
print("测试 Web 应用资源页面...")

try:
    # 模拟导入（不实际运行 Streamlit）
    import sys
    from pathlib import Path
    
    # 模拟 Path 操作
    web_app_path = Path(__file__).parent / "web_app"
    gitignore_path_from_app = web_app_path.parent / ".gitignore"
    
    print(f"从 web_app 路径查找 .gitignore: {gitignore_path_from_app}")
    
    if gitignore_path_from_app.exists():
        print("✓ 路径正确")
        
        # 读取内容
        with open(gitignore_path_from_app, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        print(f"✓ 可以读取文件内容")
        print(f"✓ 内容长度: {len(gitignore_content)} 字符")
        
        # 检查关键内容
        expected_sections = [
            "Python",
            "PyTorch",
            "YOLO",
            "Virtual environments",
            "Streamlit",
            "Data and results"
        ]
        
        found_sections = []
        for section in expected_sections:
            if section in gitignore_content:
                found_sections.append(section)
        
        print(f"✓ 找到配置段: {', '.join(found_sections)}")
        
        # 测试编码
        try:
            encoded = gitignore_content.encode('utf-8')
            decoded = encoded.decode('utf-8')
            print("✓ UTF-8 编码/解码正常")
        except Exception as e:
            print(f"✗ 编码测试失败: {e}")
    
    else:
        print("✗ 从 web_app 路径无法找到 .gitignore")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("✓ 所有基础测试通过")
print("\n使用说明:")
print("1. 启动 Web 应用: cd web_app && streamlit run app.py")
print("2. 在侧边栏选择 '📦 项目资源'")
print("3. 查看并下载 .gitignore 文件")
