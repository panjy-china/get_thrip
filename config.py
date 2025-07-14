#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件 - 修改这里的路径即可
"""

import os

# =============================================================================
# 文件路径配置 - 请根据您的实际情况修改以下路径
# =============================================================================

# 数据文件路径
FOODON_DATA_PATH = 'data/foodon_data.txt'  # 食物数据文件路径
CHEBI_DATA_PATH = 'data/chebi_data.txt'  # 化合物数据文件路径
TEST_TEXT_PATH = 'data/test_text.txt'  # 测试文本文件路径

# 结果输出路径
RESULT_OUTPUT_PATH = 'output/unified_semantic_result.json'  # 结果保存路径

# =============================================================================
# 模型配置
# =============================================================================

# 语义相似度阈值（0.3-0.8，越低越宽松）
SIMILARITY_THRESHOLD = 0.5

# 向量检索数量（10-50，越大越全面但越慢）
TOP_K_ENTITIES = 20

# 语义模型名称
SEMANTIC_MODEL_NAME = 'model/paraphrase-multilingual-MiniLM-L12-v2'

# =============================================================================
# 辅助函数
# =============================================================================

def get_absolute_path(relative_path):
    """
    将相对路径转换为绝对路径
    """
    if os.path.isabs(relative_path):
        return relative_path
    else:
        return os.path.abspath(relative_path)

def check_file_exists(file_path):
    """
    检查文件是否存在
    """
    abs_path = get_absolute_path(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"文件不存在: {abs_path}")
    return abs_path

def ensure_dir_exists(file_path):
    """
    确保文件所在目录存在
    """
    abs_path = get_absolute_path(file_path)
    dir_path = os.path.dirname(abs_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return abs_path

# =============================================================================
# 配置验证
# =============================================================================

def validate_config():
    """
    验证配置是否正确
    """
    print("正在验证配置...")
    
    try:
        # 检查数据文件
        check_file_exists(FOODON_DATA_PATH)
        check_file_exists(CHEBI_DATA_PATH)
        check_file_exists(TEST_TEXT_PATH)
        
        # 确保输出目录存在
        ensure_dir_exists(RESULT_OUTPUT_PATH)
        
        print("✅ 配置验证通过")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 配置错误: {e}")
        print("\n请检查以下文件是否存在:")
        print(f"- 食物数据: {FOODON_DATA_PATH}")
        print(f"- 化合物数据: {CHEBI_DATA_PATH}")
        print(f"- 测试文本: {TEST_TEXT_PATH}")
        return False

if __name__ == "__main__":
    validate_config()

