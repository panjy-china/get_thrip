#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可配置的向量存储和检索系统
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import FOODON_DATA_PATH, CHEBI_DATA_PATH, SEMANTIC_MODEL_NAME, TOP_K_ENTITIES

def load_foodon_data():
    """
    加载foodon食物数据
    """
    foods = []
    with open(FOODON_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                food_id, name = line.split('\t', 1)
                foods.append({'id': food_id, 'name': name, 'type': 'food'})
    return foods

def load_chebi_data():
    """
    加载chebi化合物数据
    """
    compounds = []
    with open(CHEBI_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                compound_id, name = line.split('\t', 1)
                compounds.append({'id': compound_id, 'name': name, 'type': 'compound'})
    return compounds

def create_vector_database():
    """
    创建向量数据库
    """
    print("正在初始化向量数据库...")
    
    # 加载数据
    print("正在加载foodon数据...")
    foods = load_foodon_data()
    
    print("正在加载chebi数据...")
    compounds = load_chebi_data()
    
    # 合并所有数据
    all_entities = foods + compounds
    
    # 提取文本
    texts = [entity['name'] for entity in all_entities]
    
    print(f"正在向量化 {len(texts)} 个条目...")
    
    # 创建向量
    model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    embeddings = model.encode(texts)
    
    # 创建FAISS索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
    
    # 标准化向量（用于余弦相似度）
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    print("向量数据库创建完成！")
    
    return {
        'index': index,
        'entities': all_entities,
        'model': model
    }

def search_relevant_entities(db, query_text, top_k=None):
    """
    搜索相关实体
    """
    if top_k is None:
        top_k = TOP_K_ENTITIES
    
    print("正在搜索与文本相关的实体...")
    
    # 向量化查询文本
    query_embedding = db['model'].encode([query_text])
    faiss.normalize_L2(query_embedding)
    
    # 搜索
    scores, indices = db['index'].search(query_embedding.astype('float32'), top_k)
    
    # 分离食物和化合物
    foods = []
    compounds = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(db['entities']):
            entity = db['entities'][idx]
            entity_with_score = {
                'data': entity,
                'score': float(score)
            }
            
            if entity['type'] == 'food':
                foods.append(entity_with_score)
            else:
                compounds.append(entity_with_score)
    
    return foods, compounds

if __name__ == "__main__":
    # 测试向量系统
    from config import validate_config
    
    if validate_config():
        db = create_vector_database()
        
        test_query = "马铃薯含有淀粉"
        foods, compounds = search_relevant_entities(db, test_query, top_k=5)
        
        print(f"\n查询: {test_query}")
        print(f"找到 {len(foods)} 个相关食物:")
        for food in foods:
            print(f"  {food['data']['name']} (相似度: {food['score']:.3f})")
        
        print(f"找到 {len(compounds)} 个相关化合物:")
        for compound in compounds:
            print(f"  {compound['data']['name']} (相似度: {compound['score']:.3f})")
    else:
        print("请先修改 config.py 中的文件路径配置")

