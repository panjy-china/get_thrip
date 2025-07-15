#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English Version Vector Storage and Retrieval System
"""
import json
import os
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config_english import (
    FOODON_DATA_PATH, CHEBI_DATA_PATH, SEMANTIC_MODEL_NAME, TOP_K_ENTITIES, INDEX_PATH, ENTITIES_PATH, METADATA_PATH
)

def load_foodon_data_english():
    """
    Load English foodon food data
    Format: FOODON:00001001\tapple
    """
    foods = []
    with open(FOODON_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                food_id, name = line.split('\t', 1)
                foods.append({'id': food_id, 'name': name.lower(), 'type': 'food'})
    return foods

def load_chebi_data_english():
    """
    Load English chebi compound data
    Format: CHEBI:15377\twater
    """
    compounds = []
    with open(CHEBI_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                compound_id, name = line.split('\t', 1)
                compounds.append({'id': compound_id, 'name': name.lower(), 'type': 'compound'})
    return compounds

# def create_vector_database_english():
#     """
#     Create English vector database
#     """
#     print("Initializing English vector database...")
#
#     # Load data
#     print("Loading foodon data...")
#     foods = load_foodon_data_english()
#
#     print("Loading chebi data...")
#     compounds = load_chebi_data_english()
#
#     # Combine all entities
#     all_entities = foods + compounds
#
#     # Extract texts
#     texts = [entity['name'] for entity in all_entities]
#
#     print(f"Vectorizing {len(texts)} entries...")
#
#     # Create vectors using English-optimized model
#     model = SentenceTransformer(SEMANTIC_MODEL_NAME)
#     embeddings = model.encode(texts)
#
#     # Create FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)  # Inner product similarity
#
#     # Normalize vectors for cosine similarity
#     faiss.normalize_L2(embeddings)
#     index.add(embeddings.astype('float32'))
#
#     print("English vector database created successfully!")
#
#     return {
#         'index': index,
#         'entities': all_entities,
#         'model': model
#     }

def create_or_load_vector_database():
    """
    创建或加载轻量级向量数据库（仅保存索引和实体）
    """
    # 检查是否已有保存文件
    if all(os.path.exists(p) for p in [INDEX_PATH, ENTITIES_PATH, METADATA_PATH]):
        print("Loading existing database...")
        return load_database()

    print("Creating new database...")
    # 加载数据
    foods = load_foodon_data_english()
    compounds = load_chebi_data_english()
    all_entities = foods + compounds
    texts = [entity['name'] for entity in all_entities]

    # 创建向量
    model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    embeddings = model.encode(texts)

    # 创建索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    # 保存数据
    faiss.write_index(index, INDEX_PATH)
    with open(ENTITIES_PATH, 'wb') as f:
        pickle.dump(all_entities, f)

    # 保存元信息
    metadata = {
        'model_name': SEMANTIC_MODEL_NAME,
        'embedding_dim': dimension,
        'entity_count': len(all_entities)
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

    print("Database created and saved successfully!")
    return {
        'index': index,
        'entities': all_entities,
        'model': model  # 仅内存中保留，不持久化
    }


def load_database():
    """
    加载已保存的数据库（无需模型文件）
    """
    # 加载索引和实体
    index = faiss.read_index(INDEX_PATH)
    with open(ENTITIES_PATH, 'rb') as f:
        all_entities = pickle.load(f)

    # 加载元信息
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # 动态加载模型（需确保环境已安装该模型）
    model = SentenceTransformer(metadata['model_name'])

    # 验证维度一致性
    assert metadata['embedding_dim'] == model.get_sentence_embedding_dimension(), \
        "模型版本不匹配！"

    print("Database loaded successfully!")
    return {
        'index': index,
        'entities': all_entities,
        'model': model
    }


def search_relevant_entities_english(db, query_text, top_k=None):
    """
    Search for relevant entities in English
    """
    if top_k is None:
        top_k = TOP_K_ENTITIES
    
    print("Searching for relevant entities...")
    
    # Vectorize query text
    query_embedding = db['model'].encode([query_text.lower()])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = db['index'].search(query_embedding.astype('float32'), top_k)
    
    # Separate foods and compounds
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
    # Test English vector system
    from config_english import validate_config
    
    if validate_config():
        db = create_or_load_vector_database()
        
        test_query = "potatoes contain starch"
        foods, compounds = search_relevant_entities_english(db, test_query, top_k=5)
        
        print(f"\nQuery: {test_query}")
        print(f"Found {len(foods)} relevant foods:")
        for food in foods:
            print(f"  {food['data']['name']} (similarity: {food['score']:.3f})")
        
        print(f"Found {len(compounds)} relevant compounds:")
        for compound in compounds:
            print(f"  {compound['data']['name']} (similarity: {compound['score']:.3f})")
    else:
        print("Please fix the configuration in config_english.py first")

