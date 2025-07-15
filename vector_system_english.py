#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English Version Vector Storage and Retrieval System
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config_english import (
    FOODON_DATA_PATH, CHEBI_DATA_PATH, SEMANTIC_MODEL_NAME, TOP_K_ENTITIES
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

def create_vector_database_english():
    """
    Create English vector database
    """
    print("Initializing English vector database...")
    
    # Load data
    print("Loading foodon data...")
    foods = load_foodon_data_english()
    
    print("Loading chebi data...")
    compounds = load_chebi_data_english()
    
    # Combine all entities
    all_entities = foods + compounds
    
    # Extract texts
    texts = [entity['name'] for entity in all_entities]
    
    print(f"Vectorizing {len(texts)} entries...")
    
    # Create vectors using English-optimized model
    model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    embeddings = model.encode(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    print("English vector database created successfully!")
    
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
        db = create_vector_database_english()
        
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

