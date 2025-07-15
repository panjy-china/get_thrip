#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English Version Configuration File
"""

import os

# File paths
FOODON_DATA_PATH = 'data/foodon_data_english.txt'
CHEBI_DATA_PATH = 'data/chebi_data_english.txt'
TEST_TEXT_PATH = 'data/test_text_english.txt'
RESULT_OUTPUT_PATH = 'output/english_result.json'
INDEX_PATH = "db/aiss_index.index"
ENTITIES_PATH = "db/entities.pkl"
METADATA_PATH = "db/metadata.json"

# Model settings
SIMILARITY_THRESHOLD = 0.5
TOP_K_ENTITIES = 20
SEMANTIC_MODEL_NAME = 'model/all-MiniLM-L6-v2'  # Better for English

# English food indicators
ENGLISH_FOOD_INDICATORS = [
    'apple', 'banana', 'orange', 'milk', 'egg', 'rice', 'wheat', 'corn', 
    'potato', 'carrot', 'tomato', 'spinach', 'beef', 'pork', 'chicken', 
    'fish', 'tofu', 'peanut', 'walnut', 'sesame', 'bread', 'cheese', 
    'meat', 'vegetable', 'fruit', 'grain', 'bean', 'nut'
]

# English compound indicators
ENGLISH_COMPOUND_INDICATORS = [
    'vitamin', 'acid', 'sugar', 'protein', 'fat', 'alcohol', 'amine', 
    'caffeine', 'starch', 'fiber', 'glucose', 'fructose', 'sucrose', 
    'lactose', 'amino', 'cholesterol', 'choline', 'cellulose'
]

# English relation patterns
ENGLISH_RELATION_PATTERNS = [
    r'contain[s]?', r'include[s]?', r'have', r'has', r'with',
    r'rich in', r'abundant in', r'high in', r'full of', r'loaded with',
    r'source of', r'provide[s]?', r'supply', r'supplies',
    r'made of', r'composed of', r'consist[s]? of', r'comprise[s]?',
    r'main component', r'primary component', r'major component',
    r'present in', r'found in', r'exist[s]? in', r'occur[s]? in'
]

def validate_config():
    """Validate configuration"""
    print("Validating configuration...")
    try:
        if not os.path.exists(FOODON_DATA_PATH):
            raise FileNotFoundError(f"Food data file not found: {FOODON_DATA_PATH}")
        if not os.path.exists(CHEBI_DATA_PATH):
            raise FileNotFoundError(f"Compound data file not found: {CHEBI_DATA_PATH}")
        if not os.path.exists(TEST_TEXT_PATH):
            raise FileNotFoundError(f"Test text file not found: {TEST_TEXT_PATH}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(RESULT_OUTPUT_PATH), exist_ok=True)
        
        print("✅ Configuration validation passed")
        return True
    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    validate_config()

