#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English Version Semantic Recognition Triplet Extraction Workflow
"""

import numpy as np
import json
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vector_system_english import search_relevant_entities_english, create_or_load_vector_database
from config_english import (
    TEST_TEXT_PATH, RESULT_OUTPUT_PATH, SIMILARITY_THRESHOLD, 
    TOP_K_ENTITIES, SEMANTIC_MODEL_NAME, ENGLISH_FOOD_INDICATORS,
    ENGLISH_COMPOUND_INDICATORS, ENGLISH_RELATION_PATTERNS, validate_config
)

class EnglishSemanticExtractor:
    """
    English Semantic Extractor for triplet extraction
    
    Features:
    1. Entity extraction from English text
    2. Semantic matching for synonyms (e.g., potato -> potatoes)
    3. Relation recognition unified as "contains"
    4. Confidence calculation
    """
    
    def __init__(self):
        """
        Initialize English semantic extractor
        """
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        self.relation_patterns = ENGLISH_RELATION_PATTERNS
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading NLTK POS tagger...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def extract_sentences(self, text):
        """
        Split text into sentences using NLTK
        """
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_entities_from_sentence(self, sentence):
        """
        Extract potential food and compound entities from English sentence
        
        Strategy:
        1. Use NLTK for tokenization and POS tagging
        2. Identify entities based on keywords and POS tags
        3. Handle common food and compound terms
        """
        # Tokenize and get POS tags
        tokens = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        potential_foods = []
        potential_compounds = []
        
        # Extract nouns and compound words
        for i, (word, pos) in enumerate(pos_tags):
            # Skip very short words
            if len(word) < 3:
                continue
            
            # Check for food indicators
            if (any(indicator in word for indicator in ENGLISH_FOOD_INDICATORS) or
                word in ENGLISH_FOOD_INDICATORS or
                pos in ['NN', 'NNS', 'NNP', 'NNPS']):  # Nouns
                
                # Check if it's likely a food
                if (any(indicator in word for indicator in ENGLISH_FOOD_INDICATORS) or
                    word.endswith('s') and word[:-1] in ENGLISH_FOOD_INDICATORS):  # Plural forms
                    potential_foods.append(word)
            
            # Check for compound indicators
            if (any(indicator in word for indicator in ENGLISH_COMPOUND_INDICATORS) or
                word in ENGLISH_COMPOUND_INDICATORS):
                potential_compounds.append(word)
        
        # Handle compound words (e.g., "vitamin c", "amino acid")
        for i in range(len(tokens) - 1):
            compound_word = f"{tokens[i]} {tokens[i+1]}"
            if any(indicator in compound_word for indicator in ENGLISH_COMPOUND_INDICATORS):
                potential_compounds.append(compound_word)
        
        # Remove duplicates and filter
        potential_foods = list(set([f for f in potential_foods if len(f) >= 3]))
        potential_compounds = list(set([c for c in potential_compounds if len(c) >= 3]))
        
        return potential_foods, potential_compounds
    
    def find_semantic_match(self, text_entity, database_entities):
        """
        Find best semantic match for text entity
        
        This enables automatic synonym recognition
        e.g., "potatoes" -> "potato"
        """
        if not database_entities:
            return None
        
        # Get database entity names
        db_names = [entity['data']['name'] for entity in database_entities]
        
        try:
            # Calculate semantic vectors
            text_embedding = self.semantic_model.encode([text_entity.lower()])
            db_embeddings = self.semantic_model.encode(db_names)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(text_embedding, db_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self.similarity_threshold:
                matched_entity = database_entities[best_idx]['data'].copy()
                matched_entity['original_name'] = text_entity
                matched_entity['similarity_score'] = float(best_similarity)
                return matched_entity
        except Exception as e:
            print(f"  Semantic matching error: {e}")
        
        return None
    
    def extract_triplets_from_sentence(self, sentence, relevant_foods, relevant_compounds):
        """
        Extract triplets from a single sentence
        """
        triplets = []
        
        print(f"  Analyzing sentence: {sentence}")
        
        # Step 1: Extract entities
        potential_foods, potential_compounds = self.extract_entities_from_sentence(sentence)
        
        print(f"  Extracted potential foods: {potential_foods}")
        print(f"  Extracted potential compounds: {potential_compounds}")
        
        # Step 2: Semantic matching
        matched_foods = []
        matched_compounds = []
        
        for food in potential_foods:
            match = self.find_semantic_match(food, relevant_foods)
            if match:
                matched_foods.append(match)
                print(f"  Food match: {food} → {match['name']} (similarity: {match['similarity_score']:.3f})")
        
        for compound in potential_compounds:
            match = self.find_semantic_match(compound, relevant_compounds)
            if match:
                matched_compounds.append(match)
                print(f"  Compound match: {compound} → {match['name']} (similarity: {match['similarity_score']:.3f})")
        
        # Step 3: Extract relations and build triplets
        if matched_foods and matched_compounds:
            for food in matched_foods:
                for compound in matched_compounds:
                    food_name = food.get('original_name', food['name'])
                    compound_name = compound.get('original_name', compound['name'])
                    
                    # Check for relation
                    has_relation = self.has_relation(sentence, food_name, compound_name)
                    if has_relation:
                        confidence = self.calculate_confidence(
                            sentence, food_name, compound_name,
                            food.get('similarity_score', 1.0), 
                            compound.get('similarity_score', 1.0)
                        )
                        
                        triplet = {
                            'subject': food['name'],
                            'subject_id': food['id'],
                            'relation': 'contains',  # Unified relation
                            'object': compound['name'],
                            'object_id': compound['id'],
                            'confidence': confidence,
                            'source_sentence': sentence
                        }
                        
                        if 'original_name' in food:
                            triplet['subject_original'] = food['original_name']
                            triplet['subject_similarity'] = food['similarity_score']
                        if 'original_name' in compound:
                            triplet['object_original'] = compound['original_name']
                            triplet['object_similarity'] = compound['similarity_score']
                        
                        triplets.append(triplet)
        
        return triplets
    
    def has_relation(self, sentence, food, compound):
        """
        Check if relation pattern exists in sentence
        """
        sentence_lower = sentence.lower()
        food_pos = sentence_lower.find(food.lower())
        compound_pos = sentence_lower.find(compound.lower())
        
        if food_pos == -1 or compound_pos == -1:
            return False
        
        # Ensure food comes before compound
        if food_pos > compound_pos:
            food_pos, compound_pos = compound_pos, food_pos
            food, compound = compound, food
        
        # Extract text between entities
        between_text = sentence_lower[food_pos + len(food):compound_pos]
        
        # Check for relation patterns
        for pattern in self.relation_patterns:
            if re.search(pattern, between_text):
                print(f"    Found relation pattern: '{pattern}' in '{between_text.strip()}'")
                return True
        
        # Distance-based implicit relation
        if len(between_text.strip()) < 20:
            print(f"    Close distance, assuming implicit contains relation")
            return True
        
        return False
    
    def calculate_confidence(self, sentence, food, compound, food_sim=1.0, compound_sim=1.0):
        """
        Calculate confidence score for triplet
        """
        confidence = 0.6  # Base confidence
        
        sentence_lower = sentence.lower()
        food_pos = sentence_lower.find(food.lower())
        compound_pos = sentence_lower.find(compound.lower())
        
        if food_pos != -1 and compound_pos != -1:
            between_text = sentence_lower[min(food_pos, compound_pos):max(food_pos, compound_pos)]
            
            # Explicit relation words bonus
            if any(word in between_text for word in ['contain', 'contains', 'rich in', 'high in', 'source of']):
                confidence += 0.2
            
            # Distance bonus
            distance = abs(compound_pos - food_pos)
            if distance < 15:
                confidence += 0.1
            elif distance < 30:
                confidence += 0.05
        
        # Semantic similarity bonus
        avg_similarity = (food_sim + compound_sim) / 2
        confidence += avg_similarity * 0.1
        
        return min(1.0, confidence)

def run_english_workflow():
    """
    Run English semantic recognition workflow
    """
    print("=== English Semantic Recognition Triplet Extraction Workflow ===\n")
    print("Features:")
    print("- Automatic semantic recognition (potatoes → potato)")
    print("- Unified relations as 'contains'")
    print("- English natural language processing\n")
    
    # Step 1: Validate configuration
    print("1. Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed, please check config_english.py")
        return None
    print("✅ Configuration validation passed\n")
    
    # Step 2: Create vector database
    print("2. Initializing vector database...")
    db = create_or_load_vector_database()
    print("✅ Vector database created successfully\n")
    
    # Step 3: Read test text
    print("3. Reading test text...")
    try:
        with open(TEST_TEXT_PATH, 'r', encoding='utf-8') as f:
            test_text = f.read()
        print(f"✅ Successfully read text: {test_text}\n")
    except Exception as e:
        print(f"❌ Failed to read text: {e}")
        return None
    
    # Step 4: Split sentences
    print("4. Splitting sentences...")
    extractor = EnglishSemanticExtractor()
    sentences = extractor.extract_sentences(test_text)
    print(f"✅ Split into {len(sentences)} sentences\n")
    
    # Step 5: Process each sentence
    print("5. Processing sentences...")
    all_triplets = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"Processing sentence {i}/{len(sentences)}: {sentence}")
        
        # Vector search for relevant entities
        foods, compounds = search_relevant_entities_english(db, sentence, top_k=TOP_K_ENTITIES)
        
        if foods or compounds:
            print(f"  ✅ Found {len(foods)} relevant foods, {len(compounds)} relevant compounds")
            
            # Extract triplets
            triplets = extractor.extract_triplets_from_sentence(sentence, foods, compounds)
            
            if triplets:
                print(f"  🎉 Extracted {len(triplets)} triplets")
                all_triplets.extend(triplets)
            else:
                print("  ⚠️  No triplets extracted")
        else:
            print("  ⚠️  No relevant entities found")
        print()
    
    # Step 6: Output results
    print("=" * 60)
    print("Final Extraction Results")
    print("=" * 60)
    
    if all_triplets:
        print(f"🎉 Total extracted {len(all_triplets)} triplets:\n")
        
        for i, triplet in enumerate(all_triplets, 1):
            print(f"Triplet {i}:")
            
            # Display subject (food)
            subject_display = triplet['subject']
            if 'subject_original' in triplet:
                subject_display = f"{triplet['subject_original']} → {triplet['subject']} (similarity: {triplet['subject_similarity']:.3f})"
            
            # Display object (compound)
            object_display = triplet['object']
            if 'object_original' in triplet:
                object_display = f"{triplet['object_original']} → {triplet['object']} (similarity: {triplet['object_similarity']:.3f})"
            
            print(f"  Subject: {subject_display}")
            print(f"  Relation: {triplet['relation']}")
            print(f"  Object: {object_display}")
            print(f"  Confidence: {triplet['confidence']:.3f}")
            print(f"  Source: {triplet['source_sentence']}")
            print()
    else:
        print("❌ No triplets extracted")
    
    # Step 7: Save results
    print("6. Saving results...")
    result = {"triplets": all_triplets}
    try:
        with open(RESULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {RESULT_OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    return result

if __name__ == "__main__":
    # Install NLTK if needed
    try:
        import nltk
    except ImportError:
        print("Installing NLTK...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'nltk'])
        import nltk

    # nltk.download('punkt',
    #               download_dir=r'E:\anaconda\envs\faiss_env\nltk_data',  # 指定环境路径
    #               quiet=False,
    #               halt_on_error=False,
    #               server_index="https://mirrors.aliyun.com/nltk/")  # 阿里云镜像
    # Run English workflow
    try:
        result = run_english_workflow()
        
        if result and result['triplets']:
            print(f"\n🎉 Task completed! Successfully extracted {len(result['triplets'])} triplets")
        else:
            print("\n⚠️  Task completed, but no triplets extracted")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted the program")
    except Exception as e:
        print(f"\n❌ Program execution error: {e}")
        print("Please check configuration files and data files")

