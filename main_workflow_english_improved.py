#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved English Version - Fixes Duplicate Triplets Issue
改进版英文工作流 - 解决重复三元组问题
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

class ImprovedEnglishSemanticExtractor:
    """
    Improved English Semantic Extractor
    解决重复三元组问题的改进版本
    """
    
    def __init__(self):
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        self.relation_patterns = ENGLISH_RELATION_PATTERNS
        
        # 停用词列表 - 过滤无意义的词
        self.stop_words = {
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'can', 'large', 'amounts', 'amount'
        }
        
        # 下载NLTK数据
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """确保NLTK数据已下载"""
        try:
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    def extract_sentences(self, text):
        """分割句子"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_entities_from_sentence(self, sentence):
        """
        改进的实体提取 - 避免重复和无效实体
        """
        tokens = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        potential_foods = []
        potential_compounds = []
        
        # 步骤1: 提取单词实体
        for word, pos in pos_tags:
            # 过滤停用词和短词
            if word in self.stop_words or len(word) < 3:
                continue
            
            # 检查是否为食物
            if (self._is_food_entity(word, pos) and 
                not self._contains_stop_words(word)):
                potential_foods.append(word)
            
            # 检查是否为化合物
            elif (self._is_compound_entity(word, pos) and 
                  not self._contains_stop_words(word)):
                potential_compounds.append(word)
        
        # 步骤2: 检查相邻词组合（但避免包含停用词）
        for i in range(len(tokens) - 1):
            if (tokens[i] not in self.stop_words and 
                tokens[i+1] not in self.stop_words and
                len(tokens[i]) >= 3 and len(tokens[i+1]) >= 3):
                
                compound_word = f"{tokens[i]} {tokens[i+1]}"
                
                if self._is_compound_entity(compound_word, 'NN'):
                    potential_compounds.append(compound_word)
        
        # 步骤3: 去重并过滤重叠实体
        potential_foods = self._remove_overlapping_entities(potential_foods)
        potential_compounds = self._remove_overlapping_entities(potential_compounds)
        
        return potential_foods, potential_compounds
    
    def _is_food_entity(self, word, pos):
        """判断是否为食物实体"""
        return (any(indicator in word for indicator in ENGLISH_FOOD_INDICATORS) or
                word in ENGLISH_FOOD_INDICATORS or
                (pos in ['NN', 'NNS', 'NNP', 'NNPS'] and 
                 any(indicator in word for indicator in ENGLISH_FOOD_INDICATORS)))
    
    def _is_compound_entity(self, word, pos):
        """判断是否为化合物实体"""
        return (any(indicator in word for indicator in ENGLISH_COMPOUND_INDICATORS) or
                word in ENGLISH_COMPOUND_INDICATORS)
    
    def _contains_stop_words(self, entity):
        """检查实体是否包含停用词"""
        words = entity.split()
        return any(word in self.stop_words for word in words)
    
    def _remove_overlapping_entities(self, entities):
        """
        移除重叠的实体
        优先保留较短、更精确的实体
        """
        if not entities:
            return []
        
        # 去重
        entities = list(set(entities))
        
        # 按长度排序，优先保留较短的实体
        entities = sorted(entities, key=len)
        
        filtered = []
        for entity in entities:
            # 检查是否与已有实体重叠
            is_overlapping = False
            for existing in filtered:
                # 如果当前实体包含已有实体，或被已有实体包含，则认为重叠
                if (entity in existing or existing in entity or
                    self._has_word_overlap(entity, existing)):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(entity)
        
        return filtered
    
    def _has_word_overlap(self, entity1, entity2):
        """检查两个实体是否有词汇重叠"""
        words1 = set(entity1.split())
        words2 = set(entity2.split())
        
        # 如果有超过50%的词汇重叠，认为是重叠实体
        overlap = words1.intersection(words2)
        min_words = min(len(words1), len(words2))
        
        return len(overlap) / min_words > 0.5 if min_words > 0 else False
    
    def find_semantic_match(self, text_entity, database_entities):
        """语义匹配 - 添加额外的质量检查"""
        if not database_entities:
            return None
        
        # 预过滤：如果实体包含太多停用词，降低优先级
        if self._contains_stop_words(text_entity):
            # 提高相似度阈值
            threshold = self.similarity_threshold + 0.1
        else:
            threshold = self.similarity_threshold
        
        db_names = [entity['data']['name'] for entity in database_entities]
        
        try:
            text_embedding = self.semantic_model.encode([text_entity.lower()])
            db_embeddings = self.semantic_model.encode(db_names)
            
            similarities = cosine_similarity(text_embedding, db_embeddings)[0]
            
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= threshold:
                matched_entity = database_entities[best_idx]['data'].copy()
                matched_entity['original_name'] = text_entity
                matched_entity['similarity_score'] = float(best_similarity)
                return matched_entity
        except Exception as e:
            print(f"  Semantic matching error: {e}")
        
        return None
    
    def extract_triplets_from_sentence(self, sentence, relevant_foods, relevant_compounds):
        """提取三元组 - 添加去重逻辑"""
        triplets = []
        
        print(f"  Analyzing sentence: {sentence}")
        
        # 实体提取
        potential_foods, potential_compounds = self.extract_entities_from_sentence(sentence)
        
        print(f"  Extracted potential foods: {potential_foods}")
        print(f"  Extracted potential compounds: {potential_compounds}")
        
        # 语义匹配
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
        
        # 构建三元组
        if matched_foods and matched_compounds:
            for food in matched_foods:
                for compound in matched_compounds:
                    food_name = food.get('original_name', food['name'])
                    compound_name = compound.get('original_name', compound['name'])
                    
                    if self.has_relation(sentence, food_name, compound_name):
                        confidence = self.calculate_confidence(
                            sentence, food_name, compound_name,
                            food.get('similarity_score', 1.0), 
                            compound.get('similarity_score', 1.0)
                        )
                        
                        triplet = {
                            'subject': food['name'],
                            'subject_id': food['id'],
                            'relation': 'contains',
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
        
        # 去重处理
        triplets = self.deduplicate_triplets(triplets)
        
        return triplets
    
    def deduplicate_triplets(self, triplets):
        """
        去除重复的三元组
        基于主体ID、关系、客体ID的唯一性
        """
        if not triplets:
            return []
        
        unique_triplets = []
        seen_combinations = {}
        
        for triplet in triplets:
            # 创建唯一标识符
            key = (
                triplet['subject_id'],
                triplet['relation'], 
                triplet['object_id'],
                triplet['source_sentence']
            )
            
            if key not in seen_combinations:
                seen_combinations[key] = triplet
                unique_triplets.append(triplet)
            else:
                # 如果重复，保留质量更高的
                existing = seen_combinations[key]
                
                # 质量评分：置信度 + 原始实体质量
                current_quality = self._calculate_triplet_quality(triplet)
                existing_quality = self._calculate_triplet_quality(existing)
                
                if current_quality > existing_quality:
                    # 替换为更高质量的三元组
                    seen_combinations[key] = triplet
                    for i, t in enumerate(unique_triplets):
                        if (t['subject_id'] == existing['subject_id'] and
                            t['object_id'] == existing['object_id'] and
                            t['source_sentence'] == existing['source_sentence']):
                            unique_triplets[i] = triplet
                            break
        
        return unique_triplets
    
    def _calculate_triplet_quality(self, triplet):
        """
        计算三元组质量分数
        考虑置信度、相似度、原始实体质量
        """
        quality = triplet['confidence']  # 基础置信度
        
        # 语义相似度加分
        if 'subject_similarity' in triplet:
            quality += triplet['subject_similarity'] * 0.1
        if 'object_similarity' in triplet:
            quality += triplet['object_similarity'] * 0.1
        
        # 原始实体质量加分
        if 'subject_original' in triplet:
            # 更短、不含停用词的实体质量更高
            subject_orig = triplet['subject_original']
            if not self._contains_stop_words(subject_orig):
                quality += 0.1
            if len(subject_orig.split()) == 1:  # 单词实体更准确
                quality += 0.05
        
        if 'object_original' in triplet:
            object_orig = triplet['object_original']
            if not self._contains_stop_words(object_orig):
                quality += 0.1
            if len(object_orig.split()) == 1:
                quality += 0.05
        
        return quality
    
    def has_relation(self, sentence, food, compound):
        """关系识别"""
        sentence_lower = sentence.lower()
        food_pos = sentence_lower.find(food.lower())
        compound_pos = sentence_lower.find(compound.lower())
        
        if food_pos == -1 or compound_pos == -1:
            return False
        
        if food_pos > compound_pos:
            food_pos, compound_pos = compound_pos, food_pos
            food, compound = compound, food
        
        between_text = sentence_lower[food_pos + len(food):compound_pos]
        
        for pattern in self.relation_patterns:
            if re.search(pattern, between_text):
                print(f"    Found relation pattern: '{pattern}' in '{between_text.strip()}'")
                return True
        
        if len(between_text.strip()) < 20:
            print(f"    Close distance, assuming implicit contains relation")
            return True
        
        return False
    
    def calculate_confidence(self, sentence, food, compound, food_sim=1.0, compound_sim=1.0):
        """置信度计算"""
        confidence = 0.6
        
        sentence_lower = sentence.lower()
        food_pos = sentence_lower.find(food.lower())
        compound_pos = sentence_lower.find(compound.lower())
        
        if food_pos != -1 and compound_pos != -1:
            between_text = sentence_lower[min(food_pos, compound_pos):max(food_pos, compound_pos)]
            
            if any(word in between_text for word in ['contain', 'contains', 'rich in', 'high in', 'source of']):
                confidence += 0.2
            
            distance = abs(compound_pos - food_pos)
            if distance < 15:
                confidence += 0.1
            elif distance < 30:
                confidence += 0.05
        
        avg_similarity = (food_sim + compound_sim) / 2
        confidence += avg_similarity * 0.1
        
        return min(1.0, confidence)

def run_improved_english_workflow():
    """运行改进版英文工作流"""
    print("=== Improved English Semantic Recognition (No Duplicates) ===\n")
    print("Improvements:")
    print("- Enhanced entity extraction (filters stop words)")
    print("- Overlap detection and removal")
    print("- Triplet deduplication")
    print("- Quality-based selection\n")
    
    if not validate_config():
        print("❌ Configuration validation failed")
        return None
    
    print("1. Initializing vector database...")
    db = create_or_load_vector_database()
    
    print("\n2. Reading test text...")
    with open(TEST_TEXT_PATH, 'r', encoding='utf-8') as f:
        test_text = f.read()
    print(f"✅ Text: {test_text}\n")
    
    print("3. Processing with improved extractor...")
    extractor = ImprovedEnglishSemanticExtractor()
    sentences = extractor.extract_sentences(test_text)
    
    all_triplets = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"Processing sentence {i}/{len(sentences)}: {sentence}")
        
        foods, compounds = search_relevant_entities_english(db, sentence, top_k=TOP_K_ENTITIES)
        
        if foods or compounds:
            print(f"  ✅ Found {len(foods)} foods, {len(compounds)} compounds")
            triplets = extractor.extract_triplets_from_sentence(sentence, foods, compounds)
            
            if triplets:
                print(f"  🎉 Extracted {len(triplets)} unique triplets")
                all_triplets.extend(triplets)
            else:
                print("  ⚠️  No triplets extracted")
        else:
            print("  ⚠️  No relevant entities found")
        print()
    
    # 全局去重（跨句子）
    print("4. Performing global deduplication...")
    all_triplets = extractor.deduplicate_triplets(all_triplets)
    
    print("=" * 60)
    print("Final Results (Deduplicated)")
    print("=" * 60)
    
    if all_triplets:
        print(f"🎉 Total extracted {len(all_triplets)} unique triplets:\n")
        
        for i, triplet in enumerate(all_triplets, 1):
            print(f"Triplet {i}:")
            
            subject_display = triplet['subject']
            if 'subject_original' in triplet:
                subject_display = f"{triplet['subject_original']} → {triplet['subject']} (sim: {triplet['subject_similarity']:.3f})"
            
            object_display = triplet['object']
            if 'object_original' in triplet:
                object_display = f"{triplet['object_original']} → {triplet['object']} (sim: {triplet['object_similarity']:.3f})"
            
            print(f"  Subject: {subject_display}")
            print(f"  Relation: {triplet['relation']}")
            print(f"  Object: {object_display}")
            print(f"  Confidence: {triplet['confidence']:.3f}")
            print(f"  Source: {triplet['source_sentence']}")
            print()
    else:
        print("❌ No triplets extracted")
    
    # 保存结果
    result = {"triplets": all_triplets}
    output_path = RESULT_OUTPUT_PATH.replace('.json', '_improved.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to {output_path}")
    
    return result

if __name__ == "__main__":
    try:
        result = run_improved_english_workflow()
        
        if result and result['triplets']:
            print(f"\n🎉 Task completed! Extracted {len(result['triplets'])} unique triplets")
            print("✅ Duplicate triplets have been removed!")
        else:
            print("\n⚠️  Task completed, but no triplets extracted")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check configuration and data files")

