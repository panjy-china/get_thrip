#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可配置的统一关系语义识别三元组提取工作流

使用 config.py 中的配置，便于修改路径和参数
"""

import jieba
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vector_system_configurable import create_vector_database, search_relevant_entities
from config import (
    TEST_TEXT_PATH, RESULT_OUTPUT_PATH, SIMILARITY_THRESHOLD, 
    TOP_K_ENTITIES, SEMANTIC_MODEL_NAME, validate_config
)
# from huggingface_hub import configure_hf
# configure_hf(mirror="https://hf-mirror.com")  # 清华大学镜像


class ConfigurableSemanticExtractor:
    def __init__(self):
        """
        初始化可配置的语义提取器
        """
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        
        # 扩展的关系模式 - 所有这些都会被识别为"包含"关系
        self.relation_patterns = [
            r'含有', r'富含', r'包含', r'含', r'有',
            r'是.*来源', r'来源', r'源于',
            r'提供', r'供应', r'给予',
            r'含.*量.*的', r'含量',
            r'主要成分.*是', r'主要成分', r'成分.*是', r'成分',
            r'组成.*是', r'组成', r'构成',
            r'含.*丰富', r'丰富.*含',
            r'存在.*中', r'存在于',
            r'具有', r'拥有', r'带有',
            r'蕴含', r'内含', r'含.*大量',
            r'营养.*含', r'营养成分'
        ]
    
    def extract_sentences(self, text):
        """分割句子"""
        sentences = re.split(r'[。！？]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_entities_from_sentence(self, sentence):
        """
        实体提取：使用jieba分词
        """
        
        words = jieba.lcut(sentence)
        potential_foods = []
        potential_compounds = []
        
        # 预定义的食物和化合物关键词
        food_indicators = ['薯', '豆', '菜', '果', '肉', '鱼', '蛋', '奶', '米', '麦', '玉米', '萝卜', '茄']
        compound_indicators = ['维生素', '维', '酸', '糖', '蛋白', '脂', '素', '醇', '胺', '因', '淀粉']
        
        for word in words:
            if len(word) >= 2:
                # 检查食物
                if (any(indicator in word for indicator in food_indicators) or 
                    word in ['马铃薯', '番茄', '红萝卜', '玉蜀黍']):
                    potential_foods.append(word)
                
                # 检查化合物
                if (any(indicator in word for indicator in compound_indicators) or
                    word in ['VC', 'VA', 'VE', 'VD', 'VK']):
                    potential_compounds.append(word)
        
        return list(set(potential_foods)), list(set(potential_compounds))
    
    def find_semantic_match(self, text_entity, database_entities):
        """
        为文本实体找到最佳的语义匹配
        """
        if not database_entities:
            return None
        
        # 获取数据库实体名称
        db_names = [entity['data']['name'] for entity in database_entities]
        
        # 计算语义向量
        try:
            text_embedding = self.semantic_model.encode([text_entity])
            db_embeddings = self.semantic_model.encode(db_names)
            
            # 使用cosine_similarity计算相似度
            similarities = cosine_similarity(text_embedding, db_embeddings)[0]
            
            # 找到最佳匹配
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self.similarity_threshold:
                matched_entity = database_entities[best_idx]['data'].copy()
                matched_entity['original_name'] = text_entity
                matched_entity['similarity_score'] = float(best_similarity)
                return matched_entity
        except Exception as e:
            print(f"  语义匹配出错: {e}")
        
        return None
    
    def extract_triplets_from_sentence(self, sentence, relevant_foods, relevant_compounds):
        """
        从句子中提取三元组
        """
        triplets = []
        
        print(f"  分析句子: {sentence}")
        
        # 1. 提取实体
        potential_foods, potential_compounds = self.extract_entities_from_sentence(sentence)
        
        print(f"  提取的潜在食物: {potential_foods}")
        print(f"  提取的潜在化合物: {potential_compounds}")
        
        # 2. 语义匹配
        matched_foods = []
        matched_compounds = []
        
        for food in potential_foods:
            match = self.find_semantic_match(food, relevant_foods)
            if match:
                matched_foods.append(match)
                print(f"  食物匹配: {food} → {match['name']} (相似度: {match['similarity_score']:.3f})")
        
        for compound in potential_compounds:
            match = self.find_semantic_match(compound, relevant_compounds)
            if match:
                matched_compounds.append(match)
                print(f"  化合物匹配: {compound} → {match['name']} (相似度: {match['similarity_score']:.3f})")
        
        # 3. 提取关系 - 统一为"包含"
        if matched_foods and matched_compounds:
            for food in matched_foods:
                for compound in matched_compounds:
                    food_name = food.get('original_name', food['name'])
                    compound_name = compound.get('original_name', compound['name'])
                    
                    # 检查是否存在任何关系模式
                    has_relation = self.has_relation(sentence, food_name, compound_name)
                    if has_relation:
                        confidence = self.calculate_confidence(
                            sentence, food_name, compound_name,
                            food.get('similarity_score', 1.0), compound.get('similarity_score', 1.0)
                        )
                        
                        triplet = {
                            'subject': food['name'],
                            'subject_id': food['id'],
                            'relation': '包含',  # 统一为"包含"关系
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
        检查句子中是否存在任何关系模式
        """
        food_pos = sentence.find(food)
        compound_pos = sentence.find(compound)
        
        if food_pos == -1 or compound_pos == -1:
            return False
        
        # 允许任意顺序，但优先食物在前
        if food_pos > compound_pos:
            food_pos, compound_pos = compound_pos, food_pos
            food, compound = compound, food
        
        # 提取中间文本
        between_text = sentence[food_pos + len(food):compound_pos]
        
        # 检查是否存在任何关系模式
        for pattern in self.relation_patterns:
            if re.search(pattern, between_text):
                print(f"    发现关系模式: '{pattern}' 在 '{between_text.strip()}' 中")
                return True
        
        # 如果距离很近，也认为存在隐含关系
        if len(between_text.strip()) < 15:
            print(f"    距离较近，认为存在隐含包含关系")
            return True
        
        return False
    
    def calculate_confidence(self, sentence, food, compound, food_sim=1.0, compound_sim=1.0):
        """计算置信度"""
        confidence = 0.6
        
        # 检查关系词的明确性
        food_pos = sentence.find(food)
        compound_pos = sentence.find(compound)
        
        if food_pos != -1 and compound_pos != -1:
            between_text = sentence[min(food_pos, compound_pos):max(food_pos, compound_pos)]
            
            # 明确的关系词加分
            if any(word in between_text for word in ['含有', '富含', '包含', '主要成分', '成分']):
                confidence += 0.2
            
            # 距离加分
            distance = abs(compound_pos - food_pos)
            if distance < 10:
                confidence += 0.1
            elif distance < 20:
                confidence += 0.05
        
        # 语义相似度加分
        avg_similarity = (food_sim + compound_sim) / 2
        confidence += avg_similarity * 0.1
        
        return min(1.0, confidence)

def run_main_workflow():
    """
    运行主工作流
    """
    print("=== 可配置的语义识别三元组提取工作流 ===\n")
    print("所有关系统一为'包含'关系\n")
    
    # 验证配置
    if not validate_config():
        print("请先修改 config.py 中的配置")
        return None
    
    # 1. 创建向量数据库
    print("1. 初始化向量数据库...")
    db = create_vector_database()
    
    # 2. 读取测试文本
    print("\n2. 读取测试文本...")
    with open(TEST_TEXT_PATH, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    print(f"测试文本: {test_text}\n")
    
    # 3. 分割句子
    print("3. 分割句子...")
    extractor = ConfigurableSemanticExtractor()
    sentences = extractor.extract_sentences(test_text)
    print(f"分割得到 {len(sentences)} 个句子\n")
    
    # 4. 处理每个句子
    all_triplets = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"处理句子 {i}: {sentence}")
        
        # 向量检索相关实体
        foods, compounds = search_relevant_entities(db, sentence, top_k=TOP_K_ENTITIES)
        
        if foods or compounds:
            print(f"  检索到 {len(foods)} 个相关食物，{len(compounds)} 个相关化合物")
            
            triplets = extractor.extract_triplets_from_sentence(sentence, foods, compounds)
            
            if triplets:
                print(f"  ✓ 提取到 {len(triplets)} 个三元组")
                all_triplets.extend(triplets)
            else:
                print("  未提取到三元组")
        else:
            print("  未找到相关实体")
        print()
    
    # 5. 输出结果
    print("=== 最终提取结果 ===")
    if all_triplets:
        print(f"总共提取 {len(all_triplets)} 个三元组：\n")
        for i, triplet in enumerate(all_triplets, 1):
            print(f"三元组 {i}:")
            
            subject_display = triplet['subject']
            if 'subject_original' in triplet:
                subject_display = f"{triplet['subject_original']} → {triplet['subject']} (相似度: {triplet['subject_similarity']:.3f})"
            
            object_display = triplet['object']
            if 'object_original' in triplet:
                object_display = f"{triplet['object_original']} → {triplet['object']} (相似度: {triplet['object_similarity']:.3f})"
            
            print(f"  主体: {subject_display}")
            print(f"  关系: {triplet['relation']}")
            print(f"  客体: {object_display}")
            print(f"  置信度: {triplet['confidence']:.3f}")
            print(f"  来源: {triplet['source_sentence']}")
            print()
    else:
        print("未提取到任何三元组")
    
    # 6. 保存结果
    result = {"triplets": all_triplets}
    with open(RESULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {RESULT_OUTPUT_PATH}")
    
    return result

if __name__ == "__main__":
    # # 安装jieba（如果需要）
    # try:
    #     import jieba
    # except ImportError:
    #     print("正在安装jieba分词库...")
    #     import subprocess
    #     subprocess.check_call(['pip', 'install', 'jieba'])
    #     import jieba
    
    result = run_main_workflow()

