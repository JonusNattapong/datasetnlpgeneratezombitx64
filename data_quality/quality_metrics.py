"""
Quality Metrics
--------------
ระบบวัดคุณภาพชุดข้อมูล NLP
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

class QualityMetrics:
    """เครื่องมือคำนวณเมตริกคุณภาพชุดข้อมูล"""
    
    @staticmethod
    def calculate_class_balance(labels: List[Any]) -> Dict[str, Any]:
        """
        คำนวณความสมดุลของคลาส
        
        Args:
            labels (List[Any]): รายการเลเบล
            
        Returns:
            Dict[str, Any]: เมตริกความสมดุล
        """
        counter = Counter(labels)
        total = len(labels)
        
        # หาความถี่สัมพัทธ์ของแต่ละคลาส
        frequencies = {str(label): count / total for label, count in counter.items()}
        
        # คำนวณ entropy (ค่าสูงหมายถึงความสมดุลสูง)
        entropy = -sum(p * np.log2(p) for p in frequencies.values() if p > 0)
        
        # คำนวณ Gini impurity (ค่าต่ำหมายถึงความไม่สมดุลสูง)
        gini = 1 - sum(p**2 for p in frequencies.values())
        
        # อัตราส่วนความถี่ต่ำสุดต่อความถี่สูงสุด (ค่าสูงหมายถึงสมดุลมากขึ้น)
        min_freq = min(frequencies.values()) if frequencies else 0
        max_freq = max(frequencies.values()) if frequencies else 0
        min_max_ratio = min_freq / max_freq if max_freq else 0
        
        # ค่าความเหมาะสมโดยรวม (1 หมายถึงสมดุลสมบูรณ์)
        num_classes = len(counter)
        ideal_entropy = np.log2(num_classes) if num_classes > 1 else 0
        balance_score = entropy / ideal_entropy if ideal_entropy else 1
        
        return {
            "class_distribution": dict(counter),
            "frequencies": frequencies,
            "entropy": entropy,
            "gini_impurity": gini,
            "min_max_ratio": min_max_ratio,
            "balance_score": balance_score
        }
    
    @staticmethod
    def calculate_text_diversity(texts: List[str]) -> Dict[str, Any]:
        """
        คำนวณความหลากหลายของข้อความ
        
        Args:
            texts (List[str]): รายการข้อความ
            
        Returns:
            Dict[str, Any]: เมตริกความหลากหลาย
        """
        if not texts:
            return {"error": "ไม่มีข้อความในรายการ"}
            
        # คำนวณความยาวข้อความ
        lengths = [len(text) for text in texts]
        
        # นับข้อความที่ไม่ซ้ำกัน
        unique_texts = set(texts)
        
        # คำนวณความหลากหลายของคำ (อย่างง่าย)
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        
        # Type-Token Ratio (TTR) - อัตราส่วนคำที่ไม่ซ้ำกันต่อจำนวนคำทั้งหมด
        ttr = len(unique_words) / len(all_words) if all_words else 0
        
        return {
            "text_count": len(texts),
            "unique_text_count": len(unique_texts),
            "uniqueness_ratio": len(unique_texts) / len(texts) if texts else 0,
            "vocabulary_size": len(unique_words),
            "total_words": len(all_words),
            "type_token_ratio": ttr,
            "length_stats": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0,
                "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
                "std": np.std(lengths) if lengths else 0
            }
        }
    
    @staticmethod
    def calculate_entity_coverage(entities: List[Dict[str, Any]], text_length: int) -> float:
        """
        คำนวณความครอบคลุมของเอนทิตี้ในข้อความ
        
        Args:
            entities (List[Dict[str, Any]]): รายการเอนทิตี้ (ที่มี start และ end)
            text_length (int): ความยาวของข้อความต้นฉบับ
            
        Returns:
            float: อัตราส่วนความครอบคลุม (0-1)
        """
        if not entities or text_length <= 0:
            return 0.0
            
        # สร้างอาร์เรย์บูลีนเพื่อระบุว่าแต่ละตำแหน่งถูกครอบคลุมหรือไม่
        coverage = [False] * text_length
        
        # ทำเครื่องหมายตำแหน่งที่ถูกครอบคลุมโดยเอนทิตี้
        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            
            if 0 <= start < text_length and 0 < end <= text_length and start < end:
                for i in range(start, end):
                    coverage[i] = True
        
        # คำนวณอัตราส่วนความครอบคลุม
        return sum(coverage) / text_length
    
    @staticmethod
    def calculate_summarization_quality(articles: List[str], summaries: List[str]) -> Dict[str, Any]:
        """
        คำนวณคุณภาพของการสรุปความ
        
        Args:
            articles (List[str]): รายการบทความต้นฉบับ
            summaries (List[str]): รายการบทสรุป
            
        Returns:
            Dict[str, Any]: เมตริกคุณภาพการสรุปความ
        """
        if len(articles) != len(summaries):
            return {"error": "จำนวนบทความและบทสรุปไม่เท่ากัน"}
            
        # คำนวณอัตราส่วนการบีบอัด (compression ratio)
        compression_ratios = [len(summary) / len(article) if len(article) > 0 else 0 
                              for article, summary in zip(articles, summaries)]
        
        # ตรวจสอบบทสรุปที่เป็นส่วนหนึ่งของบทความต้นฉบับ
        substring_counts = sum(1 for article, summary in zip(articles, summaries) 
                               if summary in article)
        
        # คำนวณอัตราส่วนคำที่ซ้ำกัน
        overlap_scores = []
        for article, summary in zip(articles, summaries):
            article_words = set(article.lower().split())
            summary_words = set(summary.lower().split())
            
            if not summary_words:
                overlap_scores.append(0)
                continue
                
            intersection = article_words & summary_words
            overlap = len(intersection) / len(summary_words)
            overlap_scores.append(overlap)
        
        return {
            "sample_count": len(articles),
            "compression_ratio": {
                "mean": np.mean(compression_ratios),
                "min": min(compression_ratios),
                "max": max(compression_ratios)
            },
            "substring_ratio": substring_counts / len(articles) if articles else 0,
            "content_overlap": {
                "mean": np.mean(overlap_scores),
                "min": min(overlap_scores),
                "max": max(overlap_scores)
            }
        }
