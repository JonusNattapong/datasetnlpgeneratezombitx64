"""
Dataset Validator
---------------
ระบบตรวจสอบคุณภาพชุดข้อมูลที่สร้างขึ้น
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter
import pandas as pd
import numpy as np

class DatasetValidator:
    """ตัวตรวจสอบคุณภาพชุดข้อมูล NLP"""
    
    def __init__(self, min_text_length: int = 10, max_text_length: int = 1000):
        """
        ตัวแปลงเริ่มต้นตัวตรวจสอบข้อมูล
        
        Args:
            min_text_length (int): ความยาวข้อความขั้นต่ำที่ยอมรับได้
            max_text_length (int): ความยาวข้อความสูงสุดที่ยอมรับได้
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # รายการคำที่พบบ่อยที่อาจเป็นปัญหา
        self.suspicious_patterns = [
            r"error", r"undefined", r"null", r"nan", 
            r"เกิดข้อผิดพลาด", r"ไม่พบข้อมูล", r"ไม่สามารถ"
        ]
    
    def validate_sample(self, sample: Dict[str, Any], task_type: str) -> Tuple[bool, str]:
        """
        ตรวจสอบตัวอย่างข้อมูลหนึ่งรายการ
        
        Args:
            sample (Dict[str, Any]): ตัวอย่างข้อมูล
            task_type (str): ประเภทงาน
            
        Returns:
            Tuple[bool, str]: ผลการตรวจสอบ (True/False) และเหตุผล
        """
        if not sample:
            return False, "ตัวอย่างว่างเปล่า"
        
        if task_type == "text_classification":
            text = sample.get("text", "")
            label = sample.get("label", "")
            
            if not text or not label:
                return False, "ไม่มีข้อความหรือป้ายกำกับ"
            
            if not (self.min_text_length <= len(text) <= self.max_text_length):
                return False, f"ความยาวข้อความไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัย: {pattern}"
        
        elif task_type in ["token_classification", "ner"]:
            text = sample.get("text", "")
            tokens = sample.get("tokens", [])
            tags = sample.get("tags", [])
            
            if not text or not tokens or not tags:
                return False, "ไม่มีข้อความ โทเค็น หรือแท็ก"
            
            if len(tokens) != len(tags):
                return False, "จำนวนโทเค็นและแท็กไม่ตรงกัน"
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัย: {pattern}"
        
        elif task_type == "question_answering":
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            
            if not context or not questions:
                return False, "ไม่มีบริบทหรือคำถาม"
            
            for question in questions:
                if not question.get("question") or not question.get("answer"):
                    return False, "คำถามหรือคำตอบว่างเปล่า"
                
                if not (self.min_text_length <= len(question["question"]) <= self.max_text_length):
                    return False, f"ความยาวคำถามไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
                
                for pattern in self.suspicious_patterns:
                    if re.search(pattern, question["question"], re.IGNORECASE):
                        return False, f"พบคำที่น่าสงสัยในคำถาม: {pattern}"
                    if re.search(pattern, question["answer"], re.IGNORECASE):
                        return False, f"พบคำที่น่าสงสัยในคำตอบ: {pattern}"
        
        elif task_type == "summarization":
            article = sample.get("article", "")
            summary = sample.get("summary", "")
            
            if not article or not summary:
                return False, "ไม่มีบทความหรือสรุป"
            
            if not (self.min_text_length <= len(article) <= self.max_text_length):
                return False, f"ความยาวบทความไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
            
            if not (self.min_text_length <= len(summary) <= self.max_text_length):
                return False, f"ความยาวสรุปไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, article, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัยในบทความ: {pattern}"
                if re.search(pattern, summary, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัยในสรุป: {pattern}"
        
        elif task_type == "translation":
            source = sample.get("source", "")
            target = sample.get("target", "")
            
            if not source or not target:
                return False, "ไม่มีข้อความต้นทางหรือข้อความปลายทาง"
            
            if not (self.min_text_length <= len(source) <= self.max_text_length):
                return False, f"ความยาวข้อความต้นทางไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
            
            if not (self.min_text_length <= len(target) <= self.max_text_length):
                return False, f"ความยาวข้อความปลายทางไม่อยู่ในช่วงที่ยอมรับได้ ({self.min_text_length}-{self.max_text_length})"
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัยในข้อความต้นทาง: {pattern}"
                if re.search(pattern, target, re.IGNORECASE):
                    return False, f"พบคำที่น่าสงสัยในข้อความปลายทาง: {pattern}"
        
        else:
            return False, f"ไม่รองรับประเภทงาน: {task_type}"
        
        return True, "ผ่านการตรวจสอบ"
    
    def validate_dataset(self, samples: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """
        ตรวจสอบคุณภาพชุดข้อมูลทั้งหมด
        
        Args:
            samples (List[Dict[str, Any]]): ชุดข้อมูล
            task_type (str): ประเภทงาน
            
        Returns:
            Dict[str, Any]: รายงานผลการตรวจสอบ
        """
        if not samples:
            return {"status": "error", "message": "ไม่มีตัวอย่างในชุดข้อมูล"}
            
        results = []
        valid_count = 0
        invalid_count = 0
        issues = []
        
        for idx, sample in enumerate(samples):
            is_valid, reason = self.validate_sample(sample, task_type)
            results.append({
                "index": idx,
                "is_valid": is_valid,
                "reason": reason
            })
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                issues.append({
                    "index": idx,
                    "reason": reason
                })
        
        # คำนวณอัตราส่วนตัวอย่างที่ถูกต้อง
        validity_ratio = valid_count / len(samples) if samples else 0
        
        # ตรวจสอบความหลากหลายของข้อมูล
        diversity_metrics = self._calculate_diversity(samples, task_type)
        
        return {
            "status": "success",
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "validity_ratio": validity_ratio,
            "diversity_metrics": diversity_metrics,
            "issues": issues,
            "passed": validity_ratio >= 0.8  # ผ่านหากมีตัวอย่างที่ถูกต้องอย่างน้อย 80%
        }
    
    def _calculate_diversity(self, samples: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """
        คำนวณความหลากหลายของข้อมูล
        
        Args:
            samples (List[Dict[str, Any]]): ชุดข้อมูล
            task_type (str): ประเภทงาน
            
        Returns:
            Dict[str, Any]: เมตริกความหลากหลาย
        """
        metrics = {}
        
        if task_type == "text_classification":
            # ตรวจสอบความสมดุลของคลาส
            labels = [s.get("label") for s in samples if "label" in s]
            label_counts = Counter(labels)
            metrics["label_distribution"] = {str(k): v for k, v in label_counts.items()}
            
            # คำนวณความสมดุล
            total = sum(label_counts.values())
            balance_score = sum((count/total)**2 for count in label_counts.values())
            # ค่า balance score ยิ่งเข้าใกล้ 1/n (โดย n คือจำนวนคลาส) แสดงว่ายิ่งสมดุล
            metrics["class_balance_score"] = balance_score
            
        elif task_type in ["token_classification", "ner"]:
            # นับประเภท entity
            entity_types = []
            for sample in samples:
                if "entities" in sample:
                    entity_types.extend([e["type"] for e in sample["entities"] if "type" in e])
            
            entity_type_counts = Counter(entity_types)
            metrics["entity_type_distribution"] = {str(k): v for k, v in entity_type_counts.items()}
        
        # คำนวณ word overlap - ตรวจสอบว่ามีประโยคซ้ำกันมากไหม
        text_fields = []
        for s in samples:
            if "text" in s:
                text_fields.append(s["text"])
            elif "article" in s:
                text_fields.append(s["article"])
            elif "context" in s:
                text_fields.append(s["context"])
                
        # นับจำนวนข้อความที่ไม่ซ้ำ
        unique_texts = set(text_fields)
        metrics["unique_text_ratio"] = len(unique_texts) / len(text_fields) if text_fields else 0
        
        # คำนวณความยาวของข้อความ
        if text_fields:
            lengths = [len(t) for t in text_fields]
            metrics["length_stats"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "std": (sum((l - (sum(lengths) / len(lengths)))**2 for l in lengths) / len(lengths))**0.5
            }
            
        return metrics