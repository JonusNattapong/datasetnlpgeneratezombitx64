"""
Thai-Specific Quality Metrics
--------------------------
ระบบวัดคุณภาพเฉพาะสำหรับข้อมูลภาษาไทย
"""

import re
from typing import List, Dict, Any, Optional, Set
import numpy as np
from collections import Counter

class ThaiSpecificMetrics:
    """เครื่องมือวัดคุณภาพสำหรับข้อมูลภาษาไทยโดยเฉพาะ"""

    def __init__(self):
        """ตัวแปลงเริ่มต้นของเมตริกภาษาไทย"""
        # กำหนดวรรณยุกต์ไทยและรูปแบบที่เกี่ยวข้อง
        self.thai_tones = ['่', '้', '๊', '๋', '']  # รวมไม่มีวรรณยุกต์
        
        # คำที่มักสะกดผิดในภาษาไทย
        self.common_misspellings = {
            'ทำงาน': ['ทํางาน'],
            'เสร็จ': ['เสร็ช', 'เสรจ'],
            'ประเทศ': ['ปะเทศ', 'ประเทษ'],
            # เพิ่มเติมตามต้องการ
        }

    def calculate_tonal_metrics(self, text: str) -> Dict[str, Any]:
        """
        คำนวณเมตริกเกี่ยวกับวรรณยุกต์ในข้อความ
        
        Args:
            text (str): ข้อความภาษาไทย
            
        Returns:
            Dict[str, Any]: เมตริกเกี่ยวกับวรรณยุกต์
        """
        # นับการใช้วรรณยุกต์แต่ละชนิด
        tone_counts = {tone: len(re.findall(tone, text)) for tone in self.thai_tones if tone}
        
        # นับคำที่ไม่มีวรรณยุกต์
        words = text.split()
        no_tone_words = sum(1 for word in words if not any(tone in word for tone in self.thai_tones if tone))
        
        # คำนวณสัดส่วนการใช้วรรณยุกต์
        total_words = len(words)
        tonal_distribution = {
            tone: count/total_words if total_words > 0 else 0 
            for tone, count in tone_counts.items()
        }
        
        return {
            "tone_counts": tone_counts,
            "no_tone_words": no_tone_words,
            "tonal_distribution": tonal_distribution,
            "tonal_complexity_score": len(tone_counts) / total_words if total_words > 0 else 0
        }

    def validate_thai_orthography(self, text: str) -> Dict[str, Any]:
        """
        ตรวจสอบความถูกต้องของการสะกดภาษาไทย
        
        Args:
            text (str): ข้อความภาษาไทย
            
        Returns:
            Dict[str, Any]: ผลการตรวจสอบการสะกด
        """
        # ตรวจสอบการสะกดผิด
        misspelling_found = []
        for correct, variants in self.common_misspellings.items():
            for variant in variants:
                if variant in text:
                    misspelling_found.append({
                        "incorrect": variant,
                        "correct": correct
                    })
        
        # ตรวจสอบการใช้วรรณยุกต์ซ้ำซ้อน
        double_tones = []
        for i in range(len(text)-1):
            if text[i] in self.thai_tones and text[i+1] in self.thai_tones:
                double_tones.append(text[i:i+2])
        
        return {
            "misspellings": misspelling_found,
            "double_tone_instances": double_tones,
            "orthographic_score": 1.0 - (len(misspelling_found) / len(text.split()) if text else 0)
        }

    def calculate_semantic_clusters(self, texts: List[str]) -> Dict[str, Any]:
        """
        วิเคราะห์กลุ่มความหมายในชุดข้อความภาษาไทย
        
        Args:
            texts (List[str]): รายการข้อความภาษาไทย
            
        Returns:
            Dict[str, Any]: การวิเคราะห์กลุ่มความหมาย
        """
        # แบ่งประเภทข้อความตามลักษณะเบื้องต้น
        formal_indicators = ['ครับ', 'ค่ะ', 'ท่าน', 'กระผม', 'ดิฉัน']
        informal_indicators = ['จ้า', 'ครับผม', 'จ๊ะ', 'น้า']
        question_indicators = ['ไหม', 'หรือ', 'เหรอ', 'มั้ย']
        
        style_counts = {
            "formal": sum(1 for text in texts if any(ind in text for ind in formal_indicators)),
            "informal": sum(1 for text in texts if any(ind in text for ind in informal_indicators)),
            "question": sum(1 for text in texts if any(ind in text for ind in question_indicators))
        }
        
        return {
            "style_distribution": {
                style: count/len(texts) if texts else 0 
                for style, count in style_counts.items()
            },
            "semantic_diversity_score": len(set(texts))/len(texts) if texts else 0,
            "style_counts": style_counts
        }

    def analyze_text_formality(self, text: str) -> Dict[str, Any]:
        """
        วิเคราะห์ความเป็นทางการของข้อความภาษาไทย
        
        Args:
            text (str): ข้อความภาษาไทย
            
        Returns:
            Dict[str, Any]: ผลการวิเคราะห์ความเป็นทางการ
        """
        # คำบ่งชี้ความเป็นทางการ
        formal_markers = {
            "honorifics": ['ท่าน', 'คุณ', 'นาย', 'นาง', 'นางสาว'],
            "formal_pronouns": ['กระผม', 'ดิฉัน', 'ข้าพเจ้า'],
            "polite_particles": ['ครับ', 'ค่ะ', 'นะคะ', 'นะครับ']
        }
        
        # นับการปรากฏของคำบ่งชี้
        marker_counts = {
            category: sum(1 for marker in markers if marker in text)
            for category, markers in formal_markers.items()
        }
        
        # คำนวณคะแนนความเป็นทางการ (0-1)
        total_markers = sum(marker_counts.values())
        words = len(text.split())
        formality_score = total_markers / words if words > 0 else 0
        
        return {
            "marker_counts": marker_counts,
            "formality_score": formality_score,
            "formal_word_ratio": total_markers / words if words > 0 else 0
        }
