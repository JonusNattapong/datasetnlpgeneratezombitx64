"""
Thai Text Cleaner
---------------
เครื่องมือทำความสะอาดและปรับปรุงข้อความภาษาไทย
"""

import re
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter

# ตรวจสอบว่ามี pythainlp หรือไม่
try:
    import pythainlp
    from pythainlp.corpus import thai_stopwords
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import normalize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False
    print("คำเตือน: ไม่พบ PyThaiNLP - การทำความสะอาดข้อความภาษาไทยจะทำงานได้ไม่สมบูรณ์")

class ThaiTextCleaner:
    """เครื่องมือทำความสะอาดข้อความภาษาไทยสำหรับชุดข้อมูล NLP"""
    
    def __init__(self, remove_stopwords: bool = False, normalize_chars: bool = True):
        """
        ตัวแปลงเริ่มต้นเครื่องมือทำความสะอาดข้อความไทย
        
        Args:
            remove_stopwords (bool): ต้องการลบ stopwords หรือไม่
            normalize_chars (bool): ต้องการปรับอักขระให้เป็นมาตรฐานหรือไม่
        """
        self.remove_stopwords = remove_stopwords
        self.normalize_chars = normalize_chars
        
        # โหลด stopwords ถ้ามี pythainlp
        if HAS_PYTHAINLP and remove_stopwords:
            self.stopwords = set(thai_stopwords())
        else:
            self.stopwords = set()
            
        # รายการ regular expressions สำหรับการทำความสะอาดข้อความ
        self.cleanup_patterns = [
            (r'http\S+|www\S+|https\S+', ''),           # ลบ URLs
            (r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.!?]', ' '),   # เก็บแค่ตัวอักษรไทย อังกฤษ ตัวเลข และเครื่องหมายบางอย่าง
            (r'\s+', ' '),                              # แทนที่เว้นวรรคซ้ำด้วยเว้นวรรคเดียว
            (r'\n+', '\n')                              # แทนที่ขึ้นบรรทัดใหม่ซ้ำด้วยขึ้นบรรทัดใหม่เดียว
        ]
    
    def clean_text(self, text: str) -> str:
        """
        ทำความสะอาดข้อความภาษาไทย
        
        Args:
            text (str): ข้อความที่ต้องการทำความสะอาด
            
        Returns:
            str: ข้อความที่ทำความสะอาดแล้ว
        """
        if not text or not isinstance(text, str):
            return ""
            
        # ทำความสะอาดด้วย regular expressions
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
            
        # ปรับอักขระให้เป็นมาตรฐาน ถ้าเปิดใช้งาน
        if self.normalize_chars and HAS_PYTHAINLP:
            text = normalize(text)
            
        # ลบ stopwords ถ้าเปิดใช้งาน
        if self.remove_stopwords and HAS_PYTHAINLP:
            words = word_tokenize(text)
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
            
        return text.strip()
    
    def clean_dataset(self, samples: List[Dict[str, Any]], text_fields: List[str]) -> List[Dict[str, Any]]:
        """
        ทำความสะอาดข้อความในชุดข้อมูล
        
        Args:
            samples (List[Dict[str, Any]]): รายการตัวอย่างข้อมูล
            text_fields (List[str]): ชื่อฟิลด์ที่เก็บข้อความ (เช่น ["text", "article"])
            
        Returns:
            List[Dict[str, Any]]: ชุดข้อมูลที่ทำความสะอาดแล้ว
        """
        cleaned_samples = []
        
        for sample in samples:
            cleaned_sample = sample.copy()
            
            for field in text_fields:
                if field in sample and isinstance(sample[field], str):
                    cleaned_sample[field] = self.clean_text(sample[field])
                    
            cleaned_samples.append(cleaned_sample)
            
        return cleaned_samples
    
    def fix_common_thai_errors(self, text: str) -> str:
        """
        แก้ไขข้อผิดพลาดทั่วไปในข้อความภาษาไทย
        
        Args:
            text (str): ข้อความภาษาไทย
            
        Returns:
            str: ข้อความที่แก้ไขแล้ว
        """
        # แก้ไขเว้นวรรคผิดที่
        if HAS_PYTHAINLP:
            # ทอเคไนซ์และเชื่อมกลับด้วยการเว้นวรรคที่ถูกต้อง
            tokens = word_tokenize(text)
            text = ' '.join(tokens)
            
        # แก้ไขไม้ยมก (ๆ) ที่มีช่องว่างผิดที่
        text = re.sub(r'(\S) ๆ', r'\1ๆ', text)
        
        # แก้ไขวรรคตอนตาม Thai Writing Style
        text = re.sub(r' ([.,!?])', r'\1', text)
        
        # แก้ไขการมีอักษรภาษาอังกฤษติดกับภาษาไทย
        text = re.sub(r'([ก-๙])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([ก-๙])', r'\1 \2', text)
        
        return text.strip()
    
    def fix_thai_unicode_issues(self, text: str) -> str:
        """
        แก้ไขปัญหา Unicode ในภาษาไทย
        
        Args:
            text (str): ข้อความภาษาไทย
            
        Returns:
            str: ข้อความที่แก้ไขแล้ว
        """
        # แก้ไขปัญหาสระซ้อนหรือวรรณยุกต์ที่ไม่ถูกลำดับ
        if HAS_PYTHAINLP:
            text = normalize(text)
        
        # การแทนที่เฉพาะ
        replacements = {
            # แก้ไขอักขระพิเศษ
            '\u200b': '',  # ลบ Zero-Width Space
            '\u2060': '',  # ลบ Word Joiner
            '\ufeff': '',  # ลบ Zero-Width No-Break Space
            
            # แก้ไขวรรณยุกต์และสระที่อาจมีปัญหา
            '\u0e33': '\u0e30\u0e30',  # แก้ไข สระอำ ที่เป็นปัญหาบางครั้ง
            
            # แก้ไขเลขไทยเป็นเลขอารบิก
            '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4', 
            '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
