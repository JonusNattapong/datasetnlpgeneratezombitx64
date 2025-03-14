import re
import nltk
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from .data_quality.thai_specific_metrics import ThaiSpecificMetrics
from .data_quality.dual_adversarial_validator import DualAdversarialValidator

# ดาวน์โหลดทรัพยากร NLTK ที่จำเป็น
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    """ประมวลผลข้อความดิบให้พร้อมสำหรับการสร้างชุดข้อมูล"""
    
    def __init__(self, lang: str = "english", ai_service_manager=None):
        """
        ตัวแปลงเริ่มต้นตัวประมวลผลข้อความ
        
        Args:
            lang (str): ภาษาสำหรับการประมวลผล (สำหรับ stopwords)
            ai_service_manager: ตัวจัดการ AI service สำหรับการตรวจสอบคุณภาพ
        """
        self.lang = lang
        self.stop_words = set(stopwords.words(lang)) if lang in stopwords.fileids() else set()
        self.thai_metrics = ThaiSpecificMetrics()
        self.dual_validator = DualAdversarialValidator(ai_service_manager) if ai_service_manager else None
        
    def clean_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        ทำความสะอาดข้อความ เช่น ลบอักขระพิเศษ, เว้นวรรค, ฯลฯ
        
        Args:
            text (str): ข้อความที่ต้องการทำความสะอาด
            
        Returns:
            str: ข้อความที่ทำความสะอาดแล้ว
        """
        # เก็บข้อมูลการทำความสะอาด
        cleaning_stats = {
            "original_length": len(text),
            "urls_removed": len(re.findall(r'http\S+|www\S+|https\S+', text)),
            "special_chars_removed": 0,
            "spaces_normalized": 0
        }

        # ลบ URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # ลบอักขระพิเศษและตัวเลข (แต่รักษา . ! ? สำหรับการแยกประโยค)
        special_chars = re.findall(r'[^\w\s.!?]', text)
        cleaning_stats["special_chars_removed"] = len(special_chars)
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        
        # นับช่องว่างซ้ำ
        spaces = re.findall(r'\s+', text)
        cleaning_stats["spaces_normalized"] = sum(len(s) - 1 for s in spaces)
        
        # ลบเว้นวรรคซ้ำ
        text = re.sub(r'\s+', ' ', text).strip()
        
        # ถ้าเป็นภาษาไทย ตรวจสอบการสะกดและวรรณยุกต์
        if self.lang == "thai":
            thai_validation = self.thai_metrics.validate_thai_orthography(text)
            cleaning_stats["thai_validation"] = thai_validation
        
        cleaning_stats["final_length"] = len(text)
        cleaning_stats["reduction_ratio"] = 1 - (len(text) / cleaning_stats["original_length"])
        
        return text, cleaning_stats
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """
        แบ่งบทความเป็นส่วนต่างๆ (บทคัดย่อ, บทนำ, วิธีการ, ฯลฯ)
        
        Args:
            text (str): ข้อความบทความเต็ม
            
        Returns:
            Dict[str, str]: พจนานุกรมของส่วนต่างๆ
        """
        # รูปแบบการจับคู่หัวข้อส่วนทั่วไปในบทความวิชาการ
        section_patterns = {
            'abstract': r'(?i)abstract[\s\n]+(.+?)(?=\n\s*(?:introduction|1[\.\s]|i[\.\s]|$))',
            'introduction': r'(?i)(?:introduction|1[\.\s]|i[\.\s])[\s\n]+(.+?)(?=\n\s*(?:related work|background|method|2[\.\s]|ii[\.\s]|$))',
            'method': r'(?i)(?:method|methodology|approach|proposed method|3[\.\s]|iii[\.\s])[\s\n]+(.+?)(?=\n\s*(?:experiment|evaluation|result|4[\.\s]|iv[\.\s]|$))',
            'results': r'(?i)(?:result|experiment|evaluation|4[\.\s]|iv[\.\s])[\s\n]+(.+?)(?=\n\s*(?:discussion|conclusion|5[\.\s]|v[\.\s]|$))',
            'conclusion': r'(?i)(?:conclusion|discussion|future work|5[\.\s]|v[\.\s])[\s\n]+(.+?)(?=\n\s*(?:reference|acknowledgment|appendix|$))'
        }
        
        sections = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
            else:
                sections[section_name] = ""
                
        # เพิ่มส่วน "full_text" สำหรับข้อความทั้งหมด
        sections['full_text'] = text
        
        return sections
    
    def extract_sentences(self, text: str, min_length: int = 10, max_length: int = 1000) -> List[str]:
        """
        แยกประโยคจากข้อความที่สะอาด
        
        Args:
            text (str): ข้อความที่ทำความสะอาดแล้ว
            min_length (int): ความยาวประโยคขั้นต่ำที่ต้องการ
            max_length (int): ความยาวประโยคสูงสุดที่ต้องการ
            
        Returns:
            List[str]: รายการประโยค
        """
        sentences = sent_tokenize(text)
        
        # กรองประโยคตามความยาว
        filtered_sentences = [
            s.strip() for s in sentences 
            if min_length <= len(s) <= max_length
        ]
        
        return filtered_sentences
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        แบ่งข้อความเป็นส่วนที่มีขนาดเท่ากัน (สำหรับการประมวลผลข้อความยาว)
        
        Args:
            text (str): ข้อความ
            chunk_size (int): ขนาดในอักขระของแต่ละส่วน
            overlap (int): จำนวนอักขระที่ซ้อนทับระหว่างส่วน
            
        Returns:
            List[str]: รายการส่วนข้อความ
        """
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # หาจุดสิ้นสุดของส่วนนี้
            end = min(start + chunk_size, len(text))
            
            # ถ้าไม่ได้อยู่ที่ตำแหน่งสุดท้าย พยายามค้นหาเว้นวรรคเพื่อตัดคำให้เรียบร้อย
            if end < len(text):
                # หาช่องว่างถัดไปเพื่อไม่ตัดคำกลางคำ
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # ไม่พบช่องว่างในระยะที่เหมาะสม
                    end = start + chunk_size  # กลับไปใช้ความยาวต้นฉบับ
            
            # เพิ่มส่วนนี้ในรายการ
            chunks.append(text[start:end].strip())
            
            # ขยับจุดเริ่มต้นสำหรับส่วนถัดไป โดยลบความซ้อนทับ
            start = end - overlap
            if start < 0:
                start = 0
                
        return chunks
    
    async def process_text(self, text: str, validate: bool = True) -> Dict[str, Any]:
        """
        ประมวลผลข้อความเต็มเพื่อเตรียมสำหรับการสร้างชุดข้อมูล
        
        Args:
            text (str): ข้อความดิบ
            
        Returns:
            Dict[str, Any]: ข้อความที่ประมวลผลแล้วในรูปแบบต่างๆ
        """
        # ทำความสะอาดข้อความ
        clean_text, cleaning_stats = self.clean_text(text)
        
        # แยกส่วน
        sections = self.split_into_sections(clean_text)
        
        # แยกประโยคจากข้อความเต็ม
        sentences = self.extract_sentences(clean_text)
        
        # สร้างส่วนสำหรับข้อความยาว
        chunks = self.create_chunks(clean_text)
        
        # สถิติพื้นฐาน
        basic_stats = {
            'word_count': len(word_tokenize(clean_text)),
            'sentence_count': len(sentences),
            'chunk_count': len(chunks),
            'cleaning_stats': cleaning_stats
        }
        
        # วิเคราะห์ภาษาไทยถ้าเป็นภาษาไทย
        if self.lang == "thai":
            thai_metrics = {
                'tonal_metrics': self.thai_metrics.calculate_tonal_metrics(clean_text),
                'formality': self.thai_metrics.analyze_text_formality(clean_text)
            }
            if len(sentences) > 1:
                thai_metrics['semantic_clusters'] = self.thai_metrics.calculate_semantic_clusters(sentences)
            basic_stats['thai_metrics'] = thai_metrics
        
        # ตรวจสอบคุณภาพด้วย Dual-Adversarial System
        validation_results = None
        if validate and self.dual_validator:
            validation_results = await self.dual_validator.validate_samples(
                [{'text': s} for s in sentences],
                'text_classification'  # หรือประเภทงานที่เหมาะสม
            )
        
        return {
            'clean_text': clean_text,
            'sections': sections,
            'sentences': sentences,
            'chunks': chunks,
            'stats': basic_stats,
            'validation_results': validation_results
        }
