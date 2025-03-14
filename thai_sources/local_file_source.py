"""
Local File Source
----------------
อ่านข้อมูลจากไฟล์ท้องถิ่นในรูปแบบต่างๆ (PDF, DOCX, TXT)
"""

import os
import glob
from typing import List, Dict, Any, Optional
import uuid
from .base_source import BaseSource

class LocalFileSource(BaseSource):
    """แหล่งข้อมูลจากไฟล์ท้องถิ่น"""
    
    def __init__(self, cache_dir: str = "thai_sources_cache/local_files"):
        """
        ตัวแปลงเริ่มต้นสำหรับแหล่งข้อมูลไฟล์ท้องถิ่น
        
        Args:
            cache_dir (str): ไดเรกทอรีสำหรับจัดเก็บข้อมูล cache
        """
        super().__init__(cache_dir)
    
    def search(self, path_pattern: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาไฟล์ตามรูปแบบเส้นทาง
        
        Args:
            path_pattern (str): รูปแบบเส้นทางไฟล์ (เช่น "data/*.pdf")
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการไฟล์
        """
        files = glob.glob(path_pattern)[:max_results]
        
        results = []
        for file_path in files:
            file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_path))
            results.append({
                "file_id": file_id,
                "path": file_path,
                "filename": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1].lower()
            })
        
        return results
    
    def get_content(self, file_path: str) -> str:
        """
        อ่านเนื้อหาไฟล์ตามประเภท
        
        Args:
            file_path (str): เส้นทางไฟล์
            
        Returns:
            str: เนื้อหาข้อความ
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.pdf':
            try:
                import fitz  # PyMuPDF
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            except ImportError:
                raise ImportError("PyMuPDF (fitz) เป็นตัวแปรที่จำเป็นสำหรับการอ่าน PDF. ติดตั้งด้วย `pip install PyMuPDF`")
        
        elif extension in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                raise ImportError("python-docx เป็นตัวแปรที่จำเป็นสำหรับการอ่าน DOCX. ติดตั้งด้วย `pip install python-docx`")
        
        else:
            raise ValueError(f"ไม่รองรับนามสกุลไฟล์: {extension}")
    
    def process_query(self, path_pattern: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาและอ่านเนื้อหาจากไฟล์ท้องถิ่น
        
        Args:
            path_pattern (str): รูปแบบเส้นทางไฟล์
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการไฟล์พร้อมเนื้อหา
        """
        search_results = self.search(path_pattern, max_results)
        
        documents = []
        for result in search_results:
            try:
                file_path = result.get("path")
                content = self.get_content(file_path)
                
                if content:
                    documents.append({
                        "paper_id": result.get("file_id"),
                        "title": result.get("filename"),
                        "text": content,
                        "source": "local_file"
                    })
                
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_path}: {e}")
        
        return documents
