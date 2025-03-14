"""
Base Source 
-----------
Abstract class สำหรับแหล่งข้อมูลภาษาไทยทุกแหล่ง
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseSource(ABC):
    """แหล่งข้อมูลพื้นฐานสำหรับข้อมูลภาษาไทย"""
    
    def __init__(self, cache_dir: str = "thai_sources_cache"):
        """
        ตัวแปลงเริ่มต้นสำหรับแหล่งข้อมูลพื้นฐาน
        
        Args:
            cache_dir (str): ไดเรกทอรีสำหรับจัดเก็บข้อมูล cache
        """
        self.cache_dir = cache_dir
        self._setup_cache()
    
    def _setup_cache(self):
        """ตั้งค่าระบบ cache"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาข้อมูลจากแหล่งข้อมูล
        
        Args:
            query (str): คำค้นหา
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการผลลัพธ์
        """
        pass
    
    @abstractmethod
    def get_content(self, item_id: str) -> str:
        """
        ดึงเนื้อหาจากแหล่งข้อมูล
        
        Args:
            item_id (str): ID ของรายการข้อมูล
            
        Returns:
            str: เนื้อหาข้อความ
        """
        pass
    
    @abstractmethod
    def process_query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาและประมวลผลข้อมูลจากแหล่งข้อมูล
        
        Args:
            query (str): คำค้นหา
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการข้อมูลที่ประมวลผลแล้ว
        """
        pass
