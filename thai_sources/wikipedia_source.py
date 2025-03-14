"""
Wikipedia Source
---------------
เชื่อมต่อกับ Wikipedia ภาษาไทยเพื่อดึงข้อมูลสำหรับการสร้างชุดข้อมูล
"""

import os
import requests
import json
import time
from typing import List, Dict, Any, Optional
from .base_source import BaseSource

class WikipediaSource(BaseSource):
    """แหล่งข้อมูลจาก Wikipedia ภาษาไทย"""
    
    def __init__(self, cache_dir: str = "thai_sources_cache/wikipedia"):
        """
        ตัวแปลงเริ่มต้นสำหรับแหล่งข้อมูล Wikipedia ภาษาไทย
        
        Args:
            cache_dir (str): ไดเรกทอรีสำหรับจัดเก็บข้อมูล cache
        """
        super().__init__(cache_dir)
        self.api_url = "https://th.wikipedia.org/w/api.php"
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาบทความจาก Wikipedia ภาษาไทย
        
        Args:
            query (str): คำค้นหา
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการบทความ
        """
        cache_file = os.path.join(self.cache_dir, f"search_{query.replace(' ', '_')}.json")
        
        # ตรวจสอบ cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # ถ้าไม่มี cache ให้ค้นหาจาก API
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "srnamespace": 0,
            "srwhat": "text"
        }
        
        response = requests.get(self.api_url, params=params)
        data = response.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "page_id": item.get("pageid"),
                "title": item.get("title"),
                "snippet": item.get("snippet").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            })
        
        # บันทึก cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def get_content(self, page_id: str) -> str:
        """
        ดึงเนื้อหาบทความจาก Wikipedia
        
        Args:
            page_id (str): ID ของหน้า
            
        Returns:
            str: เนื้อหาข้อความ
        """
        cache_file = os.path.join(self.cache_dir, f"content_{page_id}.txt")
        
        # ตรวจสอบ cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        # ถ้าไม่มี cache ให้ดึงข้อมูลจาก API
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": 0,
            "explaintext": 1,
            "pageids": page_id
        }
        
        response = requests.get(self.api_url, params=params)
        data = response.json()
        
        content = data.get("query", {}).get("pages", {}).get(str(page_id), {}).get("extract", "")
        
        # บันทึก cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    
    def process_query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาและดึงเนื้อหาบทความจาก Wikipedia
        
        Args:
            query (str): คำค้นหา
            max_results (int): จำนวนผลลัพธ์สูงสุด
            
        Returns:
            List[Dict[str, Any]]: รายการบทความพร้อมเนื้อหา
        """
        search_results = self.search(query, max_results)
        
        articles = []
        for result in search_results:
            try:
                page_id = result.get("page_id")
                content = self.get_content(page_id)
                
                if content:
                    articles.append({
                        "paper_id": f"wiki_{page_id}",
                        "title": result.get("title"),
                        "text": content,
                        "source": "wikipedia_th"
                    })
                
                # หน่วงเวลาเพื่อไม่ให้ส่งคำขอถี่เกินไป
                time.sleep(1)
                
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการดึงข้อมูลหน้า {page_id}: {e}")
        
        return articles
