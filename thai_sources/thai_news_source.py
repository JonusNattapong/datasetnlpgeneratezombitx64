"""
Thai News Source
--------------
เชื่อมต่อกับแหล่งข่าวภาษาไทยผ่าน RSS/API
"""

import os
import requests
import json
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from .base_source import BaseSource

class ThaiNewsSource(BaseSource):
    """แหล่งข้อมูลข่าวภาษาไทย"""
    
    def __init__(self, cache_dir: str = "thai_sources_cache/thai_news"):
        super().__init__(cache_dir)
        
        # แหล่งข่าวที่รองรับ
        self.news_sources = {
            "thairath": {
                "name": "ไทยรัฐ",
                "rss_url": "https://www.thairath.co.th/rss/news",
                "base_url": "https://www.thairath.co.th"
            },
            "matichon": {
                "name": "มติชน",
                "rss_url": "https://www.matichon.co.th/feed",
                "base_url": "https://www.matichon.co.th" 
            }
        }
    
    def search(self, query: str, max_results: int = 10, source: str = None) -> List[Dict[str, Any]]:
        """ค้นหาข่าวตามคำค้นหา"""
        results = []
        sources = [source] if source else self.news_sources.keys()
        
        for src in sources:
            if src not in self.news_sources:
                continue
                
            feed = feedparser.parse(self.news_sources[src]["rss_url"])
            
            for entry in feed.entries[:max_results]:
                if query.lower() in entry.title.lower() or query.lower() in entry.description.lower():
                    results.append({
                        "title": entry.title,
                        "description": entry.description,
                        "link": entry.link,
                        "source": src,
                        "published": entry.published
                    })
        
        return results[:max_results]
    
    def get_content(self, url: str) -> str:
        """ดึงเนื้อหาข่าวจาก URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ลบส่วนที่ไม่ต้องการ
            for tag in soup(['script', 'style', 'iframe', 'nav']):
                tag.decompose()
            
            return soup.get_text(strip=True)
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def process_query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """ค้นหาและดึงเนื้อหาข่าว"""
        search_results = self.search(query, max_results)
        
        articles = []
        for result in search_results:
            content = self.get_content(result["link"])
            if content:
                articles.append({
                    "paper_id": f"news_{hash(result['link'])}",
                    "title": result["title"],
                    "text": content,
                    "source": f"{result['source']}_{result['published']}"
                })
        
        return articles
