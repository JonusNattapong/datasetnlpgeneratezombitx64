import arxiv
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import requests
import tempfile

class ArxivDownloader:
    """ดาวน์โหลดและแปลงเอกสาร PDF จาก arXiv"""
    
    def __init__(self, output_dir: str = "arxiv_data"):
        """
        ตัวแปลงเริ่มต้น arXiv
        
        Args:
            output_dir (str): ไดเรกทอรีที่จะบันทึกเอกสาร PDF และข้อความที่แปลงแล้ว
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        ค้นหาบทความตามคำค้นหา
        
        Args:
            query (str): คำค้นหา (เช่น 'nlp', 'transformer neural networks')
            max_results (int): จำนวนผลลัพธ์สูงสุดที่จะส่งคืน
            
        Returns:
            List[Dict]: รายการบทความพร้อมข้อมูลสำคัญ
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            results.append({
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'pdf_url': paper.pdf_url,
                'published': paper.published,
                'paper_id': paper.get_short_id()
            })
        
        return results
    
    def download_pdf(self, pdf_url: str, paper_id: str) -> str:
        """
        ดาวน์โหลด PDF จาก URL
        
        Args:
            pdf_url (str): URL ของ PDF
            paper_id (str): ID ของบทความสำหรับการตั้งชื่อไฟล์
            
        Returns:
            str: เส้นทางไฟล์ PDF ที่บันทึกไว้
        """
        pdf_path = os.path.join(self.output_dir, f"{paper_id}.pdf")
        
        if os.path.exists(pdf_path):
            print(f"PDF มีอยู่แล้ว: {pdf_path}")
            return pdf_path
            
        response = requests.get(pdf_url)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        print(f"ดาวน์โหลด PDF แล้ว: {pdf_path}")
        return pdf_path
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        แปลง PDF เป็นข้อความล้วน
        
        Args:
            pdf_path (str): เส้นทางไฟล์ PDF
            
        Returns:
            str: ข้อความที่แปลงแล้ว
        """
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการแปลง PDF: {e}")
        
        # บันทึกข้อความไปยังไฟล์
        text_path = pdf_path.replace('.pdf', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text
    
    def process_query(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        ค้นหา ดาวน์โหลด และแปลงเอกสารจาก arXiv
        
        Args:
            query (str): คำค้นหา
            max_results (int): จำนวนบทความสูงสุด
            
        Returns:
            List[Dict]: รายการบทความพร้อมข้อความที่แปลงแล้ว
        """
        papers = self.search_papers(query, max_results)
        
        for paper in papers:
            pdf_path = self.download_pdf(paper['pdf_url'], paper['paper_id'])
            paper['text'] = self.extract_text_from_pdf(pdf_path)
            paper['pdf_path'] = pdf_path
            paper['text_path'] = pdf_path.replace('.pdf', '.txt')
        
        return papers