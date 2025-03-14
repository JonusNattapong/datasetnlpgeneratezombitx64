"""
Thai Sources Module
------------------
โมดูลนี้รวบรวม connector สำหรับแหล่งข้อมูลภาษาไทยต่างๆ
"""

from .wikipedia_source import WikipediaSource
from .thai_corpus_source import ThaiCorpusSource
from .thai_news_source import ThaiNewsSource
from .local_file_source import LocalFileSource

__all__ = [
    'WikipediaSource',
    'ThaiCorpusSource', 
    'ThaiNewsSource',
    'LocalFileSource'
]
