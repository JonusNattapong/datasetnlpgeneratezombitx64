import os
import json
import httpx
import asyncio
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
import sqlite3
from datetime import datetime, timedelta

class AIServiceConfig:
    def __init__(self):
        # Initialize with environment variables or configuration file
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.thai_models = {
            "wangchanberta": "airesearch/wangchanberta-base-att-spm-uncased",
            "thaibert": "monsoon-nlp/bert-base-thai",
            "xlm-roberta-thai": "xlm-roberta-base"
        }
        
        # Cache settings
        self.cache_dir = "ai_cache"
        self.cache_db = os.path.join(self.cache_dir, "cache.db")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache database
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_db)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS cache
                     (key TEXT PRIMARY KEY, value TEXT, timestamp DATETIME)''')
        conn.commit()
        conn.close()

class BaseAIService(ABC):
    def __init__(self, config: AIServiceConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        pass
    
    def get_cache(self, key: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        conn = sqlite3.connect(self.config.cache_db)
        c = conn.cursor()
        c.execute('''SELECT value, timestamp FROM cache WHERE key = ?''', (key,))
        result = c.fetchone()
        conn.close()
        
        if result:
            value, timestamp = result
            cache_time = datetime.fromisoformat(timestamp)
            if datetime.now() - cache_time < timedelta(days=1):
                return value
        return None
    
    def set_cache(self, key: str, value: str):
        """Cache response with timestamp"""
        conn = sqlite3.connect(self.config.cache_db)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO cache VALUES (?, ?, ?)''',
                 (key, value, datetime.now().isoformat()))
        conn.commit()
        conn.close()

class MistralService(BaseAIService):
    async def generate_text(self, prompt: str, **kwargs) -> str:
        cache_key = f"mistral_{prompt}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        headers = {
            "Authorization": f"Bearer {self.config.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": kwargs.get("model", "mistral-tiny"),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with self.client as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            
        self.set_cache(cache_key, result)
        return result

class DeepseekService(BaseAIService):
    async def generate_text(self, prompt: str, **kwargs) -> str:
        cache_key = f"deepseek_{prompt}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": kwargs.get("model", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with self.client as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            
        self.set_cache(cache_key, result)
        return result

class HuggingFaceService(BaseAIService):
    async def generate_text(self, prompt: str, **kwargs) -> str:
        cache_key = f"hf_{prompt}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        headers = {
            "Authorization": f"Bearer {self.config.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        model_id = kwargs.get("model", "mistralai/Mistral-7B-v0.1")
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        async with self.client as client:
            response = await client.post(
                api_url,
                headers=headers,
                json={"inputs": prompt}
            )
            response.raise_for_status()
            result = response.json()[0]["generated_text"]
            
        self.set_cache(cache_key, result)
        return result

class OllamaService(BaseAIService):
    async def generate_text(self, prompt: str, **kwargs) -> str:
        cache_key = f"ollama_{prompt}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        data = {
            "model": kwargs.get("model", "mistral"),
            "prompt": prompt
        }
        
        async with self.client as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=data
            )
            response.raise_for_status()
            result = response.json()["response"]
            
        self.set_cache(cache_key, result)
        return result

class ThaiModelService(BaseAIService):
    """Service for Thai-specific language models from Hugging Face"""
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        cache_key = f"thai_{kwargs.get('model', 'default')}_{prompt}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        model_name = kwargs.get("model", "wangchanberta")
        # Use the predefined model ID or fallback to the model name itself
        model_id = self.config.thai_models.get(model_name, model_name)
        
        headers = {
            "Authorization": f"Bearer {self.config.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Different payload structure for Thai models
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "thai_target": True  # Flag for Thai-specific processing
            }
        }
        
        try:
            async with self.client as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        output = result[0]["generated_text"]
                    else:
                        output = result[0]
                else:
                    output = str(result)
                    
                self.set_cache(cache_key, output)
                return output
        except Exception as e:
            print(f"Error with Thai model {model_id}: {e}")
            return f"เกิดข้อผิดพลาดในการเรียกใช้โมเดล: {str(e)}"

class AIServiceManager:
    def __init__(self, config: AIServiceConfig):
        self.config = config
        self.mistral = MistralService(config)
        self.deepseek = DeepseekService(config)
        self.huggingface = HuggingFaceService(config)
        self.ollama = OllamaService(config)
        self.thai = ThaiModelService(config)  # Add Thai model service
        
    async def generate_with_fallback(self, prompt: str, services: List[str], **kwargs) -> Optional[str]:
        """Try multiple services with fallback"""
        for service_name in services:
            try:
                service = getattr(self, service_name)
                return await service.generate_text(prompt, **kwargs)
            except Exception as e:
                print(f"Error with {service_name}: {str(e)}")
                continue
        return None

    def get_available_thai_models(self) -> Dict[str, str]:
        """Return available Thai models"""
        return self.config.thai_models

    async def generate_with_thai_model(self, prompt: str, model_name: str = "wangchanberta", **kwargs) -> Optional[str]:
        """Generate text using Thai-specific model"""
        try:
            return await self.thai.generate_text(prompt, model=model_name, **kwargs)
        except Exception as e:
            print(f"Error with Thai model {model_name}: {str(e)}")
            return None

    async def verify_thai_model_access(self, model_name: str) -> bool:
        """Verify access to Thai model"""
        try:
            test_prompt = "ทดสอบการเข้าถึงโมเดลภาษาไทย"
            result = await self.thai.generate_text(test_prompt, model=model_name)
            return bool(result)
        except:
            return False
