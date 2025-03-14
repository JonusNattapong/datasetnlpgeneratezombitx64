import os
import json
import time
import requests
import re
from typing import List, Dict, Any, Optional, Union, Tuple
import csv
from abc import ABC, abstractmethod
import pandas as pd

class MistralAIClient:
    """ตัวจัดการ API ของ Mistral AI"""
    
    def __init__(self, api_key: str, api_url: str = "https://api.mistral.ai/v1/chat/completions"):
        """
        ตัวแปลงเริ่มต้น Mistral AI API client
        
        Args:
            api_key (str): API Key สำหรับ Mistral AI
            api_url (str): URL ของ API
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """
        สร้างการตอบสนองจาก Mistral AI API
        
        Args:
            prompt (str): ข้อความสำหรับ AI
            system_prompt (str): ข้อความระบบเพื่อกำหนดบริบท
            model (str): รุ่น AI ที่จะใช้
            temperature (float): ค่าความสุ่ม (0-1)
            max_tokens (int): จำนวนโทเค็นสูงสุดในผลลัพธ์
            top_p (float): sampling parameter
            
        Returns:
            Dict[str, Any]: การตอบสนองจาก API
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        # ลบรายการ None
        payload["messages"] = [msg for msg in payload["messages"] if msg is not None]
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return {"error": str(e)}
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        แยกเนื้อหาข้อความจากการตอบสนอง API
        
        Args:
            response (Dict[str, Any]): การตอบสนอง API
            
        Returns:
            str: เนื้อหาข้อความจากการตอบสนอง
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""
    
    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        แยก JSON จากการตอบสนองข้อความ
        
        Args:
            text (str): ข้อความตอบสนอง
            
        Returns:
            Dict[str, Any]: ข้อมูล JSON ที่แยกออกมา
        """
        try:
            # หา JSON ในข้อความ
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # ลองหา JSON ที่ไม่ได้อยู่ใน code block
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return {}
                
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            print(f"ไม่สามารถแยก JSON จากการตอบสนอง: {text}")
            return {}

# ตัวสร้างชุดข้อมูลพื้นฐาน
class DatasetGenerator(ABC):
    """คลาสพื้นฐานสำหรับตัวสร้างชุดข้อมูล NLP"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets",
        task_name: str = "default_task"
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูล
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูลที่สร้างขึ้น
            task_name (str): ชื่องาน NLP
        """
        self.mistral_client = mistral_client
        self.output_dir = os.path.join(output_dir, task_name)
        self.task_name = task_name
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงานนี้"""
        pass
    
    @abstractmethod
    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        pass
    
    @abstractmethod
    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        pass
    
    def generate_sample(self, text: str) -> Dict[str, Any]:
        """
        สร้างตัวอย่างชุดข้อมูลหนึ่งรายการ
        
        Args:
            text (str): ข้อความต้นฉบับสำหรับการสร้างตัวอย่าง
            
        Returns:
            Dict[str, Any]: ตัวอย่างชุดข้อมูลที่สร้างขึ้น
        """
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(text)
        
        response = self.mistral_client.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        response_text = self.mistral_client.extract_content(response)
        return self.process_response(response_text)
    
    def generate_dataset(
        self, 
        texts: List[str], 
        num_samples: int = 100,
        batch_size: int = 10,
        sleep_time: int = 1
    ) -> List[Dict[str, Any]]:
        """
        สร้างชุดข้อมูลจากรายการข้อความ
        
        Args:
            texts (List[str]): รายการข้อความสำหรับการสร้างชุดข้อมูล
            num_samples (int): จำนวนตัวอย่างที่ต้องการสร้าง
            batch_size (int): ขนาดกลุ่มที่จะสร้างก่อนหยุดพัก
            sleep_time (int): เวลาพักระหว่างแต่ละกลุ่ม (วินาที)
            
        Returns:
            List[Dict[str, Any]]: ชุดข้อมูลที่สร้างขึ้น
        """
        samples = []
        count = 0
        
        # สลับรายการข้อความเพื่อความหลากหลาย
        import random
        random.shuffle(texts)
        
        for i, text in enumerate(texts):
            if count >= num_samples:
                break
                
            try:
                sample = self.generate_sample(text)
                samples.append(sample)
                count += 1
                
                print(f"สร้างตัวอย่าง {count}/{num_samples}")
                
                # พักระหว่างกลุ่ม
                if count % batch_size == 0 and count < num_samples:
                    print(f"พัก {sleep_time} วินาที...")
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการสร้างตัวอย่าง: {e}")
                continue
                
        return samples
    
    def save_dataset(self, samples: List[Dict[str, Any]], format: str = "json") -> str:
        """
        บันทึกชุดข้อมูลในรูปแบบที่ต้องการ
        
        Args:
            samples (List[Dict[str, Any]]): ชุดข้อมูล
            format (str): รูปแบบที่ต้องการ (json, csv, text)
            
        Returns:
            str: เส้นทางไฟล์ที่บันทึก
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.task_name}_{timestamp}"
        
        if format == "json":
            output_path = os.path.join(self.output_dir, f"{filename}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
        elif format == "csv":
            output_path = os.path.join(self.output_dir, f"{filename}.csv")
            if samples:
                df = pd.DataFrame(samples)
                df.to_csv(output_path, index=False, encoding='utf-8')
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("")
                    
        elif format == "text":
            output_path = os.path.join(self.output_dir, f"{filename}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    
        else:
            raise ValueError(f"รูปแบบไม่รองรับ: {format}")
            
        print(f"บันทึกชุดข้อมูลไปยัง {output_path}")
        return output_path


# ตัวอย่างตัวสร้างชุดข้อมูลการจัดประเภทข้อความ
class TextClassificationGenerator(DatasetGenerator):
    """ตัวสร้างชุดข้อมูลสำหรับการจัดประเภทข้อความ"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets",
        class_names: List[str] = None
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูลการจัดประเภทข้อความ
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูล
            class_names (List[str]): รายชื่อคลาส
        """
        super().__init__(mistral_client, output_dir, "text_classification")
        self.class_names = class_names or ["positive", "negative", "neutral"]
    
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงาน"""
        return f"""คุณเป็นผู้เชี่ยวชาญในการสร้างชุดข้อมูลสำหรับการจัดประเภทข้อความ
คุณจะสร้างตัวอย่างที่เกี่ยวข้องกับประเภทต่อไปนี้: {', '.join(self.class_names)}

ระบุประเภทที่เหมาะสมสำหรับข้อความที่กำหนดให้
ตอบในรูปแบบ JSON เท่านั้น ใช้ฟอร์แมตนี้:
{{
        "text": "ข้อความดั้งเดิม",
    "label": "ชื่อคลาส",
    "label_id": คลาส ID (ตัวเลข),
    "confidence": ความมั่นใจ (0-1)
}}"""

    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        return f"""ช่วยจัดประเภทข้อความต่อไปนี้เป็นหนึ่งใน {', '.join(self.class_names)}:

ข้อความ: {text}

โปรดตอบกลับด้วย JSON ตามรูปแบบที่กำหนด"""

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        try:
            data = self.mistral_client.parse_json_response(response_text)
            
            # ตรวจสอบว่ามีฟิลด์ที่จำเป็นครบถ้วน
            if not all(key in data for key in ['text', 'label', 'label_id']):
                # สร้างข้อมูลเริ่มต้นหากไม่มีข้อมูลที่จำเป็น
                return {'error': 'Missing required fields in response'}
                
            return data
        except Exception as e:
            print(f"ข้อผิดพลาดในการประมวลผลการตอบสนอง: {e}")
            return {'error': str(e)}


# ตัวสร้างชุดข้อมูลสำหรับ NER (ชื่อเฉพาะ)
class TokenClassificationGenerator(DatasetGenerator):
    """ตัวสร้างชุดข้อมูลสำหรับการจัดประเภทโทเค็น (NER)"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets",
        entity_types: List[str] = None
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูล NER
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูล
            entity_types (List[str]): ประเภทเอนทิตี้ที่ต้องการ (เช่น PER, ORG)
        """
        super().__init__(mistral_client, output_dir, "token_classification")
        self.entity_types = entity_types or ["PER", "ORG", "LOC", "DATE"]
    
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงาน"""
        return f"""คุณเป็นผู้เชี่ยวชาญในการสร้างชุดข้อมูลสำหรับการรู้จำชื่อเฉพาะ (NER)
ประเภทเอนทิตี้ที่ต้องการคือ: {', '.join(self.entity_types)}

สำหรับข้อความที่กำหนด ให้ระบุโทเค็นและประเภทเอนทิตี้ที่เกี่ยวข้อง
ตอบในรูปแบบ JSON เท่านั้น ใช้ฟอร์แมตนี้:
{{
    "text": "ข้อความดั้งเดิม",
    "tokens": ["โทเค็น1", "โทเค็น2", ...],
    "tags": ["O", "B-PER", "I-PER", ...],
    "entities": [
        {{"entity": "ชื่อเอนทิตี้", "type": "ประเภท", "start": ตำแหน่งเริ่มต้น, "end": ตำแหน่งสิ้นสุด}}
    ]
}}"""

    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        return f"""ช่วยระบุชื่อเฉพาะในข้อความต่อไปนี้:

ข้อความ: {text}

โปรดตอบกลับด้วย JSON ตามรูปแบบที่กำหนด แสดงโทเค็น ประเภท และตำแหน่งของเอนทิตี้"""

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        try:
            return self.mistral_client.parse_json_response(response_text)
        except Exception as e:
            print(f"ข้อผิดพลาดในการประมวลผลการตอบสนอง: {e}")
            return {'error': str(e)}


# ตัวสร้างชุดข้อมูลสำหรับการตอบคำถาม
class QuestionAnsweringGenerator(DatasetGenerator):
    """ตัวสร้างชุดข้อมูลสำหรับการตอบคำถาม"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets"
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูลการตอบคำถาม
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูล
        """
        super().__init__(mistral_client, output_dir, "question_answering")
    
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงาน"""
        return """คุณเป็นผู้เชี่ยวชาญในการสร้างชุดข้อมูลสำหรับการตอบคำถาม

สำหรับบริบทที่กำหนด ให้สร้างคู่คำถาม-คำตอบที่สมจริง
ตอบในรูปแบบ JSON เท่านั้น ใช้ฟอร์แมตนี้:
{
    "context": "บริบทดั้งเดิม",
    "questions": [
        {
            "question": "คำถาม",
            "answer": "คำตอบที่สกัดจากบริบท",
            "answer_start": ตำแหน่งเริ่มต้นของคำตอบในบริบท
        }
    ]
}"""

    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        return f"""ช่วยสร้างชุดข้อมูลการตอบคำถามจากบริบทต่อไปนี้:

บริบท: {text}

โปรดสร้างชุดคำถาม-คำตอบ 3-5 คู่ที่เกี่ยวข้องกับบริบทข้างต้น ระบุตำแหน่งเริ่มต้นของคำตอบในบริบท (index ของอักขระ)"""

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        try:
            return self.mistral_client.parse_json_response(response_text)
        except Exception as e:
            print(f"ข้อผิดพลาดในการประมวลผลการตอบสนอง: {e}")
            return {'error': str(e)}


# ตัวสร้างชุดข้อมูลสำหรับการสรุปความ
class SummarizationGenerator(DatasetGenerator):
    """ตัวสร้างชุดข้อมูลสำหรับการสรุปความ"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets"
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูลการสรุปความ
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูล
        """
        super().__init__(mistral_client, output_dir, "summarization")
    
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงาน"""
        return """คุณเป็นผู้เชี่ยวชาญในการสร้างชุดข้อมูลสำหรับการสรุปความ

สำหรับข้อความที่กำหนด ให้สร้างสรุปที่สั้นและกระชับ
ตอบในรูปแบบ JSON เท่านั้น ใช้ฟอร์แมตนี้:
{
    "article": "บทความดั้งเดิม",
    "summary": "บทสรุปสั้น (1-3 ประโยค)",
    "metadata": {
        "compression_ratio": อัตราส่วนการบีบอัด (ความยาวสรุป/ความยาวบทความ),
        "key_points": ["ประเด็นสำคัญที่ 1", "ประเด็นสำคัญที่ 2", ...]
    }
}"""

    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        return f"""ช่วยสร้างสรุปสำหรับข้อความต่อไปนี้:

บทความ: {text}

โปรดสร้างสรุปที่สั้นและกระชับ (1-3 ประโยค) และระบุประเด็นสำคัญ 2-3 ข้อ"""

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        try:
            return self.mistral_client.parse_json_response(response_text)
        except Exception as e:
            print(f"ข้อผิดพลาดในการประมวลผลการตอบสนอง: {e}")
            return {'error': str(e)}


# ตัวสร้างชุดข้อมูลสำหรับการแปลภาษา
class TranslationGenerator(DatasetGenerator):
    """ตัวสร้างชุดข้อมูลสำหรับการแปลภาษา"""
    
    def __init__(
        self, 
        mistral_client: MistralAIClient,
        output_dir: str = "generated_datasets",
        source_lang: str = "en",
        target_lang: str = "th"
    ):
        """
        ตัวแปลงเริ่มต้นตัวสร้างชุดข้อมูลการแปลภาษา
        
        Args:
            mistral_client (MistralAIClient): ตัวลูกค้า Mistral AI API
            output_dir (str): ไดเรกทอรีสำหรับบันทึกชุดข้อมูล
            source_lang (str): รหัสภาษาต้นทาง (เช่น 'en')
            target_lang (str): รหัสภาษาปลายทาง (เช่น 'th')
        """
        super().__init__(mistral_client, output_dir, f"translation_{source_lang}_to_{target_lang}")
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def generate_system_prompt(self) -> str:
        """สร้าง system prompt สำหรับงาน"""
        return f"""คุณเป็นผู้เชี่ยวชาญด้านการแปลจากภาษา {self.source_lang} เป็นภาษา {self.target_lang}

สำหรับข้อความที่กำหนด ให้สร้างคู่ประโยคแปล
ตอบในรูปแบบ JSON เท่านั้น ใช้ฟอร์แมตนี้:
{{
    "translation": {{
        "{self.source_lang}": "ข้อความภาษาต้นทาง",
        "{self.target_lang}": "ข้อความภาษาปลายทาง"
    }},
    "metadata": {{
        "quality": ระดับคุณภาพการแปล (1-5),
        "difficulty": ระดับความยาก (1-5),
        "register": "รูปแบบภาษา (formal/informal/technical)"
    }}
}}"""

    def generate_user_prompt(self, text: str) -> str:
        """สร้าง prompt สำหรับผู้ใช้"""
        return f"""ช่วยแปลข้อความต่อไปนี้จากภาษา {self.source_lang} เป็นภาษา {self.target_lang}:

ข้อความ: {text}"""

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """ประมวลผลการตอบสนองจาก API"""
        try:
            return self.mistral_client.parse_json_response(response_text)
        except Exception as e:
            print(f"ข้อผิดพลาดในการประมวลผลการตอบสนอง: {e}")
            return {'error': str(e)}